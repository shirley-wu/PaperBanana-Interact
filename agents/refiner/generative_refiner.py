# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from typing import Dict, Any
from io import BytesIO
import base64

from google.genai import types

from utils import generation_utils, image_utils
from utils.base64_rw_helper import read_base64
from ..base_agent import BaseAgent
from .pb_directrefine import format_chat_history


class GenerativeRefiner(BaseAgent):
    """Vanilla generative feedback agent for multi-turn refinement."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = self.exp_config.image_gen_model_name
        self.system_prompt = DIAGRAM_VANILLA_FEEDBACK_SYSTEM_PROMPT
        self.task_config = {
            "task_name": "diagram",
            "content_label": "Method Section",
            "visual_intent_label": "Diagram Caption",
        }

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Regenerate the figure directly from the most recent image plus a
        comprehensive feedback prompt, and set ``eval_image_field`` to the
        freshly generated image.
        """
        cfg = self.task_config
        task_name = cfg["task_name"]

        # Version bookkeeping so successive feedback turns don't overwrite.
        feedback_round = data.get("vanilla_feedback_round", 0) + 1
        data["vanilla_feedback_round"] = feedback_round

        # The most recent image to revise (set by the prior / initial turn).
        prev_image_field = data.get("eval_image_field")
        image_base64 = await read_base64(data.get(prev_image_field)) if prev_image_field else None

        raw_content = data["content"]
        content = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else raw_content
        visual_intent = data["visual_intent"]
        chat_history = format_chat_history(data.get("chat_history", []))

        prompt_text = (
            f"**{cfg['content_label']}**: {content}\n"
            f"**{cfg['visual_intent_label']}**: {visual_intent}\n"
            "Note that do not include figure titles in the image.\n"
            f"**Chat History**: {chat_history}\n"
        )

        content_list = [{"type": "text", "text": prompt_text}]
        # Attach the current image so the model can revise it in place.
        if image_base64 and len(image_base64) > 100:
            content_list.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "data": image_base64,
                        "media_type": "image/jpeg",
                    },
                }
            )
        else:
            print("⚠️ [Generative Refiner] No valid prior image found; regenerating from text only.")

        revised_prompt_text = prompt_text + "**Revised Diagram**: "
        content_list.append({"type": "text", "text": "**Revised Diagram**: "})

        gen_config_args = {
            "system_instruction": self.system_prompt,
            "temperature": self.exp_config.temperature,
            "candidate_count": 1,
            "max_output_tokens": 50000,
        }

        aspect_ratio = data.get("additional_info", {}).get("rounded_ratio", "1:1")

        if "gpt-image" in self.model_name:
            # gpt-image generation only accepts a text prompt (no input image).
            image_config = {
                "size": "1536x1024",
                "quality": "high",
                "background": "opaque",
                "output_format": "png",
            }
            revised_prompt_text = self.system_prompt + "\n\n" + revised_prompt_text

            if image_base64 and len(image_base64) > 100:
                image_file = BytesIO(base64.b64decode(image_base64))
                image_file.name = "input.jpg"
            else:
                image_file = None

            response_list = await generation_utils.call_openai_image_generation_with_retry_async(
                model_name=self.model_name,
                prompt=revised_prompt_text[:30000],
                edit_image=image_file,
                config=image_config,
                max_attempts=5,
                retry_delay=30,
            )
        elif generation_utils.provider_key("openrouter"):
            image_config = {
                "system_prompt": self.system_prompt,
                "temperature": self.exp_config.temperature,
                "aspect_ratio": aspect_ratio,
                "image_size": "1k",
            }
            response_list = await generation_utils.call_openrouter_image_generation_with_retry_async(
                model_name=self.model_name,
                contents=content_list,
                config=image_config,
                max_attempts=5,
                retry_delay=30,
            )
        else:
            gen_config_args["response_modalities"] = ["IMAGE"]
            gen_config_args["image_config"] = types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size="1k",
            )
            response_list = await generation_utils.call_gemini_with_retry_async(
                model_name=self.model_name,
                contents=content_list,
                config=types.GenerateContentConfig(**gen_config_args),
                max_attempts=5,
                retry_delay=30,
            )

        output_key = f"vanilla_{task_name}_feedback{feedback_round}_base64_jpg"
        new_image = None
        if response_list and response_list[0]:
            new_image = await asyncio.to_thread(image_utils.convert_png_b64_to_jpg_b64, response_list[0])

        if new_image:
            data[output_key] = new_image
            data["eval_image_field"] = output_key
        else:
            # Regeneration failed; keep the prior image as the eval target so
            # downstream evaluation still has a valid figure to score.
            print(
                f"⚠️ [Vanilla Feedback] Round {feedback_round} produced no valid "
                f"image. Keeping previous image '{prev_image_field}'."
            )

        return data


DIAGRAM_VANILLA_FEEDBACK_SYSTEM_PROMPT = """
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given the current version of a scientific diagram (attached image), the original request ('Method Section' and 'Diagram Caption'), and the iterative user feedback in the 'Chat History'. Your task is to generate a **revised** high-quality scientific diagram that preserves the correct elements from the current version and directly addresses the feedback given in the **latest** turn of the Chat History.

**CRITICAL INSTRUCTION ON CAPTION:**
The "Diagram Caption" is provided solely to describe the visual content and logic you need to draw. **DO NOT render, write, or include the caption text itself (e.g., "Figure 1: ...") inside the generated image.**

## INPUT DATA
-   **Current Diagram**: [The attached image to be revised]
-   **Method Section**: [Contextual content from the method section]
-   **Diagram Caption**: [Target figure caption]
-   **Chat History**: [List of chat utterances from all prior turns. Focus on addressing the **LATEST** turn]

## OUTPUT
Generate a single, high-resolution revised image that addresses the latest feedback while remaining faithful to the method and caption.
"""
