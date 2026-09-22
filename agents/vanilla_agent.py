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

"""
Vanilla Agent - Directly rendering images based on the method section and diagram caption,
or writing code to generate plots based on the raw data and plot caption.
"""

from typing import Dict, Any
from google.genai import types
import asyncio
import json

from utils import generation_utils, image_utils
from .base_agent import BaseAgent


class VanillaAgent(BaseAgent):
    """Vanilla Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = self.exp_config.image_gen_model_name
        self.system_prompt = DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT
        self.process_executor = None
        self.task_config = {
            "task_name": "diagram",
            "use_image_generation": True,  # Use image generation
            "content_label": "Method Section",
            "visual_intent_label": "Diagram Caption",
        }

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate image based on the user prompt.
        Supports both diagram (image generation) and plot (matplotlib code generation).
        """
        cfg = self.task_config

        raw_content = data["content"]
        content = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else raw_content
        visual_intent = data["visual_intent"]

        prompt_text = f"**{cfg['content_label']}**: {content}\n**{cfg['visual_intent_label']}**: {visual_intent}\n"
        if cfg["task_name"] == "diagram":
            prompt_text += "Note that do not include figure titles in the image."

        prompt_text += "**Generated Diagram**: "

        content_list = [{"type": "text", "text": prompt_text}]

        gen_config_args = {
            "system_instruction": self.system_prompt,
            "temperature": self.exp_config.temperature,
            "candidate_count": 1,
            "max_output_tokens": 50000,
        }

        aspect_ratio = data["additional_info"]["rounded_ratio"]

        if "gpt-image" in self.model_name:
            image_config = {
                "size": "1536x1024",
                "quality": "high",
                "background": "opaque",
                "output_format": "png",
            }
            prompt_text = self.system_prompt + "\n\n" + prompt_text
            response_list = await generation_utils.call_openai_image_generation_with_retry_async(
                model_name=self.model_name,
                prompt=prompt_text[:30000],
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

        output_key = f"vanilla_{cfg['task_name']}_base64_jpg"
        data[output_key] = await asyncio.to_thread(image_utils.convert_png_b64_to_jpg_b64, response_list[0])
        return data


DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with a "Method Section" and a "Diagram Caption". Your task is to generate a high-quality scientific diagram that effectively illustrates the method described in the text, as the caption requires, and adhering strictly to modern academic visualization standards.

**CRITICAL INSTRUCTION ON CAPTION:**
The "Diagram Caption" is provided solely to describe the visual content and logic you need to draw. **DO NOT render, write, or include the caption text itself (e.g., "Figure 1: ...") inside the generated image.**

## INPUT DATA
-   **Method Section**: [Content of method section]
-   **Diagram Caption**: [Diagram caption]
## OUTPUT
Generate a single, high-resolution image that visually explains the method and aligns well with the caption. 
"""
