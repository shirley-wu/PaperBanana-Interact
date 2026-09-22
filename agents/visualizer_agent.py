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
Vanilla Agent - Directly rendering images based on the method section.
"""

from typing import Dict, Any
from google.genai import types
import asyncio

from utils import generation_utils, image_utils
from .base_agent import BaseAgent


class VisualizerAgent(BaseAgent):
    """Visualizer Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = self.exp_config.image_gen_model_name
        self.system_prompt = DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT
        self.task_config = {
            "task_name": "diagram",  # this codebase only supports diagram
            "prompt_template": "Render an image based on the following detailed description: {desc}\n Note that do not include figure titles in the image. Diagram: ",
            "max_output_tokens": 50000,
        }

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified processing method that works for both diagram and plot tasks.
        Uses task_config to determine task-specific parameters.
        """
        cfg = self.task_config
        task_name = cfg["task_name"]

        desc_keys_to_process = []
        for key in [
            f"target_{task_name}_desc0",
            f"target_{task_name}_stylist_desc0",
        ]:
            if key in data and f"{key}_base64_jpg" not in data and f"{key}_gen_failed" not in data:
                desc_keys_to_process.append(key)

        # Dynamically find any critic_desc keys that lack a base64_jpg
        for key in list(data.keys()):
            if (
                key.startswith(f"target_{task_name}_critic_desc")
                and not key.endswith("_base64_jpg")
                and not key.endswith("_gen_failed")
            ):
                if f"{key}_base64_jpg" not in data and f"{key}_gen_failed" not in data:
                    suffix = key[len(f"target_{task_name}_critic_desc") :]
                    critic_suggestions_key = f"target_{task_name}_critic_suggestions{suffix}"
                    critic_suggestions = data.get(critic_suggestions_key, "")
                    if str(critic_suggestions).strip() == "No changes needed.":
                        # Converged round: the desc is a no-op, never render it.
                        continue
                    desc_keys_to_process.append(key)

        for desc_key in desc_keys_to_process:
            prompt_text = cfg["prompt_template"].format(desc=data[desc_key])
            content_list = [{"type": "text", "text": prompt_text}]

            gen_config_args = {
                "temperature": self.exp_config.temperature,
                "candidate_count": 1,
                "max_output_tokens": cfg["max_output_tokens"],
            }
            aspect_ratio = "1:1"
            if "additional_info" in data and "rounded_ratio" in data["additional_info"]:
                aspect_ratio = data["additional_info"]["rounded_ratio"]
            image_size = "1k"

            if "gpt-image" in self.model_name:
                image_config = {
                    "size": "1536x1024",
                    "quality": "high",
                    "background": "opaque",
                    "output_format": "png",
                }
                response_list = await generation_utils.call_openai_image_generation_with_retry_async(
                    model_name=self.model_name,
                    prompt=prompt_text,
                    config=image_config,
                    max_attempts=5,
                    retry_delay=30,
                )
            elif generation_utils.provider_key("openrouter"):
                # OpenRouter image generation
                image_config = {
                    "system_prompt": self.system_prompt,
                    "temperature": self.exp_config.temperature,
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_size,
                }
                response_list = await generation_utils.call_openrouter_image_generation_with_retry_async(
                    model_name=self.model_name,
                    contents=content_list,
                    config=image_config,
                    max_attempts=5,
                    retry_delay=30,
                )
            else:
                # Gemini direct image generation
                gen_config_args["response_modalities"] = ["IMAGE"]
                gen_config_args["image_config"] = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                )
                response_list = await generation_utils.call_gemini_with_retry_async(
                    model_name=self.model_name,
                    contents=content_list,
                    config=types.GenerateContentConfig(**gen_config_args),
                    max_attempts=5,
                    retry_delay=30,
                )

            if not response_list or not response_list[0]:
                continue

            # Convert PNG to JPG
            converted_jpg = await asyncio.to_thread(image_utils.convert_png_b64_to_jpg_b64, response_list[0])
            if converted_jpg:
                data[f"{desc_key}_base64_jpg"] = converted_jpg
            else:
                # Mark as failed so we don't retry this buggy prompt every turn.
                data[f"{desc_key}_gen_failed"] = True
                print(f"⚠️  Skipping {desc_key}: image conversion failed")

        return data


DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert scientific diagram illustrator. Generate high-quality scientific diagrams based on user requests."""
