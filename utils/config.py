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

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import uuid


@dataclass
class ExpConfig:
    """Experiment configuration"""

    dataset_name: str
    split_name: str = "test"
    temperature: float = 1.0
    exp_mode: str = ""
    retrieval_setting: Literal["auto", "manual", "random", "none"] = "auto"
    max_critic_rounds: int = 3
    main_model_name: str = ""
    image_gen_model_name: str = ""

    # user simulation args
    user_model_name: str = ""
    n_surfaced_req_per_turn: int = 1  # >1 switches to the multi-req user simulator
    # refiner args
    refiner: str = "generative"
    refiner_internal_critic_rounds: int = 1

    # additional args: mostly for automatic save
    work_dir: Path = Path(__file__).parent.parent
    timestamp: str | None = None

    # llm calling args
    concurrent_num: int = 8
    vertex_max_attempts: int = 8
    vertex_max_backoff: float = 600.0
    vertex_timeout_ms: int = 20 * 60 * 1000

    @property
    def dataset_dir(self) -> Path:
        """Directory containing the selected dataset."""
        return self.work_dir / "data" / self.dataset_name

    def __post_init__(self):
        from . import generation_utils

        generation_utils.configure_vertex_requests(
            max_attempts=self.vertex_max_attempts,
            max_backoff=self.vertex_max_backoff,
            timeout_ms=self.vertex_timeout_ms,
        )

        os.environ["TZ"] = "America/Los_Angeles"  # set the timezone as you like
        if hasattr(time, "tzset"):
            time.tzset()  # Only available on Unix; no-op guard for Windows

        # Fallback to yaml config if no model name provided
        if not self.main_model_name or not self.image_gen_model_name:
            import yaml

            config_path = self.work_dir / "configs" / "model_config.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    model_config_data = yaml.safe_load(f) or {}
                    if not self.main_model_name:
                        self.main_model_name = model_config_data.get("defaults", {}).get("main_model_name", "")
                    if not self.image_gen_model_name:
                        self.image_gen_model_name = model_config_data.get("defaults", {}).get(
                            "image_gen_model_name", ""
                        )
        # Fallback to environment variables
        if not self.main_model_name:
            self.main_model_name = os.environ.get("MAIN_MODEL_NAME", "")
        if not self.image_gen_model_name:
            self.image_gen_model_name = os.environ.get("IMAGE_GEN_MODEL_NAME", "")
        # Hard defaults so model name is never empty
        if not self.main_model_name:
            self.main_model_name = "gemini-3.1-pro-preview"
            print(
                f"Warning: main_model_name not configured, falling back to '{self.main_model_name}'. "
                "Set it in configs/model_config.yaml or via --main-model-name."
            )
        if not self.image_gen_model_name:
            self.image_gen_model_name = "gemini-3.1-flash-image"
            print(
                f"Warning: image_gen_model_name not configured, falling back to '{self.image_gen_model_name}'. "
                "Set it in configs/model_config.yaml or via --image-gen-model-name."
            )
        self.timestamp = time.strftime("%m%d_%H%M") if self.timestamp is None else self.timestamp
        self.exp_name = (
            f"{self.timestamp}_{uuid.uuid4().hex[:8]}_{self.retrieval_setting}ret_{self.exp_mode}_{self.split_name}"
        )

        # mkdir result_dir if not exists
        self.result_dir = self.work_dir / "results" / f"{self.dataset_name}"
        self.result_dir.mkdir(exist_ok=True, parents=True)
