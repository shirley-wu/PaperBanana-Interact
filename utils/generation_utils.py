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
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

from __future__ import annotations
import asyncio
import base64
import random
from typing import List, Dict, Any

import httpx
from google import genai
from google.genai import errors, types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os
from contextvars import ContextVar

import yaml
from pathlib import Path

from .config import ExpConfig

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8-sig") as f:
        model_config = yaml.safe_load(f) or {}


# Per-request credential overrides, keyed like the yaml `api_keys` section (e.g. {"google_api_key": "..."}).
# Set inside an async handler: the asyncio tasks it spawns inherit it; other sessions never see it.
api_key_overrides: ContextVar[dict] = ContextVar("api_key_overrides", default={})


def get_config_val(section, key, env_var, default=""):
    val = api_key_overrides.get().get(key) or os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default


_PROVIDER_KEYS = {
    "gemini": ("api_keys", "google_api_key", "GOOGLE_API_KEY"),
    "anthropic": ("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY"),
    "openai": ("api_keys", "openai_api_key", "OPENAI_API_KEY"),
    "openrouter": ("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY"),
}


def provider_key(provider: str) -> str:
    """Resolved API key for a provider: session override > env > yaml ('' if none)."""
    return get_config_val(*_PROVIDER_KEYS[provider])


# Initialize clients lazily or with robust defaults
gemini_client = None
anthropic_client = None
openai_client = None
openrouter_client = None
openrouter_api_key = ""


# Vertex/Gemini fault-tolerance settings. ExpConfig binds these before requests.
_vertex_max_attempts = ExpConfig.vertex_max_attempts
_vertex_max_backoff = ExpConfig.vertex_max_backoff
_vertex_timeout_ms = ExpConfig.vertex_timeout_ms


def configure_vertex_requests(*, max_attempts: int, max_backoff: float, timeout_ms: int):
    """Bind Vertex request settings from the active experiment configuration."""
    global _vertex_max_attempts, _vertex_max_backoff, _vertex_timeout_ms
    _vertex_max_attempts = max_attempts
    _vertex_max_backoff = max_backoff
    _vertex_timeout_ms = timeout_ms


# Statuses worth retrying; any other API status (400/401/403/404 ...) is raised immediately.
_RETRY_STATUS_CODES = [429, 499, 500, 502, 503, 504]


def vertex_http_options():
    """Build shared Gemini/Vertex HTTP options from the experiment config."""
    return types.HttpOptions(
        timeout=_vertex_timeout_ms,
        retry_options=types.HttpRetryOptions(
            initial_delay=2.0,
            attempts=_vertex_max_attempts,
            http_status_codes=_RETRY_STATUS_CODES,
        ),
    )


def reinitialize_clients():
    """(Re)build all API clients from current env vars / config file.

    Called once at module load and can be called again at runtime
    (e.g. after the user sets new API keys via the Gradio UI).

    Returns a list of client names that were successfully initialized.
    """
    global gemini_client, anthropic_client, openai_client
    global openrouter_client, openrouter_api_key

    initialized = []

    api_key = provider_key("gemini")
    project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")

    if api_key:
        gemini_client = genai.Client(api_key=api_key, http_options=vertex_http_options())
        print("Initialized Gemini Client with API Key")
        initialized.append("Gemini")
    elif project_id:
        location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
        gemini_client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=vertex_http_options(),
        )
        print(f"Initialized Gemini Client via Vertex AI (Project: {project_id}, Location: {location})")
        initialized.append("Gemini (Vertex AI)")
    else:
        gemini_client = None

    key = provider_key("anthropic")
    if key:
        anthropic_client = AsyncAnthropic(api_key=key)
        print("Initialized Anthropic Client with API Key")
        initialized.append("Anthropic")
    else:
        anthropic_client = None

    key = provider_key("openai")
    if key:
        openai_client = AsyncOpenAI(api_key=key)
        print("Initialized OpenAI Client with API Key")
        initialized.append("OpenAI")
    else:
        openai_client = None

    openrouter_api_key = provider_key("openrouter")
    if openrouter_api_key:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        print("Initialized OpenRouter Client with API Key")
        initialized.append("OpenRouter")
    else:
        openrouter_client = None

    return initialized


# Run once at import time (preserves original behaviour)
reinitialize_clients()


# Single global limiter for ALL outbound model API calls (text + image), shared
# across every module. Acquired only at the leaf call sites below, so it is never
# nested with itself -> no deadlock. Set API_CONCURRENCY to tune.
_GLOBAL_API_SEM = asyncio.Semaphore(int(os.getenv("API_CONCURRENCY", "64")))


def set_api_concurrency(n: int):
    """Replace the global API concurrency limiter.

    Call once at startup (e.g. from argparse) BEFORE any API calls are issued.
    Functions read _GLOBAL_API_SEM at call time, so reassigning it here takes effect.
    """
    global _GLOBAL_API_SEM
    _GLOBAL_API_SEM = asyncio.Semaphore(n)


def _convert_to_gemini_content(contents: List[Dict[str, Any]] | types.Content, role: str = "user") -> types.Content:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    if isinstance(contents, types.Content):
        return contents

    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["image_base64"]),
                        mime_type="image/jpeg",
                    )
                )
    return types.Content(role=role, parts=gemini_parts)


async def call_gemini_with_retry_async(*args, **kwargs):
    # Public entry: hold the global API concurrency limiter for the full retry loop,
    # so every caller (dispatcher, agents, eval toolkits) is gated uniformly.
    async with _GLOBAL_API_SEM:
        return await _call_gemini_with_retry_async_impl(*args, **kwargs)


async def _call_gemini_with_retry_async_impl(
    model_name,
    contents,
    config,
    max_attempts=5,
    retry_delay=5,
    error_context="",
    return_content=False,
    multiturn_input=False,
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    if not (provider_key("gemini") or get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")):
        raise RuntimeError(
            "Gemini client was not initialized. Please set GOOGLE_API_KEY / GOOGLE_CLOUD_PROJECT in environment, "
            "or configure api_keys.google_api_key / google_cloud.project_id in configs/model_config.yaml."
        )

    # Enforce the configured tolerance floor for every Vertex request.
    max_attempts = max(max_attempts, _vertex_max_attempts)

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    response = None
    for attempt in range(max_attempts):
        try:
            # Instantiate on-demand client to bind to the current event loop
            api_key = provider_key("gemini")
            if api_key:
                client = genai.Client(api_key=api_key, http_options=vertex_http_options())
            else:
                project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
                location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                    http_options=vertex_http_options(),
                )

            # Convert generic content list to Gemini's format right before the API call
            if multiturn_input:
                gemini_contents = [_convert_to_gemini_content(turn) for turn in current_contents]
            else:
                gemini_contents = _convert_to_gemini_content(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if "nanoviz" in model_name or "image" in model_name:
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    delay = random.uniform(0, min(retry_delay * (2**attempt), _vertex_max_backoff))
                    print(f"Attempt {attempt + 1} failed to generate image, retrying in {delay:.0f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(base64.b64encode(part.inline_data.data).decode("utf-8"))
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                    if part.text is not None
                ]
            result_list.extend([r for r in raw_response_list if r and r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            if isinstance(e, errors.APIError) and e.code not in _RETRY_STATUS_CODES:
                raise
            context_msg = f" for {error_context}" if error_context else ""

            # Exponential backoff with full jitter, capped at the configured max backoff.
            current_delay = random.uniform(0, min(retry_delay * (2**attempt), _vertex_max_backoff))

            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay:.0f} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"⚠️ Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    if not return_content:
        return result_list
    else:
        content_list = []
        if response is not None:
            content_list.extend([candidate.content for candidate in response.candidates])
        content_list.extend([None] * (target_candidate_count - len(content_list)))
        return result_list, content_list


def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.

    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]

    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({"type": "image_url", "image_url": {"url": data_url}})
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                openai_contents.append({"type": "image_url", "image_url": {"url": data_url}})
    return openai_contents


async def call_claude_with_retry_async(*args, **kwargs):
    # Public entry: hold the global API concurrency limiter for the full retry loop.
    async with _GLOBAL_API_SEM:
        return await _call_claude_with_retry_async_impl(*args, **kwargs)


async def _call_claude_with_retry_async_impl(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            # Instantiate on-demand client
            key = provider_key("anthropic")
            local_anthropic_client = AsyncAnthropic(api_key=key) if key else anthropic_client

            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await local_anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(f"⚠️ Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors.")
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(f"Input validated. Now generating remaining {remaining_candidates} candidates...")
        key = provider_key("anthropic")
        local_anthropic_client = AsyncAnthropic(api_key=key) if key else anthropic_client

        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            local_anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": valid_claude_contents}],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list


async def call_openai_with_retry_async(*args, **kwargs):
    # Public entry: hold the global API concurrency limiter for the full retry loop.
    async with _GLOBAL_API_SEM:
        return await _call_openai_with_retry_async_impl(*args, **kwargs)


async def _call_openai_with_retry_async_impl(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            # Instantiate on-demand client
            key = provider_key("openai")
            local_openai_client = AsyncOpenAI(api_key=key) if key else openai_client

            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await local_openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            content = first_response.choices[0].message.content or ""
            if not content.strip():
                print(f"OpenAI returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(f"⚠️ Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors.")
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(f"Input validated. Now generating remaining {remaining_candidates} candidates...")
        key = provider_key("openai")
        local_openai_client = AsyncOpenAI(api_key=key) if key else openai_client

        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            local_openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name,
    prompt,
    config,
    max_attempts=5,
    retry_delay=30,
    error_context="",
    edit_image=None,
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")

    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }

    # Add GPT-Image specific parameters
    gen_params.update(
        {
            "quality": quality,
            "background": background,
            "output_format": output_format,
        }
    )

    for attempt in range(max_attempts):
        try:
            # Instantiate on-demand client
            key = provider_key("openai")
            local_openai_client = AsyncOpenAI(api_key=key) if key else openai_client

            async with _GLOBAL_API_SEM:
                if edit_image is not None:
                    response = await local_openai_client.images.edit(image=edit_image, **gen_params)
                else:
                    response = await local_openai_client.images.generate(**gen_params)

            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"⚠️ Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_openrouter_with_retry_async(*args, **kwargs):
    # Public entry: hold the global API concurrency limiter for the full retry loop.
    async with _GLOBAL_API_SEM:
        return await _call_openrouter_with_retry_async_impl(*args, **kwargs)


async def _call_openrouter_with_retry_async_impl(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter API (OpenAI-compatible) with asynchronous retry logic.
    """
    if not provider_key("openrouter"):
        raise RuntimeError(
            "OpenRouter client was not initialized: missing API key. "
            "Please set OPENROUTER_API_KEY in environment, or configure "
            "api_keys.openrouter_api_key in configs/model_config.yaml."
        )

    model_name = _to_openrouter_model_id(model_name)
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents

    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            # Instantiate on-demand client
            key = provider_key("openrouter")
            local_openrouter_client = (
                AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key) if key else openrouter_client
            )

            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await local_openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            content = first_response.choices[0].message.content or ""
            if not content.strip():
                print(f"OpenRouter returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2**attempt), 60)
            print(
                f"OpenRouter attempt {attempt + 1} failed{context_msg}: {error_str}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)

    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        print(f"⚠️ Error: All {max_attempts} OpenRouter attempts failed{context_msg}.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        key = provider_key("openrouter")
        local_openrouter_client = (
            AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key) if key else openrouter_client
        )

        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            local_openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent OpenRouter candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


async def call_openrouter_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter image generation via direct httpx POST to avoid
    openai SDK issues with extra_body dropping the model field.
    Images are returned in choices[0].message.content as inline_data or
    in choices[0].message.images as data URLs.
    """
    key = provider_key("openrouter")
    if not key:
        raise RuntimeError("OpenRouter client was not initialized: missing API key.")

    system_prompt = config.get("system_prompt", "")
    temperature = config.get("temperature", 1.0)
    aspect_ratio = config.get("aspect_ratio", "1:1")
    image_size = config.get("image_size", "1k")

    model_name = _to_openrouter_model_id(model_name)
    openai_contents = _convert_to_openai_format(contents)

    image_config = {}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config["image_size"] = image_size

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": openai_contents},
        ],
        "temperature": temperature,
        "modalities": ["image", "text"],
    }
    if image_config:
        payload["image_config"] = image_config

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_attempts):
        try:
            async with _GLOBAL_API_SEM:
                async with httpx.AsyncClient(timeout=300) as client:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                print(f"[Warning]: OpenRouter image generation returned no choices, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

            message = choices[0].get("message", {})

            # Try extracting from inline_data in content (Gemini-style)
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "inline_data" in part:
                        b64_data = part["inline_data"].get("data", "")
                        if b64_data:
                            return [b64_data]

            # Try extracting from images field (OpenRouter standard)
            images = message.get("images")
            if images and len(images) > 0:
                img_item = images[0]
                if isinstance(img_item, dict):
                    data_url = img_item.get("image_url", {}).get("url", "")
                else:
                    data_url = str(img_item)
                if "," in data_url:
                    b64_data = data_url.split(",", 1)[1]
                else:
                    b64_data = data_url
                if b64_data:
                    return [b64_data]

            # Try extracting base64 from text content
            if isinstance(content, str) and content.startswith("data:image"):
                if "," in content:
                    b64_data = content.split(",", 1)[1]
                    if b64_data:
                        return [b64_data]

            print(f"[Warning]: OpenRouter image generation returned no images, retrying...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            continue

        except httpx.HTTPStatusError as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2**attempt), 60)
            print(
                f"OpenRouter image gen attempt {attempt + 1} failed{context_msg}: "
                f"HTTP {e.response.status_code} - {e.response.text}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"⚠️ Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2**attempt), 60)
            print(
                f"OpenRouter image gen attempt {attempt + 1} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"⚠️ Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


def _to_openrouter_model_id(model_name: str) -> str:
    """Convert a bare model name to OpenRouter format (provider/model).

    OpenRouter requires model IDs like 'google/gemini-3-pro-preview'.
    If the name already contains '/', assume it's already qualified.
    Otherwise, prefix with 'google/' for Gemini models.
    """
    if "/" in model_name:
        return model_name
    if model_name.startswith("gemini"):
        return f"google/{model_name}"
    return model_name


async def call_model_with_retry_async(
    model_name,
    contents,
    config,
    max_attempts=5,
    retry_delay=5,
    error_context="",
    return_content=False,
    multiturn_input=False,
):
    """
    Unified router that dispatches to the correct provider based on model_name.

    Routing rules:
      1. Explicit prefix overrides: "openrouter/" -> OpenRouter, "claude-" -> Anthropic,
         "gpt-"/"o1-"/"o3-"/"o4-" -> OpenAI
      2. No prefix: auto-detect based on which API key is configured.
         Priority: OpenRouter > Gemini > Anthropic > OpenAI
    """
    # Explicit provider prefix overrides auto-detection
    if model_name.startswith("openrouter/"):
        provider = "openrouter"
        actual_model = model_name[len("openrouter/") :]
    elif model_name.startswith("claude-"):
        provider = "anthropic"
        actual_model = model_name
    elif any(model_name.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
        provider = "openai"
        actual_model = model_name
    else:
        # Auto-detect provider based on which API key is configured
        actual_model = model_name
        if provider_key("openrouter"):
            provider = "openrouter"
            actual_model = _to_openrouter_model_id(model_name)
        elif provider_key("gemini") or get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", ""):
            provider = "gemini"
        elif provider_key("anthropic"):
            provider = "anthropic"
        elif provider_key("openai"):
            provider = "openai"
        else:
            raise RuntimeError(
                "No API client available. Please configure at least one API key "
                "in configs/model_config.yaml or via environment variables."
            )

    if provider == "gemini":
        # Concurrency gating now lives inside the public call_* functions.
        return await call_gemini_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
            return_content=return_content,
            multiturn_input=multiturn_input,
        )
    else:
        assert not return_content, "return_content not supported"
        assert not multiturn_input, "multiturn_input not supported"

    # Convert Gemini GenerateContentConfig -> dict for OpenAI/Claude/OpenRouter
    cfg_dict = {
        "system_prompt": (config.system_instruction if hasattr(config, "system_instruction") else ""),
        "temperature": config.temperature if hasattr(config, "temperature") else 1.0,
        "candidate_num": (config.candidate_count if hasattr(config, "candidate_count") else 1),
        "max_completion_tokens": (config.max_output_tokens if hasattr(config, "max_output_tokens") else 50000),
    }

    call_fn = {
        "openrouter": call_openrouter_with_retry_async,
        "anthropic": call_claude_with_retry_async,
        "openai": call_openai_with_retry_async,
    }[provider]

    # Concurrency gating now lives inside the public call_* functions.
    return await call_fn(
        model_name=actual_model,
        contents=contents,
        config=cfg_dict,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )
