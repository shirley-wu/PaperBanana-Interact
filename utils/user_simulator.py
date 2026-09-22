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

import copy

from google.genai import types

from . import generation_utils
from . import user_simulator_multi_req
from .base64_rw_helper import read_base64

SYSTEM_PROMPT = r"""
You are role-playing as the (non-expert) user who asked an assistant to create a figure for your research paper. You are now looking at the latest version of the figure and you are not fully happy with it.

You are NOT a designer. You do not know the technical vocabulary of visual design, and you should not sound like an expert reviewer. You just roughly know what you wanted, and you can tell when something looks off.

## Your task
Look at the current figure and the note about what is wrong with it, then write ONE short, casual message asking for that one thing to be fixed. React like a real, slightly impatient user — not a careful, exhaustive reviewer.

## What you are given
- Current figure: the image attached below.
- What's wrong: an internal evaluation explaining how the figure currently fails one specific requirement. Use this ONLY to understand what is actually wrong. Do NOT quote it or mention that an evaluation exists.
- What you wanted: the underlying requirement you originally cared about. This is the ground truth for what to ask for. Note the assistant never saw this spelled out — it only worked from the method section and caption below — so unless you already raised this in an earlier turn, it had no way of knowing it mattered to you.
- Conversation so far: earlier turns between you ("User") and the assistant. Each of your past messages may be followed by an internal note recording which requirement you raised and why — use these only to remember what you already complained about; never quote or mention them.
- Paper context (method section & figure caption): background only, so your complaint makes sense.

## How to write the feedback
- Keep it short: **AT MOST TWO** sentences, casual and conversational.
- Raise only the single issue described above. Do not list multiple problems.
- Talk like a normal person ("the arrows are kind of confusing", "can you make the boxes bigger?"), not like a designer ("increase the stroke weight", "fix the visual hierarchy").
- Describe the problem or what you want, not the exact design fix. Don't be overly specific or prescriptive.
- Don't mention the evaluation, the requirements, scores, rubrics, or that this is a simulation. Just talk about the picture.
- Usually don't repeat feedback you already gave in earlier turns. But sometimes you'll need to re-raise an issue you brought up before — either it was never really fixed, or it got fixed once and then broke again. When that happens, it's fine to sound a bit annoyed, like a real person would.

## Output
Return ONLY the user's message text — no quotes, no preamble, no explanation.
""".strip()


USER_TEMPLATE = r"""
## What you wanted (the requirement behind your complaint)
- Requirement: {requirement_description}

## What's wrong with the current figure
{eval_reasoning}

## Paper context (background only)

### Method section
{content}

### Figure caption
{visual_intent}

## Conversation so far
{chat_history}

Now write your next short, casual message as the User."""


def _format_chat_history(chat_history) -> str:
    """Render the chat history for the simulated user.

    Beyond the plain role/content turns (the way the feedback agent reads it),
    this also surfaces, for each past User turn, which requirement was raised
    and the internal evaluation reasoning behind it. This lets the simulated
    user remember what it already complained about and why.
    """
    if not chat_history:
        return "(no prior conversation)"
    lines = []
    for turn in chat_history:
        line = "{}: {}".format(turn["role"].capitalize(), turn["content"])
        lines.append(line)
        if turn.get("surfaced_index") is not None:
            lines.append(
                "    [Internal note — this message was about requirement "
                "#{}: {} | What looked off in that version: {}]\n".format(
                    turn["surfaced_index"],
                    turn["requirement_description"],
                    turn["eval_reasoning"],
                )
            )
    return "\n".join(lines)


async def simulate_user_feedback(prior_result: dict, exp_config) -> dict:
    # allow a setup for surfacing multiple req per turn
    if exp_config.n_surfaced_req_per_turn > 1:
        return await user_simulator_multi_req.simulate_user_feedback(prior_result, exp_config)

    # find the req to surface
    ind_to_surface = None
    for i in prior_result["req_order"]:
        if prior_result["req_eval_full"][i]["status_numerical"] == 0:
            ind_to_surface = i
            break
    if ind_to_surface is None:
        prior_result["early_stopped"] = True
        return prior_result

    eval_reasoning = prior_result["req_eval_full"][ind_to_surface]["req_eval_reasoning"]
    requirement_description = prior_result["requirements"][ind_to_surface]["description"]

    # format user text
    user_text = USER_TEMPLATE.format(
        requirement_description=requirement_description,
        eval_reasoning=eval_reasoning,
        content=prior_result["content"],
        visual_intent=prior_result["visual_intent"],
        chat_history=_format_chat_history(prior_result.get("chat_history")),
    )
    eval_image_b64 = await read_base64(prior_result[prior_result["eval_image_field"]])
    content_list = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "data": eval_image_b64,
                "media_type": "image/jpeg",
            },
        },
        {"type": "text", "text": user_text},
    ]

    # call model
    response_list = await generation_utils.call_model_with_retry_async(
        model_name=exp_config.user_model_name,
        contents=content_list,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=exp_config.temperature,
            candidate_count=1,
            max_output_tokens=50000,
        ),
        max_attempts=5,
        retry_delay=5,
    )
    user_utterance = response_list[0].strip()

    # add to chat_hstory
    prior_result = copy.deepcopy(prior_result)
    if "chat_history" not in prior_result:
        prior_result["chat_history"] = []
    prior_result["chat_history"] += [
        {"role": "assistant", "content": "Diagram generated."},
        {
            "role": "user",
            "content": user_utterance,
            "surfaced_index": ind_to_surface,
            "requirement_description": requirement_description,
            "eval_reasoning": eval_reasoning,
        },
    ]
    return prior_result
