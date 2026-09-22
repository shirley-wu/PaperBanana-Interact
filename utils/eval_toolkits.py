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
import json_repair
import asyncio
import base64
import re
import difflib
from google.genai import types

from prompts import diagram_eval_prompts, req_eval_prompts
from utils.generation_utils import (
    call_gemini_with_retry_async,
    call_claude_with_retry_async,
    call_openai_with_retry_async,
)
from utils.base64_rw_helper import read_base64

# Prompt mapping: task_name -> eval_dim -> system_prompt
PROMPT_MAP = {
    "diagram": {
        "faithfulness": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_FAITHFULNESS_SYSTEM_PROMPT,
        "conciseness": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_CONCISENESS_SYSTEM_PROMPT,
        "readability": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_READABILITY_SYSTEM_PROMPT,
        "aesthetics": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_AESTHETICS_SYSTEM_PROMPT,
    },
}

# Task configuration: task_name -> field labels
TASK_CONFIG = {
    "diagram": {
        "visual_intent_label": "Diagram Caption",
        "raw_content_label": "Methodology Section",
        "human_label": "Human-Drawn Diagram (Human)",
        "model_label": "Model-Generated Diagram (Model)",
    },
}


def _try_regex_extract_winner(text: str) -> str | None:
    """Try to extract winner field using regex as a fallback."""
    patterns = [
        r'"winner"\s*:\s*"([^"]+)"',  # Standard JSON: "winner": "value"
        r'\*\*winner\*\*\s*:\s*"([^"]+)"',  # Markdown bold: **winner**: "value" or **winner**:"value"
        r"\*\*winner\*\*\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)",  # Markdown bold without quotes: **winner**: value (capture until comma, newline, or end)
        r'"winner"\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)',  # Mixed format: "winner": value (no quotes on value, capture until comma, newline, or end)
        r'(?:\*\*|")winner(?:\*\*|")\s*:\s*(?:\*\*|")?([A-Za-z][A-Za-z\s]+?)(?:\*\*|"|,|\n|$)',  # Very flexible: any winner marker followed by colon and value
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = value.rstrip('*"').strip()
            return value

    return None


def _extract_winner_with_fallback(clean_json: str, eval_dim: str, valid_winners: list[str]) -> str:
    """Try regex extraction and return winner or 'Error'."""
    extracted = _try_regex_extract_winner(clean_json)
    if extracted and extracted in valid_winners:
        print(f"ℹ️  {eval_dim}: regex extracted '{extracted}'")
        return extracted
    print(f"ℹ️  {eval_dim}: failed to extract valid winner")
    return "Error"


def _determine_tier_outcome_labels(
    dim1_outcome: str,
    dim2_outcome: str,
    win_labels: tuple[str, str] = ("Human", "Model"),
) -> str:
    """Determine the outcome for a tier given two dimension outcomes.

    Generic over the pair of "winner" labels so it works for both referenced
    comparison ("Human"/"Model") and pairwise comparison ("1"/"2"). The neutral
    outcomes ("Both are good"/"Both are bad") and the tie result are shared.
    """
    o1, o2 = dim1_outcome.strip(), dim2_outcome.strip()
    neutral = ["Both are good", "Both are bad"]

    # Both agree on a clear winner
    if o1 == o2:
        if o1 in neutral:
            return "Tie"
        return o1

    # One side wins, the other is neutral (Both are good/bad)
    for label in win_labels:
        if (o1 == label and o2 in neutral) or (o2 == label and o1 in neutral):
            return label

    # All other cases (conflicting winners, errors, etc.) -> Tie
    return "Tie"


def _determine_tier_outcome(dim1_outcome: str, dim2_outcome: str) -> str:
    """Determine the outcome for a tier (referenced comparison: Human vs Model)."""
    return _determine_tier_outcome_labels(dim1_outcome, dim2_outcome, win_labels=("Human", "Model"))


async def _run_single_eval_ref(
    task_name: str,
    eval_dim: str,
    raw_content: str,
    visual_intent: str,
    gt_image_base64: str,
    model_image_base64: str,
    model_name: str,
) -> tuple[str, dict]:
    """Run a single evaluation dimension for referenced comparison."""
    # Get the appropriate prompt based on task_name and eval_dim
    sys_prompt = PROMPT_MAP[task_name][eval_dim]

    # Get task-specific labels
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Invalid task name: {task_name}")

    config = TASK_CONFIG[task_name]

    # Construct input text based on eval dimension
    if eval_dim in ["readability", "aesthetics"]:
        input_text = f"{config['visual_intent_label']}: {visual_intent}\n{config['human_label']}: "
    else:
        input_text = f"{config['raw_content_label']}: {raw_content}\n{config['visual_intent_label']}: {visual_intent}\n{config['human_label']}: "

    # Construct content list
    content_list = [
        {"type": "text", "text": input_text},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": gt_image_base64,
            },
        },
        {"type": "text", "text": f"\n{config['model_label']}: "},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": model_image_base64,
            },
        },
    ]

    valid_winners = ["Human", "Model", "Both are good", "Both are bad"]

    try:
        if "gemini" in model_name:
            response_text_list = await call_gemini_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=1,
                    candidate_count=1,
                    max_output_tokens=50000,
                ),
            )
        elif "gpt" in model_name or "o1" in model_name or "o3" in model_name:
            response_text_list = await call_openai_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config={
                    "system_prompt": sys_prompt,
                    "temperature": 1,
                    "candidate_num": 1,
                    "max_completion_tokens": 10000,
                },
                max_attempts=5,
                retry_delay=30,
            )
        else:
            response_text_list = await call_claude_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config={
                    "system_prompt": sys_prompt,
                    "temperature": 1,
                    "candidate_num": 1,
                    "max_output_tokens": 10000,
                },
                max_attempts=5,
                retry_delay=30,
            )
        clean_json = response_text_list[0].replace("```json", "").replace("```", "").strip()
        res_obj = json_repair.loads(clean_json)

        if not isinstance(res_obj, dict):
            res_obj = {
                "comparison_reasoning": clean_json,
                "winner": _extract_winner_with_fallback(clean_json, eval_dim, valid_winners),
            }
        elif "winner" not in res_obj:
            res_obj["winner"] = _extract_winner_with_fallback(clean_json, eval_dim, valid_winners)
            if "comparison_reasoning" not in res_obj:
                res_obj["comparison_reasoning"] = clean_json

        return eval_dim, res_obj
    except Exception as e:
        print(f"❌ {eval_dim}: Evaluation failed - {str(e)[:100]}")
        extracted = _try_regex_extract_winner(clean_json) if "clean_json" in locals() else None
        winner = extracted if (extracted and extracted in valid_winners) else "Error"
        return eval_dim, {"comparison_reasoning": str(e), "winner": winner}


async def get_score_for_image_referenced(
    sample_data: dict,
    task_name: str = "diagram",
    model_name: str = "gemini-3.1-pro-preview",
    work_dir=None,
    dataset_name: str | None = None,
) -> dict:
    """Get score for diagram referenced comparison.

    Args:
        sample_data: Sample data dictionary
        task_name: Task name (diagram or plot)
        model_name: Model name for evaluation
        work_dir: Work directory path for resolving relative paths (pathlib.Path)
        dataset_name: Dataset directory name under ``work_dir/data``
    """
    from pathlib import Path

    raw_content = sample_data["content"]
    visual_intent = sample_data["visual_intent"]

    if "path_to_gt_image" not in sample_data:
        print("⚠️  No ground truth image path found. Skipping evaluation.")
        for dim in [
            "faithfulness",
            "conciseness",
            "readability",
            "aesthetics",
            "overall",
        ]:
            sample_data[f"{dim}_outcome"] = "N/A - No GT"
        return sample_data

    path_to_gt_image_rel = sample_data["path_to_gt_image"]

    # Resolve relative path using work_dir
    if work_dir and dataset_name:
        path_to_gt_image = work_dir / "data" / dataset_name / path_to_gt_image_rel
    else:
        # Fallback for backward compatibility (assume it's already absolute)
        path_to_gt_image = Path(path_to_gt_image_rel)

    with open(path_to_gt_image, "rb") as f:
        gt_image_base64 = base64.b64encode(f.read()).decode("utf-8")

    eval_image_field = sample_data["eval_image_field"]

    # Check if image was successfully generated
    if eval_image_field not in sample_data:
        print(f"⚠️  Image field '{eval_image_field}' not found. Model generation failed - counting as Human win.")
        # Model failed to generate image, Human wins by default
        for dim in [
            "faithfulness",
            "conciseness",
            "readability",
            "aesthetics",
            "overall",
        ]:
            sample_data[f"{dim}_reasoning"] = "Model failed to generate image - Human wins by default"
            sample_data[f"{dim}_outcome"] = "Human"
        return sample_data

    model_image_base64 = await read_base64(sample_data[eval_image_field])

    # Run evaluations for all dimensions
    dims = ["faithfulness", "conciseness", "readability", "aesthetics"]
    tasks = [
        _run_single_eval_ref(
            task_name,
            dim,
            raw_content,
            visual_intent,
            gt_image_base64,
            model_image_base64,
            model_name,
        )
        for dim in dims
    ]

    results = await asyncio.gather(*tasks)
    for eval_dim, res_obj in results:
        sample_data[f"{eval_dim}_reasoning"] = res_obj.get("comparison_reasoning", "")
        sample_data[f"{eval_dim}_outcome"] = res_obj.get("winner", "Unknown")

    faithfulness = sample_data.get("faithfulness_outcome", "Unknown")
    readability = sample_data.get("readability_outcome", "Unknown")
    conciseness = sample_data.get("conciseness_outcome", "Unknown")
    aesthetics = sample_data.get("aesthetics_outcome", "Unknown")

    # Tier 1: Faithfulness + Readability
    tier1_outcome = _determine_tier_outcome(faithfulness, readability)

    if tier1_outcome in ["Model", "Human"]:
        overall_outcome = tier1_outcome
        decision_path = f"Tier1({faithfulness}, {readability}) -> {tier1_outcome} [Decided at Tier 1]"
    else:
        # Tier 1 is tied, check Tier 2
        tier2_outcome = _determine_tier_outcome(conciseness, aesthetics)
        overall_outcome = tier2_outcome
        decision_path = f"Tier1({faithfulness}, {readability}) -> Tie; Tier2({conciseness}, {aesthetics}) -> {tier2_outcome} [Decided at Tier 2]"

    sample_data["overall_outcome"] = overall_outcome
    sample_data["overall_reasoning"] = f"Rule-based calculation: {decision_path}"

    return sample_data


# --- requirement satisfaction evaluation


def _parse_requirement_verdict_blocks(text: str) -> list[dict]:
    """Parse the requirement-eval response into ordered verdict blocks.

    The judge prompt (``DIAGRAM_REQ_EVAL_SYSTEM_PROMPT``) returns plain text
    (not JSON), repeating the following block once per requirement::

        Requirement: <verbatim requirement text>
        Rationale: <reasoning>
        Verdict: Satisfied|Unsatisfied

    We tolerate optional markdown bolding around the field labels / verdict.
    """
    pattern = re.compile(
        r"\**\s*Requirement\s*\**\s*:\s*(?P<req>.*?)\s*"
        r"\**\s*Rationale\s*\**\s*:\s*(?P<rationale>.*?)\s*"
        r"\**\s*Verdict\s*\**\s*:\s*\**\s*(?P<verdict>Satisfied|Unsatisfied)\b",
        re.IGNORECASE | re.DOTALL,
    )
    blocks = []
    for m in pattern.finditer(text):
        blocks.append(
            {
                "requirement": m.group("req").strip(),
                "rationale": m.group("rationale").strip(),
                "verdict": m.group("verdict").strip().capitalize(),
            }
        )
    return blocks


def _verdict_to_record(rationale: str, verdict: str) -> dict:
    """Build the per-requirement result dict from a rationale + verdict."""
    status = verdict.lower().strip()
    return {
        "req_eval_reasoning": rationale,
        "status": status,
        "status_numerical": 1 if status == "satisfied" else 0,
    }


def _similarity(a: str, b: str) -> float:
    """Case-insensitive character similarity ratio in [0, 1]."""
    return difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def _strict_match_requirements(requirements: list[dict], blocks: list[dict]) -> list[dict]:
    """Strictly align verdict blocks to requirements.

    Asserts the judge emitted exactly one block per requirement, in order, each
    re-quoting the original requirement nearly verbatim (>=95% character
    similarity, case-insensitive). Raises ``ValueError`` on any deviation so the
    caller can retry or fall back to soft matching.
    """
    if len(blocks) != len(requirements):
        raise ValueError(f"parsed {len(blocks)} verdict block(s) but expected " f"{len(requirements)} requirement(s)")

    res_list = []
    for req, block in zip(requirements, blocks):
        ratio = _similarity(req["description"], block["requirement"])
        if ratio < 0.95:
            raise ValueError(
                f"requirement re-quote mismatch (similarity {ratio:.3f} < 0.95):\n"
                f"  expected: {req['description'][:120]!r}\n"
                f"  got:      {block['requirement'][:120]!r}"
            )
        assert block["verdict"].lower().strip() in ["satisfied", "unsatisfied"]
        res_list.append(_verdict_to_record(block["rationale"], block["verdict"]))
    return res_list


def _soft_match_requirements(requirements: list[dict], blocks: list[dict]) -> list[dict]:
    """Best-effort salvage when strict matching fails.

    For each requirement we greedily pick the most similar still-unused verdict
    block. Requirements with no decent match (best similarity < 0.5) get a
    placeholder ``Error`` record. Always returns one record per requirement, in
    the original order.
    """
    if not blocks:
        print("ℹ️ℹ️ℹ️ User Requirements: EVEN soft match found no parseable verdict blocks")
        return [
            {
                "req_eval_reasoning": "Could not parse any verdict from judge response.",
                "status": "Error",
                "status_numerical": 0,
            }
            for _ in requirements
        ]

    used = set()
    res_list = []
    for req in requirements:
        best_idx, best_ratio = None, -1.0
        for idx, block in enumerate(blocks):
            if idx in used:
                continue
            ratio = _similarity(req["description"], block["requirement"])
            if ratio > best_ratio:
                best_idx, best_ratio = idx, ratio

        if best_idx is not None and best_ratio >= 0.5:
            used.add(best_idx)
            block = blocks[best_idx]
            res_list.append(_verdict_to_record(block["rationale"], block["verdict"]))
        else:
            res_list.append(
                {
                    "req_eval_reasoning": (
                        f"Soft match failed (best similarity {max(best_ratio, 0.0):.3f}); "
                        "no reliable verdict found for this requirement."
                    ),
                    "status": "Error",
                    "status_numerical": 0,
                }
            )
    return res_list


async def _run_batched_eval_requirement(
    raw_content: str,
    visual_intent: str,
    requirements: list[dict],
    model_image_base64: str,
    model_name: str,
    strict_match: bool = True,
) -> list[dict] | None:
    """Run a single evaluation dimension for referenced comparison."""
    # Get the appropriate prompt based on task_name and eval_dim
    sys_prompt = req_eval_prompts.DIAGRAM_REQ_EVAL_SYSTEM_PROMPT

    input_text = "<requirements>\n{}\n</requirements>\n\n<caption>\n{}\n</caption>\n\n<method>\n{}\n</method>".format(
        "\n".join(["* " + req["description"] for req in requirements]),
        visual_intent,
        raw_content,
    )

    # Construct content list
    content_list = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": model_image_base64,
            },
        },
        {"type": "text", "text": input_text},
    ]

    try:
        if "gemini" in model_name:
            response_text_list = await call_gemini_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=1,
                    candidate_count=1,
                    max_output_tokens=50000,
                ),
            )
        elif "gpt" in model_name or "o1" in model_name or "o3" in model_name:
            response_text_list = await call_openai_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config={
                    "system_prompt": sys_prompt,
                    "temperature": 1,
                    "candidate_num": 1,
                    "max_completion_tokens": 10000,
                },
                max_attempts=5,
                retry_delay=30,
            )
        else:
            response_text_list = await call_claude_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config={
                    "system_prompt": sys_prompt,
                    "temperature": 1,
                    "candidate_num": 1,
                    "max_output_tokens": 10000,
                },
                max_attempts=5,
                retry_delay=30,
            )

    except Exception as e:
        print(f"❌ User Requirements: Evaluation failed - {str(e)[:100]}")
        return None

    # Match the judge's plain-text response back against the requirements
    blocks = _parse_requirement_verdict_blocks(response_text_list[0])
    try:
        return _strict_match_requirements(requirements, blocks)
    except Exception as e:
        if strict_match:
            print(f"User Requirements: strict match failed - RETRY!\n{str(e)[:200]}")
            return None
        # Soft match: salvage whatever verdicts we can align.
        print(
            "ℹ️ User Requirements: strict match failed after retry - FALLBACK TO SOFT!!!!!!\n" f"Reason: {str(e)[:200]}"
        )
        return _soft_match_requirements(requirements, blocks)


async def get_score_for_requirements(
    sample_data: dict,
    model_name: str = "gemini-3.1-pro-preview",
    requirements: list | None = None,
    evaluation_result_key: str = "req_eval_full",
) -> dict:
    raw_content = sample_data["content"]
    visual_intent = sample_data["visual_intent"]

    eval_image_field = sample_data["eval_image_field"]
    if requirements is None:
        requirements = sample_data["requirements"]

    # Check if image was successfully generated
    if eval_image_field not in sample_data:
        print(f"⚠️ Image field '{eval_image_field}' not found. Model generation failed - set score = 0")
        sample_data[evaluation_result_key] = [
            {
                "req_eval_reasoning": "No image",
                "status": "Unsatisfied",
                "status_numerical": 0,
            }
            for _ in range(len(requirements))
        ]
        return sample_data

    # load image for eval
    model_image_base64 = await read_base64(sample_data[eval_image_field])

    # run batched eval for all requirements
    for _ in range(3):
        results = await _run_batched_eval_requirement(
            raw_content,
            visual_intent,
            requirements,
            model_image_base64,
            model_name,
            strict_match=True,
        )
        if results is not None:
            break
    # if it still fails: perform soft eval
    if results is None:
        results = await _run_batched_eval_requirement(
            raw_content,
            visual_intent,
            requirements,
            model_image_base64,
            model_name,
            strict_match=False,
        )
    sample_data[evaluation_result_key] = results

    return sample_data
