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

import argparse
import asyncio
import copy
import json
from pathlib import Path

import aiofiles
import numpy as np
from tqdm import tqdm

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.polish_agent import PolishAgent
from agents.retriever_agent import RetrieverAgent
from agents.stylist_agent import StylistAgent
from agents.vanilla_agent import VanillaAgent
from agents.visualizer_agent import VisualizerAgent
from agents.refiner import GenerativeRefiner, PBDirectRefine, PBInteract

from utils import config, generation_utils, paperviz_processor
from utils.base64_rw_helper import offload_base64_fields


def _as_req_indices(surfaced_index):
    """Normalize ``surfaced_index`` (scalar int in single-req runs, list of ints
    in multi-req runs) to a list of ints."""
    return list(surfaced_index) if isinstance(surfaced_index, (list, tuple)) else [surfaced_index]


def _surfaced_addressed_ratio(data):
    """Ratio of user requests (the requirements surfaced in the latest chat turn)
    that are now addressed (status_numerical == 1) in this turn's eval.

    Each surfaced requirement is its own event: a multi-req turn that surfaced
    ``[0, 3]`` contributes two events.

    Early-stopped snapshots are skipped: that turn never happened, and its
    trailing ``surfaced_index`` is a stale copy of the last real turn's request
    (counting it would duplicate an event already counted at that turn).

    Returns (ratio, addressed_count, total_with_surfaced).
    """
    addressed = []
    for dp in data:
        if dp.get("early_stopped"):
            continue
        chat = dp.get("chat_history") or []
        surfaced = None
        for turn in reversed(chat):
            if turn.get("surfaced_index") is not None:
                surfaced = turn["surfaced_index"]
                break
        if surfaced is None:
            continue
        req_eval = dp.get("req_eval_full") or []
        for j in _as_req_indices(surfaced):
            if j >= len(req_eval):
                continue
            addressed.append(1 if req_eval[j]["status_numerical"] == 1 else 0)
    if not addressed:
        return None, 0, 0
    return float(np.mean(addressed)), int(np.sum(addressed)), len(addressed)


def _forgetting_rate(per_turn_results):
    """Forgot@any, pooled over (turn, datapoint) events (no dedup):

        sum_{t=1..last-1} sum_{dp} 1(req surfaced & satisfied at turn t
                                     & unsatisfied after a later real edit)
        ----------------------------------------------------------------
        sum_{t=1..last-1} sum_{dp} 1(req surfaced & satisfied at turn t
                                     & a later real edit exists)

    Each (turn t, dp, surfaced requirement) is its own event -- a requirement
    re-surfaced on two turns counts twice, and a multi-req turn surfacing
    ``[0, 3]`` contributes two events. Early-stopped turns "never happened" and contribute to
    neither side: they are skipped as surfacing turns (their trailing
    ``surfaced_index`` is stale), skipped when scanning later turns, and an
    event whose later turns are ALL early-stopped (no real edit ever followed,
    so forgetting was impossible) is dropped from the denominator -- the same
    rationale as excluding the last turn as a surfacing turn.
    """
    n_turns = len(per_turn_results)
    if n_turns < 2:
        return None, 0, 0
    last = n_turns - 1
    data_by_id = [{dp["id"]: dp for dp in turn} for turn in per_turn_results]

    forgotten = []
    for t in range(1, last):  # surfacing turns 1..last-1
        for dp in per_turn_results[t]:
            if dp.get("early_stopped"):
                continue
            chat = dp.get("chat_history") or []
            surfaced = None
            for turn in reversed(chat):
                if turn.get("surfaced_index") is not None:
                    surfaced = turn["surfaced_index"]
                    break
            if surfaced is None:
                continue
            req_eval = dp.get("req_eval_full") or []
            for j in _as_req_indices(surfaced):
                if j >= len(req_eval):
                    continue
                if req_eval[j]["status_numerical"] != 1:
                    continue  # surfaced but not satisfied this turn -> not an event

                # denominator event: surfaced & satisfied at turn t, AND at least
                # one later real (non-early-stopped) edit exists to forget it in.
                lost = False
                later_edit = False
                for t2 in range(t + 1, last + 1):  # any later turn
                    later_dp = data_by_id[t2].get(dp["id"])
                    if later_dp is None or later_dp.get("early_stopped"):
                        continue
                    later_edit = True
                    later_eval = later_dp.get("req_eval_full") or []
                    if j < len(later_eval) and later_eval[j]["status_numerical"] == 0:
                        lost = True
                        break
                if not later_edit:
                    continue  # no later edit -> no exposure to forgetting
                forgotten.append(1 if lost else 0)
    if not forgotten:
        return None, 0, 0
    return float(np.mean(forgotten)), int(np.sum(forgotten)), len(forgotten)


def print_aggregated_metrics(data, it, per_turn_results=None):
    print(f"\n===== Aggregated metrics @ iteration {it} (n={len(data)}) =====")

    def to_score(values):
        model = values.count("model")
        tie = values.count("tie") + values.count("both are good") + values.count("both are bad")
        return (model * 100 + tie * 50) / len(values)

    for key in [
        "faithfulness_outcome",
        "conciseness_outcome",
        "readability_outcome",
        "aesthetics_outcome",
        "overall_outcome",
    ]:
        values = [x[key].strip().lower() for x in data if key in x]
        if values:
            print(key, to_score(values))

    requirement_score_all = [x["status_numerical"] for dp in data for x in dp.get("req_eval_full", [])]
    if requirement_score_all:
        print("\nrequirement overall %.1f" % (float(np.mean(requirement_score_all)) * 100))

    if per_turn_results is not None:
        # Surfaced-request addressed, pooled over every surfacing event across all turns so far (not just this turn).
        cum_addressed, cum_total = 0, 0
        for turn_data in per_turn_results:
            _, a, t = _surfaced_addressed_ratio(turn_data)
            cum_addressed += a
            cum_total += t
        if cum_total:
            print(
                "surfaced user request addressed (avg): %.1f%% (%d/%d)"
                % (cum_addressed / cum_total * 100, cum_addressed, cum_total)
            )

        f_rate, forgotten_cnt, f_total = _forgetting_rate(per_turn_results)
        if f_rate is None:
            print("forgetting rate: N/A (need >=2 turns)")
        else:
            print("forgetting rate: %.1f%% (%d/%d)" % (f_rate * 100, forgotten_cnt, f_total))
    print("=" * 60)


async def main():
    """Main function"""

    # add command line args
    parser = argparse.ArgumentParser(description="Run PaperBanana multi-turn benchmark experiments")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MTPaperBananaBench",
        help="dataset directory name under data/ (default: MTPaperBananaBench)",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="dataset split to use (default: test)",
    )
    parser.add_argument(
        "--exp_mode",
        type=str,
        default="dev_full",
        help="PaperBanana experiment mode for single-turn generation (default: dev_full)",
    )
    parser.add_argument(
        "--retrieval_setting",
        type=str,
        default="auto",
        choices=["auto", "manual", "random", "none"],
        help="retrieval strategy for single-turn generation (default: auto)",
    )
    parser.add_argument(
        "--max_critic_rounds",
        type=int,
        default=3,
        help="maximum number of critic rounds during single-turn generation (default: 3)",
    )
    parser.add_argument(
        "--main_model_name",
        type=str,
        default="gemini-3.1-pro-preview",
        help="model used for text generation and reasoning (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--image_gen_model_name",
        type=str,
        default="gemini-3-pro-image",
        help="model used for image generation and generative refinement (default: gemini-3-pro-image)",
    )
    # engineering: concurrency and automatic saving & loading
    parser.add_argument(
        "--concurrent_num",
        type=int,
        default=64,
        help="maximum number of concurrently processed examples and API calls (default: 64)",
    )
    parser.add_argument(
        "--load_initial_result",
        type=str,
        default=None,
        help="path to a JSON file containing single-turn generation results; "
        "load available results instead of regenerating them",
    )
    parser.add_argument(
        "--load_prior_results",
        type=str,
        default=None,
        help="path to an experiment directory containing numbered turn files "
        "(0.json, 1.json, ...); continue each example from its latest available "
        "turn and write results to a new experiment directory. Mutually exclusive "
        "with --load_initial_result",
    )
    # --- multi-turn args
    parser.add_argument(
        "--feedback_turns",
        type=int,
        default=5,
        help="number of simulated feedback iterations after single-turn generation (default: 5)",
    )
    # --- user simulator args
    parser.add_argument(
        "--user_model_name",
        type=str,
        default="gemini-3.1-pro-preview",
        help="model used to simulate user feedback (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--n_surfaced_req_per_turn",
        type=int,
        default=1,
        help="number of unsatisfied requirements surfaced per feedback turn (k in the paper); "
        "values greater than 1 use the multi-requirement simulator (default: 1)",
    )
    # --- refiner args
    parser.add_argument(
        "--refiner",
        type=str,
        default="generative",
        choices=["generative", "pb_directrefine", "pb_interact"],
        help="refiner used for simulated feedback turns (default: generative)",
    )
    parser.add_argument(
        "--refiner_internal_critic_rounds",
        type=int,
        default=1,
        help="maximum internal refinement rounds per feedback turn for "
        "pb_directrefine and pb_interact; generative requires 1 (default: 1)",
    )
    args = parser.parse_args()

    # Bind the global API concurrency limiter to the CLI arg (before any API calls).
    generation_utils.set_api_concurrency(args.concurrent_num)

    exp_config = config.ExpConfig(
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval_setting,
        max_critic_rounds=args.max_critic_rounds,
        refiner_internal_critic_rounds=args.refiner_internal_critic_rounds,
        main_model_name=args.main_model_name,
        image_gen_model_name=args.image_gen_model_name,
        refiner=args.refiner,
        work_dir=Path(__file__).parent,
        concurrent_num=args.concurrent_num,
        user_model_name=args.user_model_name,
        n_surfaced_req_per_turn=args.n_surfaced_req_per_turn,
    )

    if args.load_initial_result is not None and args.load_prior_results is not None:
        parser.error("--load_initial_result and --load_prior_results are mutually exclusive")

    base_path = exp_config.dataset_dir
    input_filename = base_path / f"{exp_config.split_name}.json"
    output_dirname = exp_config.result_dir / f"{exp_config.exp_name}"
    output_dirname.mkdir()

    print(f"Input file: {input_filename}", f"Output dirname: {output_dirname}")

    # Dump the full run config into the experiment directory (config.json).
    config_filename = output_dirname / "config.json"
    with open(config_filename, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    print(f"Config file: {config_filename}")

    with open(input_filename, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    assert all(["requirements" in data and "req_order" in data for data in data_list])

    # Pick refiner
    if exp_config.refiner == "generative":
        assert (
            args.refiner_internal_critic_rounds == 1
        ), "Generative model as refiner does not support multiple internal iterations"
        refiner_agent = GenerativeRefiner(exp_config=exp_config)
    elif exp_config.refiner == "pb_directrefine":
        refiner_agent = PBDirectRefine(exp_config=exp_config)
    elif exp_config.refiner == "pb_interact":
        refiner_agent = PBInteract(exp_config=exp_config)
    else:
        raise ValueError(f"Unknown refiner: {exp_config.refiner}")

    # Create processor
    processor = paperviz_processor.PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
        refiner_agent=refiner_agent,
    )

    async def save_results_and_scores(current_results, it):
        output_filename = output_dirname / f"{it}.json"
        print(f"Incremental saving results (count: {len(current_results)}) to {output_filename}")
        # Offload base64 blobs
        thinned = await offload_base64_fields(current_results, output_dirname)
        json_string = await asyncio.to_thread(
            lambda: json.dumps(thinned, ensure_ascii=False, indent=4).encode("utf-8", "ignore").decode("utf-8")
        )
        async with aiofiles.open(output_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
            await f.write(json_string)

    # ----------- Pipelined driver: each datapoint flows T=0 -> 1 -> ... independently.
    N = len(data_list)
    per_turn_results = [[] for _ in range(args.feedback_turns + 1)]
    save_lock = asyncio.Lock()

    async def save_turn(it):
        async with save_lock:
            await save_results_and_scores(list(per_turn_results[it]), it)

    # record func: saves every 50 items, every 10 past 250, every 1 past 280
    async def record(it, result):
        bucket = per_turn_results[it]
        bucket.append(result)
        n = len(bucket)
        pbar.update(1)
        pbar.set_postfix(
            {
                f"remaining_{t}": f"{N - len(b)}({100 * (N - len(b)) // N}%)"
                for t, b in enumerate(per_turn_results)
                if len(b) < N
            }
        )
        interval = 1 if n > 280 else 10 if n > 250 else 50
        if n % interval == 0 and n != N:
            await save_turn(it)
        if n == N:
            await save_turn(it)
            print("Processing for iteration %d completed." % it)
            print_aggregated_metrics(bucket, it, per_turn_results[: it + 1])

    # Preload prior results. preloaded[it] = {id: result}
    preloaded = {}
    if args.load_initial_result is not None:
        print(f"Loading single-turn results from {args.load_initial_result}")
        with open(args.load_initial_result, "r", encoding="utf-8") as f:
            preloaded[0] = {r["id"]: r for r in json.load(f)}
    elif args.load_prior_results is not None:
        for it in range(args.feedback_turns + 1):
            f = Path(args.load_prior_results) / f"{it}.json"
            if f.exists():
                print(f"Loading turn {it} result from {f}")
                with open(f, "r", encoding="utf-8") as fh:
                    preloaded[it] = {r["id"]: r for r in json.load(fh)}
    # Enforce chain contiguity
    kept = set(preloaded.get(0, {}))
    for it in range(1, args.feedback_turns + 1):
        if it not in preloaded:
            kept = set()
            continue
        dropped = [idv for idv in preloaded[it] if idv not in kept]
        if dropped:
            print(
                f"⚠️ turn {it}: discarding {len(dropped)} result(s) present in "
                f"{it}.json but missing from an earlier turn file: {dropped}"
            )
            for idv in dropped:
                del preloaded[it][idv]
        kept = set(preloaded[it])
        if not preloaded[it]:
            del preloaded[it]
    # Seed buckets from preloaded results (in data_list order for stable output).
    for it in sorted(preloaded):
        for d in data_list:
            if d["id"] in preloaded[it]:
                per_turn_results[it].append(preloaded[it][d["id"]])
    # Fire completion for turns already full from preload, in ascending order.
    for it in sorted(preloaded):
        if len(per_turn_results[it]) == N:
            await save_turn(it)
            print_aggregated_metrics(per_turn_results[it], it, per_turn_results[: it + 1])

    data_by_id = {d["id"]: d for d in data_list}
    sem = asyncio.Semaphore(exp_config.concurrent_num)

    async def run_pipeline(d):
        idv = d["id"]
        resume = max((it for it in preloaded if idv in preloaded[it]), default=-1)
        if resume < 0:
            async with sem:
                cur = await processor.process_query_one(data_by_id[idv], do_eval=True)
            # Record a copy: an early stop at a later turn overwrites "early_stopped" and "prior_preferred_candidates" in place.
            await record(0, copy.deepcopy(cur))
            start = 1
        else:
            # Private working copy, so the seeded bucket entry stays pristine.
            cur = copy.deepcopy(preloaded[resume][idv])
            start = resume + 1
        for it in range(start, args.feedback_turns + 1):
            async with sem:
                cur = await processor.process_single_feedback(cur, simulate_user=True, do_eval=True)
            await record(it, copy.deepcopy(cur))

    # Single-turn retriever pre-step, only if any datapoint runs single-turn generation.
    if any(d["id"] not in preloaded.get(0, {}) for d in data_list):
        # Partial turn 0: reuse retrieval already baked into saved results (copy to every item) instead of re-retrieving.
        done0 = preloaded.get(0, {})
        if done0:
            src = next(iter(done0.values()))
            for key in ("top10_references", "retrieved_examples"):
                if key in src:
                    for d in data_list:
                        d[key] = src[key]
        await processor.prepare_retrieval(data_list)

    # total = all (datapoint, turn) units minus those already seeded from preload.
    total = N * (args.feedback_turns + 1) - sum(len(b) for b in per_turn_results)
    with tqdm(total=total, desc="Pipelined turns", ascii=True) as pbar:
        await asyncio.gather(*(run_pipeline(d) for d in data_list))


if __name__ == "__main__":
    asyncio.run(main())
