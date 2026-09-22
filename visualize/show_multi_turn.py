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

import streamlit as st
import json
import base64
import glob
import ijson
import os
import sys
import re
from io import BytesIO
from pathlib import Path

# Make the repo root importable regardless of the launch directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.base64_rw_helper import read_base64_sync

st.set_page_config(layout="wide", page_title="PaperBanana-Interact Visualizer", page_icon="🍌")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    """Load a JSON array."""
    data = []
    if not os.path.exists(path):
        return []

    file_ext = os.path.splitext(path)[1].lower()
    try:
        assert file_ext == ".json", f"expected a .json file, got {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                st.error("JSON file must contain an array at the top level")
                return []
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []
    return data


@st.cache_data
def discover_turns(dir_path):
    """Find complete ``<N>.json`` turn files in ``dir_path``.

    Stream item counts with ijson, then reject empty, incomplete, or unparseable
    files. Return sorted turn numbers and diagnostics containing ``counts``, the
    maximum count in ``full``, and excluded turns in ``dropped``.
    """
    turns = []
    for p in glob.glob(os.path.join(dir_path, "*.json")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.isdigit():
            turns.append(int(stem))
    turns.sort()

    counts = {}
    for t in turns:
        try:
            c = 0
            with open(os.path.join(dir_path, f"{t}.json"), "rb") as fh:
                for _ in ijson.items(fh, "item"):
                    c += 1
            counts[t] = c
        except (ijson.common.IncompleteJSONError, ValueError):
            counts[t] = None  # unparseable / still being written

    nonempty = [c for c in counts.values() if c]
    full = max(nonempty) if nonempty else 0
    valid_turns = [t for t in turns if counts[t] == full and full > 0]
    dropped = [t for t in turns if t not in valid_turns]
    return valid_turns, {"counts": counts, "full": full, "dropped": dropped}


def render_b64_image(b64_str):
    """Render one base64 image with Streamlit's native image component."""
    if not b64_str or not isinstance(b64_str, str):
        return False
    try:
        payload = b64_str.split(",", 1)[1] if b64_str.startswith("data:") else b64_str
        image = BytesIO(base64.b64decode(payload))
    except (ValueError, TypeError):
        return False
    st.image(image, use_container_width=True)
    return True


def render_local_image(path):
    """Render one local image with Streamlit's native image component."""
    if not path or not os.path.exists(path):
        return False
    st.image(path, use_container_width=True)
    return True


def id_sort_key(item):
    """Sort by the integer after ``id_``."""
    id_str = str(item.get("id", ""))
    m = re.search(r"id_(\d+)", id_str)
    if m:
        return (0, int(m.group(1)), "")
    return (1, 0, id_str.lower())


def detect_task_type(item):
    """Detect whether an item represents a diagram or plot task."""
    if "target_plot_desc0" in item or "target_plot_stylist_desc0" in item:
        return "plot"
    if isinstance(item.get("content"), dict):
        return "plot"
    return "diagram"


def detect_dataset_task_type(data):
    """Infer a dataset's task type from its first item."""
    if not data:
        return "diagram"
    return detect_task_type(data[0])


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def calculate_preference_stats(data):
    """Compute preference-evaluation statistics."""
    total_data_points = len(data)
    total_preferences = 0

    overall_scores = []
    category_scores = {"Content": [], "Organization": [], "Visual Representation": []}
    priority_scores = {"p0": [], "p1": [], "p2": []}

    status_counts = {"Satisfied": 0, "Unsatisfied": 0, "Error": 0}

    for dp in data:
        pref_eval = dp.get("req_eval_full", [])
        prefs = dp.get("requirements", [])

        for item, pref in zip(pref_eval, prefs):
            total_preferences += 1
            score = item.get("status_numerical")
            status = (item.get("status") or "").capitalize()

            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts["Error"] += 1

            if score is not None:
                overall_scores.append(score)
                cat = pref.get("category")
                if cat in category_scores:
                    category_scores[cat].append(score)
                prio = pref.get("priority")
                if prio in priority_scores:
                    priority_scores[prio].append(score)

    return {
        "total_data_points": total_data_points,
        "total_preferences": total_preferences,
        "overall_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
        "status_counts": status_counts,
        "categories": {cat: (sum(vals) / len(vals) if vals else 0) for cat, vals in category_scores.items()},
        "priorities": {prio: (sum(vals) / len(vals) if vals else 0) for prio, vals in priority_scores.items()},
    }


def calculate_referenced_stats(data, dimensions):
    """Compute win rates by evaluation dimension."""
    outcomes = ["Model", "Human", "Both are good", "Both are bad", "Tie", "Error", "Unknown"]
    stats = {dim: {out: 0 for out in outcomes} for dim in dimensions}

    for item in data:
        for dim in dimensions:
            outcome = item.get(f"{dim.lower()}_outcome", "Unknown")
            if outcome in outcomes:
                stats[dim][outcome] += 1
            else:
                stats[dim]["Unknown"] += 1
    return stats


def display_outcome(outcome, short=None):
    label = (short or {}).get(outcome, outcome)
    # Mirror the Preference Eval palette: best -> green, intermediate -> orange,
    # worst -> red. A model win is best, a tie is intermediate, a human win is
    # worst. "Both are good" reads as a (positive) tie, "Both are bad" as worst.
    styles = {
        "Model": f":green[**{label}**]",
        "Tie": f":orange[**{label}**]",
        "Both are good": f":orange[**{label}**]",
        "Human": f":red[**{label}**]",
        "Both are bad": f":red[**{label}**]",
    }
    return styles.get(outcome, f":gray[**{label}**]")


def format_reasoning(text):
    """Format evaluation reasoning for Streamlit."""
    if not text:
        return ""

    headers = [
        "Faithfulness of Human",
        "Faithfulness of Model",
        "Conciseness of Human",
        "Conciseness of Model",
        "Readability of Human",
        "Readability of Model",
        "Aesthetics of Human",
        "Aesthetics of Model",
        "Overall Quality of Human",
        "Overall Quality of Model",
        "Conclusion",
    ]

    formatted_text = text
    for header in headers:
        pattern = re.compile(rf"({re.escape(header)}):", re.IGNORECASE)
        formatted_text = pattern.sub(r"\n\n**\1**:", formatted_text)

    formatted_text = re.sub(r";\s*\n\n", r"\n\n", formatted_text)
    return formatted_text.strip()


# ---------------------------------------------------------------------------
# View: Chat History (the edit request)
# ---------------------------------------------------------------------------
def identify_surfaced_req(item):
    """Identify the requirement surfaced during this turn.

    Match the simulator by selecting the first ``req_order`` entry that was
    unsatisfied before the edit. Return its index, or ``None``.
    """
    req_order = item.get("req_order") or []
    req_eval = item.get("req_eval_full") or []
    for i in req_order:
        if isinstance(i, int) and 0 <= i < len(req_eval):
            if req_eval[i].get("status_numerical") == 0:
                return i
    return None


def get_turn_request(item):
    """Return the request represented by a turn snapshot.

    Read the latest user message and fall back to ``identify_surfaced_req`` when
    it lacks ``surfaced_index``. Return ``None`` for single-turn generation.
    """
    chat = item.get("chat_history") or []
    user_msgs = [m for m in chat if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs:
        return None
    msg = user_msgs[-1]
    idx = msg.get("surfaced_index")
    if idx is None:
        idx = identify_surfaced_req(item)
    return {
        "surfaced_index": idx,
        "content": msg.get("content", ""),
        "requirement_description": msg.get("requirement_description"),
        "eval_reasoning": msg.get("eval_reasoning"),
    }


def as_req_indices(surfaced_index):
    """Normalize a scalar or sequence-valued ``surfaced_index`` to a list."""
    return list(surfaced_index) if isinstance(surfaced_index, (list, tuple)) else [surfaced_index]


def surfaced_status(item, surfaced_index):
    """Return the post-edit status of one surfaced requirement.

    ``1`` means satisfied, ``0`` means unsatisfied, and ``None`` means unknown.
    """
    if surfaced_index is None:
        return None
    evals = item.get("req_eval_full") or []
    if 0 <= surfaced_index < len(evals):
        return evals[surfaced_index].get("status_numerical")
    return None


# ---------------------------------------------------------------------------
# View: Trajectory filmstrip (one compact column per turn)
# ---------------------------------------------------------------------------
def _turn_image_b64(item):
    """Return the best available image for a turn snapshot.

    Prefer ``eval_image_field``, then try critic, stylist, and vanilla outputs.
    """
    field = item.get("eval_image_field")
    if field and item.get(field):
        return item.get(field)
    for key in (
        "target_diagram_critic_desc0_base64_jpg",
        "target_diagram_stylist_desc0_base64_jpg",
        "target_diagram_desc0_base64_jpg",
        "target_plot_desc0_base64_jpg",
        "target_plot_stylist_desc0_base64_jpg",
        "vanilla_image_base64",
    ):
        if item.get(key):
            return item.get(key)
    return None


def forgotten_reqs_at_turn(trajectory, turns, t):
    """Find earlier satisfied requests that are unsatisfied at turn ``t``.

    Match ``main_multiturn._forgetting_rate`` by ignoring early-stopped turns and
    counting requirements satisfied when surfaced but unsatisfied later.
    """
    cur_eval = (trajectory.get(t) or {}).get("req_eval_full") or []
    forgotten = set()
    for tt in turns:
        if not (0 < tt < t):  # turn 0 is initial gen, not a surfacing turn
            continue
        prev = trajectory.get(tt)
        if prev is None or prev.get("early_stopped"):
            continue
        req = get_turn_request(prev)
        surfaced = req.get("surfaced_index") if req else None
        if surfaced is None:
            continue
        for j in as_req_indices(surfaced):
            if surfaced_status(prev, j) == 1 and j < len(cur_eval) and cur_eval[j].get("status_numerical") == 0:
                forgotten.add(j)
    return sorted(forgotten)


def render_filmstrip(
    trajectory, turns, dimensions, show_preference, show_referenced, show_evolution, expand_by_default
):
    """Render a trajectory top-to-bottom, one row per turn.

    Show the request and evaluation views on the left, and the generated image
    and satisfaction status on the right.
    """
    for t in turns:
        item = trajectory.get(t)
        st.markdown(f"**Turn {t}**")
        if item is None:
            st.caption("—")
            st.divider()
            continue

        req_col, img_col = st.columns([1, 1])

        # Left: the user request for this turn, followed by its eval views.
        with req_col:
            req = get_turn_request(item)
            if t == 0 or req is None:
                st.caption("🟢 Initial generation")
            else:
                surf_idx = req.get("surfaced_index")
                content = (req.get("content") or "").strip()
                if surf_idx is not None:
                    st.caption("🎯 " + ", ".join(f"#{j}" for j in as_req_indices(surf_idx)) + " surfaced")
                st.markdown(content or "_(no request text)_")

            # Eval views directly below the utterance (left side is otherwise empty).
            render_item_views(
                item,
                f"Turn {t} ({t}.json)",
                dimensions,
                show_preference,
                show_referenced,
                show_evolution,
                expand_by_default,
                forgotten_reqs_at_turn(trajectory, turns, t),
            )

        # Right: the satisfaction badge (above, so it's easy to see), then the
        # eval image for this turn, then the model description dropdown.
        with img_col:
            if t != 0 and req is not None:
                surfaced = req.get("surfaced_index")
                statuses = [surfaced_status(item, j) for j in as_req_indices(surfaced)] if surfaced is not None else []
                known = [s for s in statuses if s is not None]
                n_fixed = sum(1 for s in known if s == 1)
                if known and n_fixed == len(known):
                    st.markdown(":green[**✅ Fixed**]" if len(known) == 1 else f":green[**✅ All {len(known)} fixed**]")
                elif known:
                    st.markdown(
                        ":red[**❌ Still unsatisfied**]"
                        if len(known) == 1
                        else f":red[**❌ {n_fixed}/{len(known)} fixed**]"
                    )
                else:
                    st.markdown(":gray[**➖ n/a**]")

            field = item.get("eval_image_field")
            img_b64 = read_base64_sync(item.get(field) if field else _turn_image_b64(item))
            if not render_b64_image(img_b64):
                st.info(f"No image (`eval_image_field`={field})")

            # Evolution-stage fields for the displayed image, in dropdowns below
            # it: description, critic suggestions, contextualized request/critiques.
            render_stage_details(item, field, detect_task_type(item))

        st.divider()


# ---------------------------------------------------------------------------
# Cross-turn trajectory stats (implemented-at-turn / forgotten-later)
# ---------------------------------------------------------------------------
def compute_trajectory_stats(per_turn, turns, ids):
    """Compute implementation and cumulative forgetting counts by surfacing turn.

    Derive request order from the last loaded chat history. A requirement
    implemented at turn ``k`` counts as forgotten at column ``t`` once it was seen
    unsatisfied on any real (non early-stopped) turn in ``(k, t]``, so forgetting
    that already happened keeps counting after a trajectory has ended. Every column
    shares one denominator: implemented events that saw at least one later real
    edit, matching ``main_multiturn._forgetting_rate``. Return one summary dict per
    loaded surfacing turn.
    """
    if not turns:
        return []
    max_turn = turns[-1]

    def surf_order(id_):
        """Return normalized surfaced-requirement indices from the last turn."""
        item = per_turn[max_turn].get(id_)
        chat = (item or {}).get("chat_history") or []
        return [
            as_req_indices(m["surfaced_index"])
            for m in chat
            if isinstance(m, dict) and m.get("role") == "user" and m.get("surfaced_index") is not None
        ]

    surf_of = {id_: surf_order(id_) for id_ in ids}
    later_turns = [t for t in turns if t >= 2]

    def forgotten_by(id_, rk, k, upto):
        """Report whether ``rk`` was unsatisfied on any real turn in ``(k, upto]``.

        Early-stopped turns never happened -- their record is a copy of the
        previous turn -- so they are not evidence of new forgetting.
        """
        for t in turns:
            if not (k < t <= upto):
                continue
            item = per_turn.get(t, {}).get(id_)
            if item is None or item.get("early_stopped"):
                continue
            if surfaced_status(item, rk) == 0:
                return True
        return False

    def has_later_edit(id_, k):
        """Report whether any real (non early-stopped) turn follows turn ``k``."""
        for t in turns:
            if t > k:
                item = per_turn.get(t, {}).get(id_)
                if item is not None and not item.get("early_stopped"):
                    return True
        return False

    rows = []
    for k in [t for t in turns if t >= 1]:
        impl = forg_any = n = denom = 0
        forg = {t: 0 for t in later_turns}
        for id_ in ids:
            surf = surf_of[id_]
            if len(surf) < k:  # this item surfaced fewer than k turns
                continue
            for rk in surf[k - 1]:  # each requirement surfaced at turn k is its own event
                n += 1
                if surfaced_status(per_turn[k][id_], rk) != 1:
                    continue
                impl += 1
                if not has_later_edit(id_, k):
                    continue  # no later real edit ever happened -> forgetting was impossible
                # Cumulative: an event stays in the denominator of every later column
                # once it is in at all, whether or not its trajectory got that far.
                denom += 1
                for t in later_turns:
                    if t > k and forgotten_by(id_, rk, k, t):
                        forg[t] += 1
                if forgotten_by(id_, rk, k, max_turn):
                    forg_any += 1
        rows.append(
            {"turn": k, "n": n, "impl": impl, "denom": denom, "forgot": forg, "forgot_any": forg_any}
        )
    return rows


# ---------------------------------------------------------------------------
# View: Preference Eval
# ---------------------------------------------------------------------------
def display_preference_eval(item, expanded):
    """Render requirement-level evaluation for one item."""
    preferences = item.get("requirements", [])
    preference_eval = item.get("req_eval_full", [])

    with st.expander("⭐ Preference Eval", expanded=expanded):
        item_scores = [e.get("status_numerical") for e in preference_eval if e.get("status_numerical") is not None]
        if item_scores:
            avg_score = sum(item_scores) / len(item_scores)
            satisfied_count = sum(1 for s in item_scores if s == 1)
            st.markdown(
                f"**Item Preference Score:** `{avg_score:.1%}` ({satisfied_count}/{len(item_scores)} Satisfied)"
            )

        if not preferences:
            st.info("No preferences defined for this item.")
        elif len(preferences) != len(preference_eval):
            st.warning(f"Mismatch: found {len(preferences)} preferences and {len(preference_eval)} evaluations.")
        else:
            for k, (pref, eval_item) in enumerate(zip(preferences, preference_eval)):
                pref_desc = pref.get("description", "No description")
                pref_cat = pref.get("category", "N/A")
                pref_prio = pref.get("priority", "N/A")

                eval_reasoning = eval_item.get("req_eval_reasoning", "No reasoning provided.")
                eval_status = (eval_item.get("status") or "Unknown").capitalize()

                with st.container(border=True):
                    badge_col, meta_col = st.columns([0.3, 0.7])
                    with badge_col:
                        status_colors = {
                            "Satisfied": ":green[**Satisfied**]",
                            "Unsatisfied": ":red[**Unsatisfied**]",
                            "Error": ":gray[**Error**]",
                        }
                        st.markdown(status_colors.get(eval_status, f"❓ {eval_status}"))
                    with meta_col:
                        st.markdown(
                            f"<div style='text-align: right; color: gray; font-size: 0.9rem;'><b>Category:</b> {pref_cat} | <b>Priority:</b> {pref_prio}</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**Preference #{k}:** {pref_desc}")
                    with st.expander("📝 Evaluation Reasoning", expanded=False):
                        st.markdown(eval_reasoning)


# ---------------------------------------------------------------------------
# View: Evaluation Overview (referenced eval + preference summary)
# ---------------------------------------------------------------------------
def _pct_color(val):
    """Map a 0–1 fraction to green, orange, or red."""
    if val >= 0.8:
        return f":green[**{val:.0%}**]"
    if val >= 0.5:
        return f":orange[**{val:.0%}**]"
    return f":red[**{val:.0%}**]"


def display_preference_summary(item):
    """Render one item's preference summary on a single line."""
    if not item.get("requirements"):
        return

    stats = calculate_preference_stats([item])
    if stats["total_preferences"] == 0:
        return

    satisfied = stats["status_counts"]["Satisfied"]
    total = stats["total_preferences"]

    # Per-category and per-priority satisfied/total for this item.
    cat_short = {"Content": "Cat", "Organization": "Org", "Visual Representation": "Vis"}
    cat_counts = {c: [0, 0] for c in cat_short}  # [satisfied, total]
    prio_counts = {}  # priority -> [satisfied, total]
    for pref, ev in zip(item.get("requirements", []), item.get("req_eval_full", [])):
        is_sat = (ev.get("status") or "").capitalize() == "Satisfied"
        cat = pref.get("category")
        if cat in cat_counts:
            cat_counts[cat][1] += 1
            cat_counts[cat][0] += int(is_sat)
        prio = pref.get("priority")
        if prio is not None:
            prio_counts.setdefault(prio, [0, 0])
            prio_counts[prio][1] += 1
            prio_counts[prio][0] += int(is_sat)

    def _label(name, s, t):
        return f"**{name}** &nbsp; {_pct_color(s / t if t else 0)} ({s} / {t})"

    cat = " &nbsp;&nbsp;&nbsp; ".join(
        _label(short, *cat_counts[full]) for full, short in cat_short.items() if cat_counts[full][1]
    )
    prio = " &nbsp;&nbsp;&nbsp; ".join(_label(p, s, t) for p, (s, t) in sorted(prio_counts.items()))

    sep = " &nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp; "
    st.markdown(f"{_label('Preference', satisfied, total)}{sep}{cat}{sep}{prio}")


def display_referenced_eval(item, task_type, expanded):
    """Render model-versus-human results and a preference summary."""
    with st.expander("⚖️ Evaluation Overview", expanded=expanded):
        # One line: Model vs Human comparison outcomes, grouped by the tiered
        # decision rule (Tier 1: Faithfulness + Readability, Tier 2: Conciseness
        # + Aesthetics) and then the Overall verdict, with a larger gap between
        # groups so the tier structure is visible.
        short = {"Both are good": "Tie", "Both are bad": "Both bad"}

        def _cmp(dim):
            return f"{dim} {display_outcome(item.get(f'{dim.lower()}_outcome'), short)}"

        gap = " &nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp; "
        inner = " &nbsp;&nbsp;&nbsp; "
        groups = [
            inner.join(_cmp(d) for d in ["Faithfulness", "Readability"]),
            inner.join(_cmp(d) for d in ["Conciseness", "Aesthetics"]),
            _cmp("Overall"),
        ]
        st.markdown(f"**Comparison**{gap}{gap.join(groups)}")

        # One line: aggregated preference summary
        display_preference_summary(item)

        # Human (reference) image. The model image + description now live next to
        # the actual turn image in the trajectory stream.
        human_label = "Human Plot" if task_type == "plot" else "Human Diagram"
        st.markdown(f"**{human_label}** (Reference)")
        gt_path = item.get("path_to_gt_image")
        if not render_local_image(gt_path):
            st.error(f"Human image not found at: {gt_path}")

        # Suggestions
        suggestions = item.get("suggestions_diagram") or item.get("suggestions_plot")
        if suggestions:
            with st.expander("💡 Polish Suggestions", expanded=False):
                st.markdown(suggestions)


# ---------------------------------------------------------------------------
# View: Comparison Eval (Model vs Human reasoning)
# ---------------------------------------------------------------------------
def display_comparison_eval(item, dimensions, expanded):
    """Render model-versus-human reasoning by dimension."""
    with st.expander("📝 Comparison Eval", expanded=expanded):
        tabs = st.tabs(dimensions)
        for tab, dim in zip(tabs, dimensions):
            with tab:
                reasoning = item.get(f"{dim.lower()}_reasoning", "No reasoning provided.")
                st.markdown(format_reasoning(reasoning))


# ---------------------------------------------------------------------------
# View: Evolution Stages (pipeline)
# ---------------------------------------------------------------------------
def _collect_critic_rounds(item, prefix):
    """Group critic-description keys by timestep.

    Single-turn rounds use ``{prefix}_critic_desc{j}``; feedback rounds use
    ``{prefix}_critic_desc{i}-{j}``. Return both groups in sorted order.
    """
    t0, feedback = [], {}
    for k in item:
        m = re.match(rf"^{re.escape(prefix)}_critic_desc(\d+)-(\d+)$", k)
        if m:
            feedback.setdefault(int(m.group(1)), []).append(int(m.group(2)))
            continue
        m = re.match(rf"^{re.escape(prefix)}_critic_desc(\d+)$", k)
        if m:
            t0.append(int(m.group(1)))
    t0.sort()
    for i in feedback:
        feedback[i].sort()
    return t0, feedback


def _critique_label(key, prefix, r):
    """Convert a critique key to a readable label."""
    base = key[len(f"{prefix}_critique_") : -len(str(r))]
    if base.startswith("current_turn"):
        return "Current Turn"
    if base.startswith("latest_turn"):
        return "Latest Turn"
    if base.startswith("source_content"):
        return "Source Content"
    if base.startswith("presentation"):
        return "Presentation"
    m = re.match(r"prior_turn(\d+)", base)
    if m:
        return f"Prior Turn {m.group(1)}"
    return base


def _render_history_context(ctx, item):
    """Render contextualized history with the original user text."""
    user_msgs = [m for m in (item.get("chat_history") or []) if isinstance(m, dict) and m.get("role") == "user"]
    nums = sorted({int(m.group(1)) for k in ctx for m in [re.match(r"context_(?:diagram|user_turn)(\d+)$", k)] if m})
    for n in nums:
        u, d = ctx.get(f"context_user_turn{n}"), ctx.get(f"context_diagram{n}")
        orig = (user_msgs[n - 1].get("content") if n - 1 < len(user_msgs) else "") or ""
        # DiagramN precedes user turn N (the user reacts to the diagram shown right before).
        if d and str(d).strip():
            st.markdown(f"**Diagram {n}:** {d}")
        if str(orig).strip():
            st.markdown(f"**User turn {n}:** {orig}")
        if u and str(u).strip():
            st.markdown(f"&nbsp;&nbsp;↳ _Context:_ {u}")


def _render_context_and_critiques(item, task_type, r):
    """Render context and per-perspective critiques for one critic round."""
    if r is None:
        return
    prefix = "target_plot" if task_type == "plot" else "target_diagram"

    # history contextualization: per-turn dict at ..._contextualize_each_turn_{i}-0 (i before the dash in r).
    turn = str(r).split("-")[0]
    hist_ctx = item.get(f"{prefix}_contextualize_each_turn_{turn}-0")
    if isinstance(hist_ctx, dict) and any(str(v).strip() for v in hist_ctx.values()):
        with st.expander("🧭 History Context", expanded=False):
            _render_history_context(hist_ctx, item)

    pat = re.compile(
        rf"^{re.escape(prefix)}_critique_(?:current_turn|latest_turn|source_content|presentation|prior_turn\d+_){re.escape(str(r))}$"
    )
    critique_keys = sorted(k for k in item if pat.match(k))
    if any(str(item.get(k, "")).strip() for k in critique_keys):
        with st.expander("🔎 Critiques", expanded=False):
            for k in critique_keys:
                val = item.get(k, "")
                if not str(val).strip():
                    continue
                st.markdown(f"**{_critique_label(k, prefix, r)}**")
                st.write(val)


def render_stage_details(item, field, task_type):
    """Render evolution details for the displayed image.

    Show its description, critic suggestions, context, and critiques in expanders.
    """
    desc_key = (field or "").replace("_base64_jpg", "")
    prefix = "target_plot" if task_type == "plot" else "target_diagram"

    desc = item.get(desc_key) or "N/A"
    with st.expander("📄 Model Description", expanded=False):
        if task_type == "plot" and desc != "N/A":
            st.code(desc, language="python")
        else:
            st.write(desc)

    m = re.match(rf"^{re.escape(prefix)}_critic_desc(.+)$", desc_key)
    r = m.group(1) if m else None
    if r is not None:
        suggestions = item.get(f"{prefix}_critic_suggestions{r}", "")
        if suggestions and suggestions.strip() != "No changes needed.":
            with st.expander("💬 Critic Suggestions", expanded=False):
                st.write(suggestions)

    _render_context_and_critiques(item, task_type, r)


def _render_stage_grid(stages, item, task_type):
    """Render stage dictionaries in a two-column grid."""
    cols_per_row = 2
    for row_start in range(0, len(stages), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            k = row_start + col_idx
            if k >= len(stages):
                break
            stage = stages[k]
            with cols[col_idx]:
                st.markdown(f"**{stage['title']}**")

                if stage.get("is_human"):
                    if not render_local_image(item.get("path_to_gt_image")):
                        st.info("No Human image available")
                    with st.expander("View Caption", expanded=False):
                        st.write(item.get("brief_desc", "No caption available"))
                    continue

                if not render_b64_image(read_base64_sync(item.get(stage["img_key"]))):
                    st.info("No image available")

                desc = item.get(stage["desc_key"], "No description available")
                with st.expander("View Description", expanded=False):
                    if task_type == "plot" and desc:
                        st.code(desc, language="python")
                    else:
                        st.write(desc)

                sug_key = stage.get("suggestions_key")
                suggestions = item.get(sug_key, "") if sug_key else ""
                if suggestions and suggestions.strip() != "No changes needed.":
                    with st.expander("💬 Critic Suggestions", expanded=False):
                        st.write(suggestions)

                _render_context_and_critiques(item, task_type, stage.get("round_suffix"))


def display_evolution_stages(item, task_type, expanded):
    """Render an item's full evolution across turns.

    Show the single-turn pipeline first, followed by each turn's refinement rounds.
    """
    prefix = "target_plot" if task_type == "plot" else "target_diagram"
    t0_rounds, feedback = _collect_critic_rounds(item, prefix)

    # i-th user message drives turn i; used as the timestep caption.
    user_msgs = [m for m in (item.get("chat_history") or []) if isinstance(m, dict) and m.get("role") == "user"]

    with st.expander("🪜 Evolution Stages", expanded=expanded):
        # t=0: initial generation pipeline.
        stages = [{"title": "🎯 Human (Ground Truth)", "is_human": True}]
        if f"{prefix}_desc0" in item:
            stages.append(
                {
                    "title": "📝 Planner / Vanilla",
                    "desc_key": f"{prefix}_desc0",
                    "img_key": f"{prefix}_desc0_base64_jpg",
                }
            )
        if f"{prefix}_stylist_desc0" in item:
            stages.append(
                {
                    "title": "✨ Stylist",
                    "desc_key": f"{prefix}_stylist_desc0",
                    "img_key": f"{prefix}_stylist_desc0_base64_jpg",
                }
            )
        for j in t0_rounds:
            stages.append(
                {
                    "title": f"🔍 Critic Round {j}",
                    "desc_key": f"{prefix}_critic_desc{j}",
                    "img_key": f"{prefix}_critic_desc{j}_base64_jpg",
                    "suggestions_key": f"{prefix}_critic_suggestions{j}",
                    "round_suffix": str(j),
                }
            )
        st.markdown("## 🕐 t=0 · Initial Generation")
        _render_stage_grid(stages, item, task_type)

        # t>=1: each feedback turn's critic-refiner rounds.
        for i in sorted(feedback):
            st.divider()
            st.markdown(f"## 🕐 t={i} · Feedback Refinement")
            if i - 1 < len(user_msgs):
                req = (user_msgs[i - 1].get("content") or "").strip()
                if req:
                    st.caption(f"🎯 Request: {req}")
            fb_stages = []
            for j in feedback[i]:
                r = f"{i}-{j}"
                fb_stages.append(
                    {
                        "title": f"🔧 Refiner Round {j}",
                        "desc_key": f"{prefix}_critic_desc{r}",
                        "img_key": f"{prefix}_critic_desc{r}_base64_jpg",
                        "suggestions_key": f"{prefix}_critic_suggestions{r}",
                        "round_suffix": r,
                    }
                )
            _render_stage_grid(fb_stages, item, task_type)

        # Standalone critique field (legacy)
        if item.get("critique0"):
            with st.expander("💬 Critic's Feedback", expanded=False):
                st.write(item["critique0"])


# ---------------------------------------------------------------------------
# Global stats (rendered once per turn)
# ---------------------------------------------------------------------------
def render_global_stats(data, dimensions, show_preference, show_referenced):
    """Render global statistics for one turn."""
    if len(data) == 0:
        return

    if show_preference:
        stats = calculate_preference_stats(data)
        with st.expander("📊 Global Preference Statistics", expanded=False):
            col_overall, col_status, col_category, col_priority = st.columns(4)
            with col_overall:
                st.subheader("Overall")
                st.metric("Mean Score", f"{stats['overall_score']:.1%}")
                st.caption(f"Total cases: {stats['total_data_points']}")
                st.caption(f"Total preferences: {stats['total_preferences']}")
            with col_status:
                st.subheader("Status Breakdown")
                sc = stats["status_counts"]
                tot = sum(sc.values()) or 1
                st.markdown(f":green[**Satisfied:**] {sc['Satisfied']} ({sc['Satisfied']/tot:.1%})")
                st.markdown(f":red[**Unsatisfied:**] {sc['Unsatisfied']} ({sc['Unsatisfied']/tot:.1%})")
                if sc.get("Error", 0) > 0:
                    st.markdown(f":gray[**Error:**] {sc['Error']} ({sc['Error']/tot:.1%})")
            with col_category:
                st.subheader("By Category")
                for cat, val in stats["categories"].items():
                    st.markdown(f"**{cat}:** `{val:.1%}`")
            with col_priority:
                st.subheader("By Priority")
                for prio, val in stats["priorities"].items():
                    st.markdown(f"**{prio}:** `{val:.1%}`")

    if show_referenced:
        stats = calculate_referenced_stats(data, dimensions)
        with st.expander("📊 Global Model vs Human Statistics", expanded=False):
            cols = st.columns(len(dimensions))
            for i, dim in enumerate(dimensions):
                with cols[i]:
                    st.info(f"**{dim}**")
                    s = stats[dim]
                    total = sum(s.values()) or 1
                    st.metric("Model Win", f"{(s['Model'])/total:.1%}")
                    st.metric("Human Win", f"{(s['Human'])/total:.1%}")
                    if dim == "Overall":
                        st.metric("Tie", f"{s.get('Tie', 0)/total:.1%}")
                    else:
                        st.metric("Both Good", f"{(s['Both are good'])/total:.1%}")
                        st.metric("Both Bad", f"{(s['Both are bad'])/total:.1%}")


# ---------------------------------------------------------------------------
# Per-turn item rendering
# ---------------------------------------------------------------------------
def render_item_views(
    item,
    version_label,
    dimensions,
    show_preference,
    show_referenced,
    show_evolution,
    expand_by_default,
    forgotten=None,
):
    """Render all enabled views for one version of an item.

    Flag requirements that were satisfied earlier but are now forgotten.
    """
    item_task_type = detect_task_type(item)
    forgotten = forgotten or []

    with st.container():
        # Requirements satisfied earlier but now forgotten: flagged inline, on
        # the same line as the turn title (indices only -- details are below).
        title = f"### {version_label}"
        if forgotten:
            title += f" &nbsp;&nbsp; :red[Forgotten: {', '.join(f'#{j}' for j in forgotten)}]"
        st.markdown(title)

        if show_referenced:
            display_referenced_eval(item, item_task_type, expand_by_default)

        # Shared raw content (method section / raw data)
        raw_content = item.get("content", "N/A")
        if isinstance(raw_content, dict):
            with st.expander("📚 Raw Data", expanded=False):
                st.code(json.dumps(raw_content, indent=2), language="json")
        else:
            with st.expander("📚 Method Section", expanded=False):
                st.markdown(str(raw_content))

        # Comparison reasoning (below method section, above preference eval)
        if show_referenced:
            display_comparison_eval(item, dimensions, expand_by_default)

        if show_preference:
            display_preference_eval(item, expand_by_default)

        if show_evolution:
            display_evolution_stages(item, item_task_type, expand_by_default)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("🍌 PaperBanana-Interact Visualizer")
    dir_path = st.sidebar.text_input(
        "Results Directory Path", placeholder="Enter path to results dir (containing 0.json, 1.json, ... N.json)..."
    )

    # --- View toggles ---
    st.sidebar.subheader("Views")
    show_preference = st.sidebar.checkbox("⭐ Preference Eval", value=True)
    show_referenced = st.sidebar.checkbox("⚖️ Evaluation Overview", value=True)
    show_evolution = st.sidebar.checkbox("🪜 Evolution Stages", value=True)

    st.sidebar.caption("Each enabled view renders in its own expander per item.")
    expand_by_default = st.sidebar.checkbox("Expand views by default", value=False)

    if st.sidebar.button("🔄 Refresh Data"):
        load_data.clear()
        st.rerun()

    # Navigation stays available before a results directory is selected.
    st.sidebar.divider()
    st.sidebar.markdown("**Turns**")
    turns_input = st.sidebar.text_input(
        "Turns to display",
        placeholder="All turns (or enter 0, 1, 2)",
    )
    st.sidebar.divider()
    search_query = st.sidebar.text_input(
        "🔍 Search Id", value="", help="Exact match by id (case-insensitive), e.g. id_1 won't match id_10"
    )
    page_size = st.sidebar.number_input("Items per page", min_value=1, max_value=50, value=5, step=1)
    requested_page = st.sidebar.number_input("Page", min_value=1, value=1, step=1)

    if not dir_path:
        st.stop()

    if not os.path.isdir(dir_path):
        st.error(f"Directory not found: {dir_path}")
        st.stop()

    # --- Discover valid turn files ---
    valid_turns, diag = discover_turns(dir_path)
    if not valid_turns:
        st.error(f"No valid `<N>.json` turn files found in `{dir_path}`. " "Expected files like 0.json, 1.json, ...")
        st.stop()

    # Report what was found / dropped.
    def _count_str(t):
        c = diag["counts"].get(t)
        return "EOF" if c is None else str(c)

    st.sidebar.caption("Turn files: " + ", ".join(f"{t}.json({_count_str(t)})" for t in sorted(diag["counts"])))
    if diag["dropped"]:
        st.sidebar.warning(
            f"Dropped (unparseable or != {diag['full']} items): " + ", ".join(f"{t}.json" for t in diag["dropped"])
        )

    # A blank turn filter means all available turns.
    try:
        selected_turns = sorted({int(t.strip()) for t in turns_input.split(",") if t.strip()})
    except ValueError:
        st.warning("Turns must be comma-separated numbers, for example: 0, 1, 2")
        st.stop()
    selected_turns = [t for t in selected_turns if t in valid_turns] if selected_turns else valid_turns
    if not selected_turns:
        st.warning("None of the requested turns are available.")
        st.stop()

    # Load the selected turns. Pagination below is the only image-loading
    # boundary: every image on the current page renders normally, while later
    # pages are not rendered until the user navigates to them.
    per_turn = {}  # turn -> {id: item}
    turn_data = {}  # turn -> list (for global stats)
    for t in selected_turns:
        data = load_data(os.path.join(dir_path, f"{t}.json"))
        if not data:
            st.warning(f"{t}.json is empty or unreadable; skipping.")
            continue
        turn_data[t] = data
        per_turn[t] = {x.get("id"): x for x in data}
    turns = sorted(per_turn)
    if not turns:
        st.warning("No selected turn loaded successfully.")
        return

    # Ids present in EVERY selected turn, ordered by numeric id.
    ids = set.intersection(*[set(per_turn[t]) for t in turns])
    ids = sorted(ids, key=lambda i: id_sort_key({"id": i}))

    task_type = detect_dataset_task_type(turn_data[turns[0]])

    dimensions = ["Faithfulness", "Conciseness", "Readability", "Aesthetics", "Overall"]

    if search_query:
        query = search_query.strip().lower()
        ids = [i for i in ids if str(i).strip().lower() == query]
        st.sidebar.caption(f"Found {len(ids)} matching cases")

    total_filtered_items = len(ids)

    st.title("🍌 PaperBanana-Interact Visualizer")
    st.caption(
        f"Task type: **{task_type}** · Turns: {', '.join(str(t) for t in turns)} · "
        f"Views: {', '.join([n for n, on in [('Preference', show_preference), ('Eval Overview', show_referenced), ('Evolution', show_evolution)] if on]) or 'none'}"
    )

    # --- Cross-turn trajectory summary ---
    if total_filtered_items > 0:
        rows = compute_trajectory_stats(per_turn, turns, ids)
        later_turns = [t for t in turns if t >= 2]
        with st.expander("📈 Trajectory Summary (implemented / later forgotten)", expanded=True):
            st.caption(
                "Per surfacing turn k: how many requests surfaced that turn were "
                "implemented, then how many of those had been forgotten again by "
                "turn t. Cumulative — once forgotten, it stays counted. Out of "
                "implemented requests that got at least one later real edit."
            )
            header = ["Surfaced", "Implemented"] + [f"forgot@t{t}" for t in later_turns] + ["forgot@any"]
            table = []
            for r in rows:
                impl, n, denom = r["impl"], r["n"], r["denom"]
                row = [f"Turn {r['turn']}", f"{impl}/{n}" + (f" ({impl/n:.0%})" if n else "")]
                for t in later_turns:
                    if t <= r["turn"] or denom == 0:
                        row.append("—")
                    else:
                        f = r["forgot"][t]
                        row.append(f"{f}/{denom} ({f/denom:.0%})")
                fa = r["forgot_any"]
                row.append(f"{fa}/{denom} ({fa/denom:.0%})" if denom else "—")
                table.append(row)
            if table:
                st.dataframe([dict(zip(header, row)) for row in table], use_container_width=True, hide_index=True)
            else:
                st.info("No surfaced requests across the selected turns.")

    # --- Per-turn global stats ---
    if total_filtered_items > 0:
        with st.expander("📊 Per-turn Global Statistics", expanded=False):
            for t in turns:
                st.markdown(f"#### Turn {t}")
                render_global_stats(turn_data[t], dimensions, show_preference, show_referenced)

    st.divider()

    if total_filtered_items == 0:
        st.warning(
            f"No samples found matching '{search_query}'."
            if search_query
            else "No ids are present in all selected turns."
        )
        return

    # --- Pagination ---
    total_pages = max((total_filtered_items + page_size - 1) // page_size, 1)
    page = min(requested_page, total_pages)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_filtered_items)
    batch_ids = ids[start_idx:end_idx]

    st.sidebar.markdown(f"Displaying {start_idx + 1} - {end_idx} of {total_filtered_items}")
    st.markdown(f"**Displaying {start_idx + 1} - {end_idx} of {total_filtered_items}**")

    # --- Per-item rendering: trajectory filmstrip + per-turn detail tabs ---
    for i, id_ in enumerate(batch_ids):
        idx = start_idx + i
        trajectory = {t: per_turn[t][id_] for t in turns}
        first_item = trajectory[turns[0]]
        caption_or_desc = first_item.get("visual_intent") or first_item.get("brief_desc", "N/A")

        with st.container(border=True):
            title = caption_or_desc[:120] + "..." if len(str(caption_or_desc)) > 120 else caption_or_desc
            st.subheader(f"#{idx + 1}: {title}")
            st.caption(f"ID: `{id_}`")

            # Trajectory stream: one row per turn, each with its utterance +
            # eval views on the left and the resulting image on the right.
            render_filmstrip(
                trajectory, turns, dimensions, show_preference, show_referenced, show_evolution, expand_by_default
            )

            st.divider()


if __name__ == "__main__":
    main()
