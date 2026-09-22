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

import base64
import copy
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
import yaml
from PIL import Image

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.polish_agent import PolishAgent
from agents.refiner import PBDirectRefine, PBInteract
from agents.retriever_agent import RetrieverAgent
from agents.stylist_agent import StylistAgent
from agents.vanilla_agent import VanillaAgent
from agents.visualizer_agent import VisualizerAgent
from utils import config, generation_utils
from utils.base64_rw_helper import read_base64_sync
from utils.paperviz_processor import PaperVizProcessor

try:  # have a dummy zero-gpu for hf spaces deployment
    import spaces

    @spaces.GPU
    def dummy_gpu_function():
        pass

except:
    pass


ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "configs" / "model_config.yaml"
_DEFAULTS = (
    (yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}).get("defaults", {}) if CONFIG_PATH.exists() else {}
)
DEFAULT_MAIN_MODEL = _DEFAULTS.get("main_model_name") or "gemini-3.1-pro-preview"
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image"
generation_utils.set_api_concurrency(8)
APP_CSS = (
    ".gradio-container { --button-secondary-background-fill: var(--neutral-100);"
    " --button-secondary-background-fill-hover: var(--neutral-100); }"
    " button.secondary { --button-border-width: 1px; }"
    " .settings-accordion { background: var(--button-secondary-background-fill) !important; }"
    " .settings-accordion .form > .block { padding: var(--spacing-xl) 9px; }"
    " .settings-accordion .form { padding: 0 9px; background: var(--block-background-fill); }"
    " .section-title { margin-top: .5rem !important; }"
    " .title-row { align-items: center !important; }"
    " html { scrollbar-gutter: stable; }"
    " .candidate-box, .candidate-box .row > .column, .cols-2 .row > .column, .api-key { background: var(--block-background-fill);"
    " border: var(--block-border-width) solid var(--block-border-color) !important; border-radius: var(--block-radius); padding: var(--block-padding); }"
    " .candidate-box .row > .column { flex: 0 0 calc((100% - 3 * var(--layout-gap)) / 4) !important; min-width: 0 !important; }"
    " .cols-2 .row > .column { flex: 0 0 calc((100% - var(--layout-gap)) / 2) !important; min-width: 0 !important; }"
    " .cols-2 .row > .column.tile-full { flex-basis: 100% !important; }"
    " .chat-scroll { height: 650px; overflow-y: auto; flex-wrap: nowrap !important;"
    " background: var(--block-background-fill); border: var(--block-border-width) solid var(--block-border-color) !important;"
    " border-radius: var(--block-radius); padding: var(--block-padding); }"
    " .chat-scroll > * { flex-shrink: 0 !important; }"
    " .chat-log { height: auto !important; max-height: none !important; border: none !important; background: transparent !important; }"
    " .chat-log [role='log'] { height: auto !important; overflow-y: visible !important; }"
    " .chat-log .icon-button-wrapper { display: none !important; }"
    " .candidate-placeholder { overflow: hidden !important; }"
    " .candidate-placeholder:has(.wrap:not(.hide)) { min-height: var(--size-12); }"
    " .iter-nav > *, .api-key, .api-key > *, .badges, .badges > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }"
    " .iter-nav, .api-key, .badges { flex-wrap: nowrap !important; align-items: center !important; gap: var(--spacing-lg) !important; }"
    " .iter-nav { justify-content: flex-end; }"
    " .badges { margin: calc(var(--spacing-sm) - var(--layout-gap)) 0 var(--spacing-lg); }"
    " .tabs { display: contents; }"
    " .tabs > .tab-wrapper { order: 1; }"
    " .settings-row { order: 2; }"
    " .tabs > .tabitem { order: 3; }"
    " .gradio-container:has([role='tablist'] > button:first-child[aria-selected='true'])"
    " .refiner-accordion { display: none; }"
    " .gradio-container:has([role='tablist'] > button:nth-child(3)[aria-selected='true'])"
    " .settings-row { display: none; }"
)


EXAMPLE_METHOD = r"""## Methodology: The PaperBanana Framework

PaperBanana is a reference-driven agentic framework for automated academic illustration. A Retriever selects relevant examples from a fixed reference set. A Planner converts the source context and communicative intent into a detailed visual description. An optional Stylist improves the presentation, then a Visualizer renders the diagram. Finally, a Critic inspects the image against the original source and iteratively proposes corrections for the Visualizer. The loop produces a publication-quality scientific diagram."""

EXAMPLE_CAPTION = (
    "Figure 1: Overview of PaperBanana. The framework retrieves relevant "
    "examples, plans and styles a visual description, generates a diagram, "
    "and improves it through an iterative Visualizer-Critic loop."
)

USAGE = """Generate a scientific diagram from your paper's method section, then refine it through conversation.

1. **Enter your 🔑 Gemini API Key** in the top-right box. It is only used for your own requests in this session.
2. **Generate a first draft.** Paste your method section and figure caption (or click **Load Example**) and press **🚀 Generate Candidates**. This works on both the **📊 Single-Turn Generation** tab and the **💬 Iterative Refinement** tab. Each candidate can be saved with **⬇️ Download**; where applicable, **🔄 View Evolution Timeline** shows the internal evolution stages inside the agent.
3. **Refine it in conversation.** Click **✏️ Refine** under the candidate you like to carry it into the **💬 Iterative Refinement** tab. **📌 Pin** the candidate to iterate on, then describe the change you want in the chat box, for example "Use a horizontal layout and highlight the critic loop in orange". Every message produces a new turn of candidates.
4. **Browse and revert.** Use ◀ ▶ to move between turns. **⏪ Revert to this turn** discards every later turn so you can branch again from the one you are viewing.

Note: each generation or refinement takes a few minutes and makes several model calls per candidate."""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def decode_image(value: Any) -> Image.Image | None:
    encoded = read_base64_sync(value)
    if not encoded:
        return None
    try:
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        image.load()
        return image
    except Exception:
        return None


def make_processor(exp_config: config.ExpConfig, refiner_agent=None) -> PaperVizProcessor:
    return PaperVizProcessor(
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


async def generate_candidates(
    method: str,
    caption: str,
    exp_mode: str,
    retrieval_setting: str,
    count: int,
    ratio: str,
    rounds: int,
    main_model: str,
    image_model: str,
    api_key: str,
) -> list[dict]:
    """Run the single-turn pipeline (raises gr.Error on failure)."""
    api_key = (api_key or "").strip()  # an untouched Textbox yields None, not ""
    generation_utils.api_key_overrides.set({"google_api_key": api_key} if api_key else {})
    exp_config = config.ExpConfig(
        dataset_name="MTPaperBananaBench",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        main_model_name=main_model,
        image_gen_model_name=image_model,
    )
    # Only the fields the pipeline consumes: content, visual_intent, rounded_ratio, max_critic_rounds.
    data_list = [
        {
            "content": method,
            "visual_intent": caption,
            "additional_info": {"rounded_ratio": ratio},
            "max_critic_rounds": int(rounds),
        }
        for _ in range(int(count))
    ]
    processor = make_processor(exp_config)
    try:
        results = [result async for result in processor.process_queries_batch(data_list, do_eval=False)]
    except Exception as error:
        raise gr.Error(f"Generation failed: {error}") from error
    return results


async def refine_candidates(
    pinned_result: dict,
    chat_history: list[dict],
    count: int,
    refiner_name: str,
    retrieval_setting: str,
    main_model: str,
    image_model: str,
    internal_rounds: int,
    exp_mode: str,
    api_key: str,
) -> list[dict]:
    """Run the refiner on the pinned candidate (raises gr.Error on failure)."""
    api_key = (api_key or "").strip()  # an untouched Textbox yields None, not ""
    generation_utils.api_key_overrides.set({"google_api_key": api_key} if api_key else {})
    exp_config = config.ExpConfig(
        dataset_name="MTPaperBananaBench",
        split_name="demo",
        exp_mode="dev_feedback",
        retrieval_setting=retrieval_setting,
        main_model_name=main_model,
        image_gen_model_name=image_model,
        refiner=refiner_name,
        refiner_internal_critic_rounds=int(internal_rounds),
    )
    refiner = {"pb_directrefine": PBDirectRefine, "pb_interact": PBInteract}[refiner_name](exp_config=exp_config)
    processor = make_processor(exp_config, refiner)
    batch = [
        copy.deepcopy(pinned_result)
        | {"chat_history": chat_history, "user_input": chat_history[-1]["content"], "candidate_id": index}
        for index in range(int(count))
    ]
    try:
        results = [
            result async for result in processor.process_feedback_batch(batch, simulate_user=False, do_eval=False)
        ]
    except Exception as error:
        raise gr.Error(f"Refinement failed: {error}") from error
    return results


def final_image_and_description(result: dict, exp_mode: str) -> tuple[Image.Image | None, str]:
    # The processor always sets eval_image_field to the current best image; the
    # fallback only guards against malformed results.
    image_key = result.get("eval_image_field") or (
        "target_diagram_stylist_desc0_base64_jpg" if exp_mode == "demo_full" else "target_diagram_desc0_base64_jpg"
    )
    description_key = image_key.removesuffix("_base64_jpg")
    return decode_image(result.get(image_key)), clean_text(result.get(description_key))


def evolution_stages(result: dict, mode: str) -> list[tuple[str, str, str, str | None]]:
    rounds = sorted(
        (m.group(1) for m in (re.fullmatch(r"target_diagram_critic_desc(\d+-\d+)", k) for k in result) if m),
        key=lambda s: [int(p) for p in s.split("-")],
    )
    if rounds:  # refinement turn: only this turn's internal rounds, minus the main image shown above
        turn = rounds[-1].split("-")[0]
        stages = [
            (
                f"💬 Iteration {j}",
                f"target_diagram_critic_desc{i}-{j}",
                f"Internal iteration of PaperBanana-Interact based on user feedback (round {j})",
                f"target_diagram_critic_suggestions{i}-{j}",
            )
            for i, j in (r.split("-") for r in rounds)
            if i == turn and f"target_diagram_critic_desc{i}-{j}_base64_jpg" != result.get("eval_image_field")
        ]
    else:
        stages = [("📋 Planner", "target_diagram_desc0", "Initial diagram plan based on method content", None)]
        if mode == "demo_full":
            stages.append(("✨ Stylist", "target_diagram_stylist_desc0", "Stylistically refined description", None))
        for round_index in range(5):
            stages.append(
                (
                    f"🔍 Critic Round {round_index}",
                    f"target_diagram_critic_desc{round_index}",
                    f"Refined after critic feedback (iteration {round_index})",
                    f"target_diagram_critic_suggestions{round_index}",
                )
            )
    return [stage for stage in stages if result.get(f"{stage[1]}_base64_jpg")]


def candidate_tile(result: dict, index: int, mode: str, key_prefix: str, pin=None) -> None:
    """One candidate's image, buttons, description, and evolution timeline (both tabs).

    `pin` optionally renders an extra button next to the download button (Pin on the
    interactive tab, Refine on the single-turn tab).
    """
    image, description = final_image_and_description(result, mode)
    gr.Image(image, interactive=False, show_label=False, buttons=["fullscreen"])
    with gr.Row():
        download = gr.DownloadButton("⬇️ Download", key=f"{key_prefix}-download-{index}")

        def export_png():
            path = Path(tempfile.mkdtemp()) / f"candidate_{index}.png"
            image.save(path)
            return str(path)

        download.click(export_png, outputs=download)
        if pin:
            pin()
    with gr.Accordion("📝 Description", open=False):
        gr.Markdown(description or "No description available.")
    stages = evolution_stages(result, mode)
    if stages:
        with gr.Accordion(f"🔄 View Evolution Timeline ({len(stages)} stages)", open=False):
            gr.Markdown("See how the diagram evolved through different pipeline stages.")
            for name, key, summary, suggestion_key in stages:
                gr.Markdown(f"### {name}\n\n{summary}")
                gr.Image(decode_image(result.get(f"{key}_base64_jpg")), show_label=False)
                with gr.Accordion("📝 Description", open=False):
                    gr.Markdown(clean_text(result.get(key)) or "No description available.")
                if suggestion_key and result.get(suggestion_key):
                    with gr.Accordion("💡 Critic Suggestions", open=False):
                        gr.Markdown(clean_text(result[suggestion_key]))


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PaperBanana", fill_width=True) as app:
        generation_state = gr.State([])
        generation_mode_state = gr.State("demo_full")
        interactive_state = gr.State({"versions": [], "current": 0, "pinned": {}, "chat": []})

        # The session's API key
        with gr.Row(elem_classes="title-row"):
            gr.Markdown("# ⭐ PaperBanana-Interact Demo")
            with gr.Row(elem_classes="api-key"):
                gr.Markdown("🔑 Gemini API Key")
                google_api_key = gr.Textbox(
                    show_label=False,
                    container=False,
                    type="password",
                    min_width=480,
                    placeholder="AIza...",
                )

        with gr.Row(elem_classes="badges"):
            gr.Button("📄 arXiv", link="https://arxiv.org/abs/2608.30241", size="sm")
            gr.Button("💻 GitHub", link="https://github.com/shirley-wu/PaperBanana-Interact", size="sm")
            gr.Button("🤗 Dataset", link="https://huggingface.co/datasets/xqwu/MTPaperBananaBench", size="sm")

        # One settings block shared by both tabs
        with gr.Row(elem_classes="settings-row"):
            with gr.Accordion("⚙️ Overall Settings", open=False, elem_classes="settings-accordion"):
                with gr.Row():
                    main_model = gr.Textbox(
                        value=DEFAULT_MAIN_MODEL,
                        label="Main Model",
                        info="Text model for planning, critique and refinement.",
                        scale=2,
                    )
                    image_model = gr.Textbox(
                        value=DEFAULT_IMAGE_MODEL,
                        label="Image Model",
                        info="Image model that renders the diagram.",
                        scale=2,
                    )
                    ratio = gr.Dropdown(
                        ["21:9", "16:9", "3:2"],
                        value="21:9",
                        label="Aspect Ratio",
                        info="Width to height of the diagram.",
                        scale=1,
                        min_width=100,
                    )
            with gr.Accordion("🍌 PaperBanana Settings", open=False, elem_classes="settings-accordion"):
                with gr.Row():
                    mode = gr.Dropdown(
                        ["demo_full", "demo_planner_critic"],
                        value="demo_full",
                        label="Pipeline",
                        info="demo_planner_critic skips the Stylist.",
                        scale=2,
                    )
                    retrieval = gr.Dropdown(
                        ["auto", "manual", "random", "none"],
                        value="auto",
                        label="Retrieval",
                        info="How reference diagrams are selected.",
                        scale=1,
                        min_width=100,
                    )
                    count = gr.Number(
                        value=4,
                        minimum=1,
                        maximum=20,
                        step=1,
                        label="# Candidates",
                        info="How many candidates to generate.",
                        scale=1,
                        min_width=100,
                    )
                    rounds = gr.Number(
                        value=3,
                        minimum=0,
                        maximum=5,
                        step=1,
                        label="Critic Rounds",
                        info="Max critic iterations; 0 disables the Critic.",
                        scale=1,
                        min_width=100,
                    )
            with gr.Accordion("✏️ Refiner Settings", open=False, elem_classes="settings-accordion refiner-accordion"):
                with gr.Row():
                    refiner = gr.Dropdown(
                        ["pb_directrefine", "pb_interact"],
                        value="pb_directrefine",
                        label="Refiner",
                        info="pb_interact: PaperBanana-Interact. pb_directrefine: PaperBanana-DirectRefine.",
                        scale=2,
                    )
                    turn_count = gr.Number(
                        value=1,
                        minimum=1,
                        maximum=20,
                        step=1,
                        label="# Candidates",
                        info="How many refined candidates to generate per message.",
                        scale=1,
                        min_width=100,
                    )
                    iter_rounds = gr.Number(
                        value=1,
                        minimum=1,
                        maximum=5,
                        step=1,
                        label="Iteration Rounds",
                        info="Internal iterations per message (pb_interact only).",
                        scale=1,
                        min_width=100,
                        interactive=False,
                    )

        # A collapsed accordion rebuilds its fields from the frontend's last-known values
        for field in (main_model, image_model, ratio, mode, retrieval, count, rounds, refiner, turn_count, iter_rounds):
            field.change(lambda *_: None, field)
        # pb_directrefine has no internal loop: pin Iteration Rounds at 1 and grey it out; pb_interact unlocks it.
        refiner.change(
            lambda name: (
                gr.update(value=1, interactive=False) if name == "pb_directrefine" else gr.update(interactive=True)
            ),
            refiner,
            iter_rounds,
        )

        def validate_form(method, caption, results=()):
            if not method.strip() or not caption.strip():
                raise gr.Error("Please provide both method content and a figure caption.")
            text = "Re-generating candidates..." if results else "Generating candidates..."
            return text, *lock_form(True)

        def lock_form(locked):
            return [gr.update(interactive=not locked)] * 5

        def open_session(results, mode, pinned=0):
            """State and outputs that start an iterative session from a set of candidates."""
            state = {
                "versions": [{"results": results, "exp_mode": mode}],
                "current": 0,
                "pinned": {"0": pinned},
                "chat": [
                    {
                        "role": "assistant",
                        "content": f"I have generated {len(results)} candidates based on your method and "
                        "caption! You can view them on the left. What would you like to refine?",
                    }
                ],
            }
            # Reveal the chat input; the chained lock_form(True) greys the form out.
            return (
                f"Generated {len(results)} candidates. **Pin** the one you like best: "
                "your next message iterates on the pinned candidate.",
                state,
                state["chat"],
                gr.update(visible=True),
            )

        with gr.Tabs() as tabs:
            with gr.Tab("📊 Single-Turn Generation"):
                gr.Markdown("### Generate diagram candidates from a method section and caption (original PaperBanana)")
                with gr.Row():
                    generate_method = gr.Textbox(
                        label="Method Section Content",
                        placeholder="Paste the method section from your paper here...",
                        lines=14,
                        max_lines=28,
                    )
                    generate_caption = gr.Textbox(
                        label="Figure Caption",
                        placeholder="Enter the figure caption from your paper here...",
                        lines=14,
                    )
                with gr.Row():
                    load_generate_example = gr.Button("Load Example")
                    clear_generate = gr.Button("Clear")
                    generate_button = gr.Button("🚀 Generate Candidates", variant="primary")
                gr.Markdown("## 🎨 Generated Candidates", elem_classes="section-title")
                with gr.Column(elem_classes="candidate-box"):
                    # Doubles as the placeholder strip: always visible, shows the progress overlay while generating.
                    generation_status = gr.Markdown(
                        "Generated candidates will appear here.", elem_classes="candidate-placeholder"
                    )

                    @gr.render(inputs=[generation_state, generation_mode_state])
                    def render_candidates(results, mode):
                        if not results:
                            return
                        with gr.Row(key="candidate-row"):
                            for index, result in enumerate(results):
                                with gr.Column(key=f"candidate-{index}"):

                                    def render_refine(index=index):
                                        refine_button = gr.Button("✏️ Refine", key=f"candidate-refine-{index}")
                                        # Hand the whole session (method, caption, every candidate, this one
                                        # pinned) to the Iterative Refinement tab and switch to it.
                                        refine_button.click(
                                            lambda method, caption: (
                                                method,
                                                caption,
                                                *open_session(results, mode, index),
                                                gr.Tabs(selected="refine"),
                                            ),
                                            [generate_method, generate_caption],
                                            [
                                                interactive_method,
                                                interactive_caption,
                                                interactive_status,
                                                interactive_state,
                                                interactive_chat,
                                                chat_input,
                                                tabs,
                                            ],
                                            show_progress="hidden",
                                        ).then(
                                            lambda: lock_form(True), outputs=interactive_form, show_progress="hidden"
                                        )

                                    candidate_tile(result, index, mode, "candidate", pin=render_refine)

                load_generate_example.click(
                    lambda: (EXAMPLE_METHOD, EXAMPLE_CAPTION),
                    outputs=[generate_method, generate_caption],
                )
                clear_generate.click(lambda: ("", ""), outputs=[generate_method, generate_caption])

                async def run_generation(
                    method, caption, mode, retrieval_mode, count, ratio, rounds, text_model, img_model, api_key
                ):
                    results = await generate_candidates(
                        method, caption, mode, retrieval_mode, count, ratio, rounds, text_model, img_model, api_key
                    )
                    return (
                        f"Generated {len(results)} candidates.",
                        results,
                        mode,
                        gr.update(value="✏️ Re-Generate Candidates"),
                    )

                # Step 1 validates and swaps the status text instantly; step 2 is the real run with the normal overlay.
                generate_form = [
                    generate_method,
                    generate_caption,
                    load_generate_example,
                    clear_generate,
                    generate_button,
                ]
                generate_button.click(
                    validate_form,
                    inputs=[generate_method, generate_caption, generation_state],
                    outputs=[generation_status, *generate_form],
                    show_progress="hidden",
                ).success(
                    run_generation,
                    inputs=[
                        generate_method,
                        generate_caption,
                        mode,
                        retrieval,
                        count,
                        ratio,
                        rounds,
                        main_model,
                        image_model,
                        google_api_key,
                    ],
                    outputs=[generation_status, generation_state, generation_mode_state, generate_button],
                    show_progress_on=generation_status,  # keep the overlay off the button
                ).then(
                    lambda: lock_form(False), outputs=generate_form, show_progress="hidden"
                )

            with gr.Tab("💬 Iterative Refinement", id="refine"):
                gr.Markdown("### Generate candidates, then refine them through conversation")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 🎨 Generated Candidates", elem_classes="section-title")
                        with gr.Column(elem_classes="cols-2"):
                            # Doubles as the placeholder strip, like the single-turn tab: always
                            # visible, hosts the progress overlay while generating/refining.
                            interactive_status = gr.Markdown(
                                "No candidates generated yet. Please generate candidates to see results here.",
                                elem_classes="candidate-placeholder",
                            )

                            @gr.render(inputs=[interactive_state])
                            def render_nav(state):
                                versions = state.get("versions", [])
                                if len(versions) < 2:
                                    return
                                current = state["current"]
                                with gr.Row(key="iter-nav", elem_classes="iter-nav"):
                                    revert_button = gr.Button(
                                        "⏪ Revert to this turn",
                                        size="sm",
                                        key="iter-revert",
                                        interactive=current < len(versions) - 1 and not state.get("busy"),
                                    )
                                    previous_button = gr.Button(
                                        "◀", size="sm", key="iter-prev", interactive=current > 0
                                    )
                                    gr.Markdown(f"Turn {current + 1} / {len(versions)}", key="iter-nav-label")
                                    next_button = gr.Button(
                                        "▶", size="sm", key="iter-next", interactive=current < len(versions) - 1
                                    )

                                def shift_version(state, delta):
                                    current = max(0, min(state["current"] + delta, len(state["versions"]) - 1))
                                    return {**state, "current": current}

                                previous_button.click(
                                    lambda state: shift_version(state, -1), interactive_state, interactive_state
                                )
                                next_button.click(
                                    lambda state: shift_version(state, 1), interactive_state, interactive_state
                                )

                                def revert_version(state):
                                    # Cut the session at the viewed turn: later versions and chat dropped
                                    cut = state["current"]
                                    state = {
                                        **state,
                                        "versions": state["versions"][: cut + 1],
                                        "chat": state["chat"][: 1 + 2 * cut],
                                    }
                                    return (
                                        f"Reverted to turn {cut + 1}. **Pin** the one you like best: "
                                        "your next message iterates on the pinned candidate.",
                                        state,
                                        state["chat"],
                                    )

                                revert_button.click(
                                    revert_version,
                                    interactive_state,
                                    [interactive_status, interactive_state, interactive_chat],
                                )

                            @gr.render(inputs=[interactive_state])
                            def render_versions(state):
                                versions = state.get("versions", [])
                                index = state.get("current", 0)
                                version = versions[index] if versions and 0 <= index < len(versions) else None
                                if version is None:
                                    return
                                results, mode = version["results"], version["exp_mode"]
                                version_index = state["current"]
                                is_latest = version_index == len(versions) - 1
                                pinned = state.get("pinned", {}).get(str(version_index))
                                for start in range(0, len(results), 2):
                                    with gr.Row(key=f"iter-row-{start}"):
                                        for index in range(start, min(start + 2, len(results))):

                                            def render_pin(index=index):
                                                pin_key = f"iter-v{version_index}-pin-{index}"
                                                if index == pinned:
                                                    gr.Button("✅ Pinned", variant="primary", key=pin_key)
                                                else:
                                                    pin_button = gr.Button(
                                                        "📌 Pin",
                                                        interactive=is_latest and not state.get("busy"),
                                                        key=pin_key,
                                                    )

                                                    def pin_candidate(state, selected=index):
                                                        return {
                                                            **state,
                                                            "pinned": {
                                                                **state["pinned"],
                                                                str(state["current"]): selected,
                                                            },
                                                        }

                                                    pin_button.click(
                                                        pin_candidate, interactive_state, interactive_state
                                                    )

                                            with gr.Column(
                                                key=f"iter-candidate-{index}",
                                                elem_classes="tile-full" if len(results) == 1 else None,
                                            ):
                                                candidate_tile(
                                                    results[index],
                                                    index,
                                                    mode,
                                                    f"iter-v{version_index}",
                                                    pin=render_pin,
                                                )

                    with gr.Column(scale=1):
                        gr.Markdown("## 💬 Chat", elem_classes="section-title")
                        with gr.Column(elem_classes="chat-scroll"):
                            interactive_method = gr.Textbox(
                                label="Method Section Content",
                                lines=6,
                                max_lines=14,
                                placeholder="Paste the method section content here...",
                            )
                            interactive_caption = gr.Textbox(
                                label="Figure Caption", lines=3, placeholder="Enter the figure caption..."
                            )
                            with gr.Row():
                                load_interactive_example = gr.Button("Load Example")
                                clear_interactive = gr.Button("Clear")
                                interactive_generate = gr.Button("🚀 Generate Candidates", variant="primary")
                            interactive_chat = gr.Chatbot(
                                show_label=False, height="auto", elem_classes="chat-log", buttons=[]
                            )
                        chat_input = gr.Textbox(
                            placeholder="Describe what to change...",
                            show_label=False,
                            visible=False,
                            submit_btn=True,
                        )

                load_interactive_example.click(
                    lambda: (EXAMPLE_METHOD, EXAMPLE_CAPTION),
                    outputs=[interactive_method, interactive_caption],
                )
                clear_interactive.click(lambda: ("", ""), outputs=[interactive_method, interactive_caption])

                async def start_interactive(
                    method, caption, mode, retrieval_mode, count, ratio, rounds, text_model, img_model, api_key
                ):
                    results = await generate_candidates(
                        method, caption, mode, retrieval_mode, count, ratio, rounds, text_model, img_model, api_key
                    )
                    return open_session(results, mode)

                # Step 1 validates and swaps the status text instantly; step 2 is the real run.
                interactive_form = [
                    interactive_method,
                    interactive_caption,
                    load_interactive_example,
                    clear_interactive,
                    interactive_generate,
                ]
                interactive_generate.click(
                    validate_form,
                    inputs=[interactive_method, interactive_caption],
                    outputs=[interactive_status, *interactive_form],
                    show_progress="hidden",
                ).success(
                    start_interactive,
                    inputs=[
                        interactive_method,
                        interactive_caption,
                        mode,
                        retrieval,
                        count,
                        ratio,
                        rounds,
                        main_model,
                        image_model,
                        google_api_key,
                    ],
                    outputs=[interactive_status, interactive_state, interactive_chat, chat_input],
                    show_progress_on=interactive_status,
                ).then(
                    lambda state: lock_form(bool(state["versions"])),
                    interactive_state,
                    interactive_form,
                    show_progress="hidden",
                )

                def queue_feedback(state, message):
                    if state.get("busy"):
                        raise gr.Error("A refinement is still running. Wait for it to finish.")
                    message = message.strip()
                    if not message:
                        raise gr.Error("Type your message first.")
                    if not state.get("versions"):
                        raise gr.Error("Generate candidates before sending feedback.")
                    # busy locks the pin buttons (see render_versions) until the chain's final step clears it.
                    state = {**state, "busy": True, "chat": state["chat"] + [{"role": "user", "content": message}]}
                    return "Refining candidates...", state, state["chat"], ""

                async def run_feedback(
                    state, refiner_name, retrieval_mode, turn_count, text_model, img_model, iter_rounds, api_key
                ):
                    latest_index = len(state["versions"]) - 1
                    latest = state["versions"][latest_index]
                    pinned = state.get("pinned", {}).get(str(latest_index), 0)
                    if pinned >= len(latest["results"]):
                        pinned = 0
                    results = await refine_candidates(
                        latest["results"][pinned],
                        state["chat"],
                        turn_count,
                        refiner_name,
                        retrieval_mode,
                        text_model,
                        img_model,
                        iter_rounds,
                        latest["exp_mode"],
                        api_key,
                    )
                    state = {
                        **state,
                        "versions": state["versions"] + [{"results": results, "exp_mode": latest["exp_mode"]}],
                        "current": latest_index + 1,
                        "pinned": {**state["pinned"], str(latest_index + 1): 0},
                        "chat": state["chat"]
                        + [
                            {
                                "role": "assistant",
                                "content": "I have refined the diagram based on your feedback! "
                                "What else would you like to refine?",
                            }
                        ],
                    }
                    return (
                        f"Generated {len(results)} candidates. **Pin** the one you like best: "
                        "your next message iterates on the pinned candidate.",
                        state,
                        state["chat"],
                    )

                # Same two-step pattern as generation: show the user's message instantly, then run.
                chat_input.submit(
                    queue_feedback,
                    [interactive_state, chat_input],
                    [interactive_status, interactive_state, interactive_chat, chat_input],
                    show_progress="hidden",
                    show_progress_on=chat_input,
                ).success(
                    run_feedback,
                    [
                        interactive_state,
                        refiner,
                        retrieval,
                        turn_count,
                        main_model,
                        image_model,
                        iter_rounds,
                        google_api_key,
                    ],
                    [interactive_status, interactive_state, interactive_chat],
                    show_progress_on=interactive_status,
                ).then(
                    lambda state: {**state, "busy": False},
                    interactive_state,
                    interactive_state,
                    show_progress="hidden",
                )

            with gr.Tab("📖 How to use"):
                gr.Markdown(USAGE)

    return app


if __name__ == "__main__":
    build_app().launch(share=False, css=APP_CSS)
