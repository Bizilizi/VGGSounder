"""
Step 3: Gradio review app for Gemini-proposed label corrections.

Reads gemini_proposals.csv (produced by 02_gemini_verify.py) and lets the
annotator quickly accept / modify / reject each proposal.

Saves output.csv in the vggsounder+background-music.csv format:
    video_id, label, modality, background_music, static_image, voice_over
"""

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import vggsounder as vggs
from huggingface_hub import hf_hub_download

SCRIPT_DIR = Path(__file__).parent
HF_REPO = "11hu83/vggsound"
MAX_LABELS = 20


class ReviewBackend:
    def __init__(self, proposals_csv: Path, output_path: Path):
        self.output_path = output_path
        self.vgg = vggs.VGGSounder(background_music=None)
        self.tasks: list[dict] = []

        self.proposals_df = pd.read_csv(proposals_csv, keep_default_na=False)
        self.video_ids = self.proposals_df["video_id"].unique().tolist()
        self.proposals_by_vid = self.proposals_df.groupby("video_id", sort=False)

        self.saved_states: dict[str, dict] = {}
        self._load_saved_states()
        self.initial_annotated = set(self.saved_states.keys())
        self.session_annotated: set[str] = set()
        self._prepare_tasks()

    def _load_saved_states(self):
        if not self.output_path.exists():
            return
        try:
            df = pd.read_csv(self.output_path, keep_default_na=False)
        except Exception:
            return
        for vid_id, group in df.groupby("video_id", sort=False):
            results = {}
            for row in group.itertuples():
                if row.label:
                    results[row.label] = row.modality
            self.saved_states[vid_id] = results

    def _prepare_tasks(self):
        for vid_id in self.video_ids:
            proposals = self.proposals_by_vid.get_group(vid_id)
            candidate_labels: list[str] = []
            gemini_info: dict[str, dict] = {}
            seen: set[str] = set()

            # Original dataset labels for this video
            try:
                vid_data = self.vgg[vid_id]
                original_labels = list(zip(vid_data.labels, vid_data.modalities))
                meta = vid_data.meta_labels
            except (KeyError, IndexError):
                original_labels = []
                meta = {
                    "background_music": False,
                    "static_image": False,
                    "voice_over": False,
                }

            for lbl, _mod in original_labels:
                if lbl not in seen:
                    seen.add(lbl)
                    candidate_labels.append(lbl)

            for row in proposals.itertuples():
                lbl = row.label
                if lbl and lbl not in seen:
                    seen.add(lbl)
                    candidate_labels.append(lbl)
                gemini_info[lbl] = {
                    "label_correct": row.label_correct,
                    "modality_correct": row.modality_correct,
                    "suggested_modality": getattr(row, "suggested_modality", ""),
                    "reason": getattr(row, "reason", ""),
                }

                suggested_raw = getattr(row, "suggested_labels", "[]")
                if suggested_raw:
                    try:
                        suggestions = json.loads(suggested_raw)
                    except (json.JSONDecodeError, TypeError):
                        suggestions = []
                    for s in suggestions:
                        sl = s.get("label", "")
                        if sl and sl not in seen:
                            seen.add(sl)
                            candidate_labels.append(sl)
                            gemini_info[sl] = {
                                "label_correct": True,
                                "modality_correct": True,
                                "suggested_modality": s.get("modality", ""),
                                "reason": "(suggested by Gemini)",
                            }

            self.tasks.append(
                {
                    "video_id": vid_id,
                    "candidate_labels": candidate_labels,
                    "gemini_info": gemini_info,
                    "meta": meta,
                }
            )

    def get_task(self, index: int):
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        return None

    def save_verdict(self, video_id: str, results: dict[str, str], meta: dict):
        self.saved_states[video_id] = results
        if any(mod for mod in results.values()):
            self.session_annotated.add(video_id)

        rows = []
        for lbl, mod in results.items():
            if mod:
                rows.append(
                    {
                        "video_id": video_id,
                        "label": lbl,
                        "modality": mod,
                        "background_music": meta.get("background_music", False),
                        "static_image": meta.get("static_image", False),
                        "voice_over": meta.get("voice_over", False),
                    }
                )
        if not rows:
            rows.append(
                {
                    "video_id": video_id,
                    "label": "",
                    "modality": "",
                    "background_music": meta.get("background_music", False),
                    "static_image": meta.get("static_image", False),
                    "voice_over": meta.get("voice_over", False),
                }
            )

        header = not self.output_path.exists()
        df = pd.DataFrame(rows)
        df.to_csv(self.output_path, mode="a", index=False, header=header)

    def get_video_path(self, video_id: str) -> str | None:
        try:
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"video/{video_id}/video.mp4",
                repo_type="dataset",
            )
            tmp = os.path.join(tempfile.gettempdir(), f"vgg_{video_id}.mp4")
            if not os.path.exists(tmp):
                shutil.copy(path, tmp)
            return tmp
        except Exception as e:
            print(f"Error downloading {video_id}: {e}")
            return None


def create_app():
    backend = ReviewBackend(
        SCRIPT_DIR / "gemini_proposals.csv",
        SCRIPT_DIR / "output.csv",
    )

    custom_css = """
    .orange-btn { background-color: #FFA500 !important; color: black !important; border: none !important; font-weight: bold !important; }
    .orange-btn:hover { background-color: #FF8C00 !important; }
    .submit-btn { background-color: #FFA500 !important; color: black !important; font-weight: bold !important; border: none !important; width: 100px !important; margin-left: auto !important; display: block !important; }
    .nav-btn { background-color: #FFA500 !important; color: black !important; font-weight: bold !important; border: none !important; }
    .label-row { margin: 6px 0 !important; padding: 8px 12px !important; align-items: center; border: 1px solid var(--border-color-primary, #e5e7eb) !important; border-radius: 8px !important; transition: all 0.2s ease-in-out !important; background-color: var(--background-fill-secondary, transparent); }
    .header-row { margin-bottom: 10px !important; align-items: center; padding: 0 12px; }
    .center-nav { display: flex; justify-content: center; gap: 15px; }
    .top-bar-btn { background: transparent !important; border: 1px solid var(--border-color-primary, #ccc) !important; border-radius: 6px !important; color: var(--body-text-color, #333) !important; }
    .no-margin { margin: 0 !important; padding: 0 !important; }
    .hide-checkbox-text span.ml-2 { display: none !important; }
    .active-row { background-color: #fff3e0 !important; border-radius: 8px !important; border-color: #ffcc80 !important; box-shadow: 0 2px 6px rgba(255, 165, 0, 0.15) !important; }
    .active-row, .active-row * { color: #333 !important; }
    .wrong-label { border-left: 4px solid #ef4444 !important; }
    .correct-label { border-left: 4px solid #22c55e !important; }
    video { border-radius: 12px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.15) !important; background-color: #000; }
    .gemini-panel { border: 1px solid var(--border-color-primary, #e5e7eb); border-radius: 8px; padding: 12px; margin-top: 8px; background: var(--background-fill-secondary, #f9fafb); max-height: 200px; overflow-y: auto; }
    """

    js_onload = """
    function setup_page() {
        setInterval(() => {
            const videos = document.querySelectorAll('video');
            videos.forEach(v => { if(v.playbackRate !== 1.5) v.playbackRate = 1.5; });
        }, 500);

        let activeRowIndex = 0;

        function updateActiveRow() {
            const rows = document.querySelectorAll('.label-row');
            let visibleRows = Array.from(rows).filter(r => !r.classList.contains('hide'));
            visibleRows.forEach((r, idx) => {
                if(idx === activeRowIndex) r.classList.add('active-row');
                else r.classList.remove('active-row');
            });
            return visibleRows;
        }

        let lastKnownLabels = "";
        const observer = new MutationObserver(() => {
            const visibleRows = Array.from(document.querySelectorAll('.label-row')).filter(r => !r.classList.contains('hide'));
            if(visibleRows.length === 0) return;
            const currentLabels = visibleRows.map(r => r.innerText).join('|');
            if(currentLabels !== lastKnownLabels) { activeRowIndex = 0; lastKnownLabels = currentLabels; }
            updateActiveRow();
        });
        observer.observe(document.body, { childList: true, subtree: true, characterData: true });

        document.addEventListener('keydown', function(e) {
            if(e.target.tagName.toLowerCase() === 'input' && e.target.type === 'text') return;
            if(e.key === '?') { const b = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Shortcuts')); if(b) b.click(); return; }
            if(e.key === ' ') { e.preventDefault(); const v = document.querySelector('video'); if(v) { if(v.paused) v.play(); else v.pause(); } return; }
            if(e.key === 'ArrowRight' || e.key === 'Enter') { const b = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Next') || b.innerText === 'Submit'); if(b) b.click(); return; }
            if(e.key === 'ArrowLeft') { const b = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Prev')); if(b) b.click(); return; }
            const visibleRows = updateActiveRow();
            if(visibleRows.length === 0 || activeRowIndex >= visibleRows.length) return;
            const key = e.key.toLowerCase();
            if(['a', 'v', 'b', 'n'].includes(key)) {
                const targetRow = visibleRows[activeRowIndex];
                const cbs = targetRow.querySelectorAll('input[type="checkbox"]');
                if(cbs.length >= 2) {
                    const ac = cbs[0], vc = cbs[1];
                    if(ac.checked) ac.click(); if(vc.checked) vc.click();
                    if(key === 'a') { if(!ac.checked) ac.click(); }
                    else if(key === 'v') { if(!vc.checked) vc.click(); }
                    else if(key === 'b') { if(!ac.checked) ac.click(); if(!vc.checked) vc.click(); }
                }
                activeRowIndex = Math.min(activeRowIndex + 1, visibleRows.length - 1);
                updateActiveRow();
            }
        });
    }
    """

    with gr.Blocks(
        title="VGGSounder Label Review",
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate"),
    ) as demo:
        demo.load(None, None, None, js=js_onload)
        current_idx = gr.State(0)

        with gr.Row():
            with gr.Column(scale=10, min_width=300):
                with gr.Row(elem_classes="no-margin"):
                    btn_shortcuts = gr.Button(
                        "Shortcuts", size="sm", elem_classes="top-bar-btn"
                    )

        shortcuts_info = gr.Markdown(
            "**Keyboard Shortcuts:**\n"
            "- **Mark Audible:** `A`\n"
            "- **Mark Visible:** `V`\n"
            "- **Mark Both:** `B`\n"
            "- **Mark Neither:** `N`\n"
            "- **Play / Pause Video:** `Space`\n"
            "- **Next Video / Save:** `Enter` or `Right Arrow`\n"
            "- **Previous Video:** `Left Arrow`\n"
            "- **Toggle Shortcuts:** `?`",
            visible=False,
        )

        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=1, elem_classes="center-nav"):
                with gr.Row():
                    btn_prev = gr.Button("← Prev", elem_classes="nav-btn", size="sm")
                    jump_input = gr.Number(
                        value=1,
                        minimum=1,
                        maximum=max(len(backend.tasks), 1),
                        label="",
                        container=False,
                        scale=0,
                        min_width=80,
                    )
                    btn_jump = gr.Button("Go", elem_classes="nav-btn", size="sm")
                    btn_next = gr.Button("Next →", elem_classes="nav-btn", size="sm")
                    btn_skip = gr.Button(
                        "Skip to new", elem_classes="top-bar-btn", size="sm"
                    )
            with gr.Column(scale=1):
                pass

        total = len(backend.tasks)
        progress_text = gr.Markdown(f"**1** / {total} &nbsp; | &nbsp; Session: 0 new")

        with gr.Row():
            with gr.Column(scale=6):
                video_player = gr.Video(
                    label="",
                    autoplay=True,
                    height=450,
                    interactive=False,
                    show_label=False,
                )
                gemini_panel = gr.Markdown("", elem_classes="gemini-panel")

            with gr.Column(scale=5):
                gr.Markdown("**Can you hear and/or see the following?**")
                with gr.Row(elem_classes="header-row"):
                    with gr.Column(scale=5, min_width=0):
                        gr.Markdown(" ")
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("**Audible**")
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("**Visible**")

                label_rows = []
                for i in range(MAX_LABELS):
                    with gr.Row(elem_classes="label-row", visible=False) as r:
                        with gr.Column(scale=5, min_width=0):
                            lbl_txt = gr.Markdown(
                                f"label {i}", elem_classes="no-margin"
                            )
                        with gr.Column(scale=1, min_width=0):
                            aud = gr.Checkbox(
                                label="\u200b",
                                container=False,
                                show_label=False,
                                elem_classes="hide-checkbox-text",
                            )
                        with gr.Column(scale=1, min_width=0):
                            vis = gr.Checkbox(
                                label="\u200b",
                                container=False,
                                show_label=False,
                                elem_classes="hide-checkbox-text",
                            )
                        label_rows.append((r, lbl_txt, aud, vis))

        gr.HTML("<hr>")
        with gr.Row():
            with gr.Column(scale=9):
                pass
            with gr.Column(scale=1):
                btn_submit = gr.Button("Submit", elem_classes="submit-btn")

        # --- outputs list ---
        all_outputs = [video_player, gemini_panel]
        for r, lbl, a, v in label_rows:
            all_outputs.extend([r, lbl, a, v])

        def _build_gemini_md(task: dict) -> str:
            info = task.get("gemini_info", {})
            if not info:
                return "*No Gemini analysis available.*"
            lines = ["**Gemini Analysis:**\n"]
            for lbl, detail in info.items():
                lc = detail.get("label_correct", "")
                mc = detail.get("modality_correct", "")
                sm = detail.get("suggested_modality", "")
                reason = detail.get("reason", "")
                icon = "✅" if str(lc).lower() == "true" else "❌"
                mod_icon = "✅" if str(mc).lower() == "true" else "⚠️"
                lines.append(f"- {icon} **{lbl}** — modality {mod_icon} → `{sm}`")
                if reason:
                    lines.append(f"  - *{reason}*")
            return "\n".join(lines)

        def load_task_ui(idx):
            task = backend.get_task(idx)
            if not task:
                res = [None, ""]
                for _ in range(MAX_LABELS):
                    res.extend([gr.update(visible=False), "", False, False])
                return res

            vid_path = backend.get_video_path(task["video_id"])
            candidates = task["candidate_labels"]
            saved = backend.saved_states.get(task["video_id"], {})
            gemini_info = task.get("gemini_info", {})

            res = [vid_path, _build_gemini_md(task)]

            for i in range(MAX_LABELS):
                if i < len(candidates):
                    lbl = candidates[i]
                    mod = saved.get(lbl, "")
                    is_a = "A" in mod if mod else False
                    is_v = "V" in mod if mod else False

                    gi = gemini_info.get(lbl, {})
                    lc = (
                        str(gi.get("label_correct", "")).lower() == "true"
                        if gi
                        else True
                    )
                    prefix = "" if lc else "~~"
                    suffix = "" if lc else "~~ ❌"

                    res.extend(
                        [
                            gr.update(visible=True),
                            f"{prefix}{lbl}{suffix}",
                            is_a,
                            is_v,
                        ]
                    )
                else:
                    res.extend([gr.update(visible=False), "", False, False])
            return res

        def make_progress(idx):
            new_count = len(backend.session_annotated - backend.initial_annotated)
            return f"**{idx + 1}** / {total} &nbsp; | &nbsp; Session: {new_count} new"

        def save_current(idx, *args):
            task = backend.get_task(idx)
            if not task:
                return
            results = {}
            for i in range(MAX_LABELS):
                lbl_val = args[i * 3]
                a_val = args[i * 3 + 1]
                v_val = args[i * 3 + 2]
                if lbl_val:
                    clean_lbl = lbl_val.replace("~~", "").replace(" ❌", "").strip()
                    if not clean_lbl:
                        continue
                    mod = ""
                    if a_val and v_val:
                        mod = "AV"
                    elif a_val:
                        mod = "A"
                    elif v_val:
                        mod = "V"
                    results[clean_lbl] = mod
            backend.save_verdict(task["video_id"], results, task["meta"])

        def on_prev(idx, *args):
            save_current(idx, *args)
            new_idx = max(0, idx - 1)
            return new_idx, new_idx + 1, make_progress(new_idx)

        def on_next(idx, *args):
            save_current(idx, *args)
            new_idx = idx + 1
            return new_idx, new_idx + 1, make_progress(new_idx)

        def on_submit(idx, *args):
            save_current(idx, *args)
            new_idx = idx + 1
            return new_idx, new_idx + 1, make_progress(new_idx)

        def on_jump(idx, jump_val, *args):
            save_current(idx, *args)
            new_idx = max(0, min(int(jump_val) - 1, len(backend.tasks) - 1))
            return new_idx, new_idx + 1, make_progress(new_idx)

        def on_skip(idx, *args):
            save_current(idx, *args)
            for i in range(len(backend.tasks)):
                vid = backend.tasks[i]["video_id"]
                if vid not in backend.saved_states:
                    return i, i + 1, make_progress(i)
            return idx, idx + 1, make_progress(idx)

        # --- wiring ---
        demo.load(fn=load_task_ui, inputs=[current_idx], outputs=all_outputs)

        clean_inputs = [current_idx]
        for _r, lbl, a, v in label_rows:
            clean_inputs.extend([lbl, a, v])

        jump_inputs = [current_idx, jump_input] + clean_inputs[1:]
        nav_outputs = [current_idx, jump_input, progress_text]

        btn_submit.click(fn=on_submit, inputs=clean_inputs, outputs=nav_outputs).then(
            fn=load_task_ui, inputs=[current_idx], outputs=all_outputs
        )
        btn_prev.click(fn=on_prev, inputs=clean_inputs, outputs=nav_outputs).then(
            fn=load_task_ui, inputs=[current_idx], outputs=all_outputs
        )
        btn_next.click(fn=on_next, inputs=clean_inputs, outputs=nav_outputs).then(
            fn=load_task_ui, inputs=[current_idx], outputs=all_outputs
        )
        btn_jump.click(fn=on_jump, inputs=jump_inputs, outputs=nav_outputs).then(
            fn=load_task_ui, inputs=[current_idx], outputs=all_outputs
        )
        btn_skip.click(fn=on_skip, inputs=clean_inputs, outputs=nav_outputs).then(
            fn=load_task_ui, inputs=[current_idx], outputs=all_outputs
        )

        shortcuts_state = gr.State(False)

        def toggle_shortcuts(visible):
            return gr.update(visible=not visible), not visible

        btn_shortcuts.click(
            fn=toggle_shortcuts,
            inputs=[shortcuts_state],
            outputs=[shortcuts_info, shortcuts_state],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7861)
