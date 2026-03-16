import gradio as gr
import pandas as pd
import json
import os
import shutil
import sys
import tempfile
import traceback
import vggsounder
from huggingface_hub import hf_hub_download
from collections import defaultdict

class LabelingBackend:
    def __init__(self, proposals_csv, dataset_id, output_path):
        self.output_path = output_path
        self.dataset_id = dataset_id
        self.tasks = []
        self.vgg_data = vggsounder.VGGSounder()

        try:
            self.proposals_df = pd.read_csv(proposals_csv, keep_default_na=False)
        except Exception as e:
            print("Error loading CSV:", e)
            self.proposals_df = pd.DataFrame()

        if not self.proposals_df.empty:
            self.video_ids = self.proposals_df['video_id'].unique()
            self.proposals_by_vid = self.proposals_df.groupby('video_id', sort=False)
        else:
            self.video_ids = []
            self.proposals_by_vid = None

        self.saved_states = {} # { video_id: { label: modality_str } }
        self._load_saved_states()
        self.initial_annotated = set(self.saved_states.keys())
        self.session_annotated = set()
        self._prepare_tasks()

    def _load_saved_states(self):
        if not os.path.exists(self.output_path):
            return
        try:
            df = pd.read_csv(self.output_path, keep_default_na=False)
        except Exception as e:
            print("Error loading saved states:", e)
            return
        # Since rows are appended, later entries override earlier ones.
        # Group by video_id and take the last set of entries for each.
        for vid_id, group in df.groupby('video_id', sort=False):
            # Find the last "batch" — the last contiguous block of rows for this video_id.
            # Since save_verdict appends all labels at once, the last occurrence of
            # this video_id's block is the most recent save.
            results = {}
            for row in group.itertuples():
                if row.label:
                    results[row.label] = row.modality
            self.saved_states[vid_id] = results

    @staticmethod
    def _normalize_label(label):
        """Normalize a label for dedup: lowercase, collapse whitespace, strip punctuation."""
        import re
        label = label.lower().strip()
        label = re.sub(r'[,/]+', ' ', label)
        label = re.sub(r'\s+', ' ', label)
        return label

    def _add_label(self, seen, candidate_labels, label):
        """Add label only if its normalized form hasn't been seen yet."""
        norm = self._normalize_label(label)
        if norm not in seen:
            seen[norm] = label
            candidate_labels.append(label)

    def _prepare_tasks(self):
        for vid_id in self.video_ids:
            seen = {}  # normalized -> original label
            candidate_labels = []

            # Original labels (prefer these over suggested ones)
            vid_info = self.vgg_data[vid_id]
            for lbl in vid_info.labels:
                self._add_label(seen, candidate_labels, lbl)

            # Proposed labels
            proposals = self.proposals_by_vid.get_group(vid_id)
            for row in proposals.itertuples():
                if row.label:
                    self._add_label(seen, candidate_labels, row.label)
                if hasattr(row, 'suggested_labels') and row.suggested_labels:
                    suggestions = json.loads(row.suggested_labels)
                    for s in suggestions:
                        self._add_label(seen, candidate_labels, s['label'])

            self.tasks.append({
                'video_id': vid_id,
                'candidate_labels': sorted(candidate_labels)
            })

    def get_task(self, index):
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        return None

    def save_verdict(self, video_id, results_dict):
        # Update memory
        self.saved_states[video_id] = results_dict
        # Only count as annotated if at least one label has a modality set
        if any(mod for mod in results_dict.values()):
            self.session_annotated.add(video_id)

        rows = []
        for lbl, mod in results_dict.items():
            if mod:  # Only save if they selected something
                rows.append({
                    "video_id": video_id, 
                    "label": lbl, 
                    "modality": mod
                })
        if not rows:
            # If nothing checked, just save a record saying video was reviewed
            rows.append({
                "video_id": video_id, 
                "label": "", 
                "modality": ""
            })

        # To avoid duplicating entries in the CSV endlessly on Prev/Next,
        # we can rewrite the CSV, or just append and let the latest row win in analysis. 
        # For simplicity and robustness against crashes, we just append here.
        df = pd.DataFrame(rows)
        header = not os.path.exists(self.output_path)
        df.to_csv(self.output_path, mode='a', index=False, header=header)

    def get_video_path(self, video_id):
        try:
            path = hf_hub_download(
                repo_id=self.dataset_id,
                filename=f'video/{video_id}/video.mp4',
                repo_type='dataset'
            )
            temp_dir = tempfile.gettempdir()
            target = os.path.join(temp_dir, f"vgg_{video_id}.mp4")
            if not os.path.exists(target):
                shutil.copy(path, target)
            return target
        except Exception as e:
            print(f"Error downloading video {video_id}: {e}")
            return None

def create_app():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend = LabelingBackend(
        os.path.join(script_dir, 'cleaned_wrong_labels.csv'),
        '11hu83/vggsound',
        os.path.join(script_dir, 'output.csv')
    )
    MAX_LABELS = 20

    custom_css = """
    .orange-btn {
        background-color: #FFA500 !important;
        color: black !important;
        border: none !important;
        font-weight: bold !important;
    }
    .orange-btn:hover {
        background-color: #FF8C00 !important;
    }
    .submit-btn {
        background-color: #FFA500 !important;
        color: black !important;
        font-weight: bold !important;
        border: none !important;
        width: 100px !important;
        margin-left: auto !important;
        display: block !important;
    }
    .nav-btn {
        background-color: #FFA500 !important;
        color: black !important;
        font-weight: bold !important;
        border: none !important;
    }
    .label-row {
        margin: 6px 0 !important;
        padding: 8px 12px !important;
        align-items: center;
        border: 1px solid var(--border-color-primary, #e5e7eb) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
        background-color: var(--background-fill-secondary, transparent);
    }
    .header-row {
        margin-bottom: 10px !important;
        align-items: center;
        padding: 0 12px;
    }
    .center-nav {
        display: flex;
        justify-content: center;
        gap: 15px;
    }
    .top-bar-btn {
        background: transparent !important;
        border: 1px solid var(--border-color-primary, #ccc) !important;
        border-radius: 6px !important;
        color: var(--body-text-color, #333) !important;
    }
    .no-margin { margin: 0 !important; padding: 0 !important; }
    /* Hide the text associated with the label component inside the checkbox if it tries to grab accessibility names */
    .hide-checkbox-text span.ml-2 {
        display: none !important;
    }
    .active-row {
        background-color: #fff3e0 !important;
        border-radius: 8px !important;
        border-color: #ffcc80 !important;
        box-shadow: 0 2px 6px rgba(255, 165, 0, 0.15) !important;
    }
    .active-row, .active-row * {
        color: #333 !important;
    }
    /* Style the video element for a premium look */
    video {
        border-radius: 12px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15) !important;
        background-color: #000;
    }
    """

    # We use a javascript snippet to force 1.5x playback speed and add global keyboard shortcuts.
    js_onload = """
    function setup_page() {
        setInterval(() => {
            const videos = document.querySelectorAll('video');
            videos.forEach(v => {
                if(v.playbackRate !== 1.5) v.playbackRate = 1.5;
            });
        }, 500);

        let activeRowIndex = 0;

        function updateActiveRow() {
            const rows = document.querySelectorAll('.label-row');
            let visibleRows = Array.from(rows).filter(r => !r.classList.contains('hide'));
            
            visibleRows.forEach((r, idx) => {
                if(idx === activeRowIndex) {
                    r.classList.add('active-row');
                } else {
                    r.classList.remove('active-row');
                }
            });
            return visibleRows;
        }

        // To avoid resetting active row when the video finishes, 
        // we'll only reset it if the text content of the labels actually changes (i.e. we loaded a new task)
        let lastKnownLabels = "";
        
        const observer = new MutationObserver((mutations) => {
            const visibleRows = Array.from(document.querySelectorAll('.label-row')).filter(r => !r.classList.contains('hide'));
            if(visibleRows.length === 0) return;
            
            // Generate a simple fingerprint of the currently visible labels
            const currentLabels = visibleRows.map(r => r.innerText).join('|');
            
            if(currentLabels !== lastKnownLabels) {
                // The actual tasks changed (clicked Next/Prev)
                activeRowIndex = 0;
                lastKnownLabels = currentLabels;
            }
            // re-apply active row styling inside the same tick
            updateActiveRow();
        });
        observer.observe(document.body, { childList: true, subtree: true, characterData: true });

        document.addEventListener('keydown', function(e) {
            if(e.target.tagName.toLowerCase() === 'input' && e.target.type === 'text') return;
            
            // Toggle Shortcuts menu
            if(e.key === '?') {
                const buttons = Array.from(document.querySelectorAll('button'));
                const shortcutBtn = buttons.find(b => b.innerText.includes('Shortcuts'));
                if(shortcutBtn) shortcutBtn.click();
                return;
            }

            // Spacebar = toggle play/pause on the video
            if(e.key === ' ') {
                e.preventDefault(); // prevent scrolling down the page
                const videos = document.querySelectorAll('video');
                if(videos.length > 0) {
                    const v = videos[0];
                    if(v.paused) v.play();
                    else v.pause();
                }
                return;
            }

            // Video Navigation navigation
            if(e.key === 'ArrowRight' || e.key === 'Enter') {
                const buttons = Array.from(document.querySelectorAll('button'));
                const nextBtn = buttons.find(b => b.innerText.includes('Next') || b.innerText === 'Submit');
                if(nextBtn) nextBtn.click();
                return;
            } else if(e.key === 'ArrowLeft') {
                const buttons = Array.from(document.querySelectorAll('button'));
                const prevBtn = buttons.find(b => b.innerText.includes('Prev'));
                if(prevBtn) prevBtn.click();
                return;
            }

            // A / V / B Labeling Shortcuts
            const visibleRows = updateActiveRow();
            if(visibleRows.length === 0 || activeRowIndex >= visibleRows.length) return;

            const key = e.key.toLowerCase();
            if(['a', 'v', 'b', 'n'].includes(key)) {
                const targetRow = visibleRows[activeRowIndex];
                const checkboxes = targetRow.querySelectorAll('input[type="checkbox"]');
                if(checkboxes.length >= 2) {
                    const audCb = checkboxes[0];
                    const visCb = checkboxes[1];

                    // Reset first
                    if(audCb.checked) audCb.click();
                    if(visCb.checked) visCb.click();

                    if(key === 'a') {
                        if(!audCb.checked) audCb.click();
                    } else if(key === 'v') {
                        if(!visCb.checked) visCb.click();
                    } else if(key === 'b') {
                        if(!audCb.checked) audCb.click();
                        if(!visCb.checked) visCb.click();
                    }
                    // 'n' does nothing else, leaving both unchecked (which is "Neither")
                }
                
                // Advance to next row automatically
                activeRowIndex++;
                if(activeRowIndex >= visibleRows.length) {
                    // Automatically next video if they finished the list? 
                    // Let's leave it on the last item so they can still see it.
                    // Or they just press right arrow to submit.
                    activeRowIndex = visibleRows.length - 1; 
                }
                updateActiveRow();
            }
        });
    }
    """

    with gr.Blocks(title="VGGSounder Labeler", css=custom_css, theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate")) as demo:
        demo.load(None, None, None, js=js_onload)
        current_idx = gr.State(0)

        # Top Bar
        with gr.Row():
            with gr.Column(scale=10, min_width=300):
                with gr.Row(elem_classes="no-margin"):
                    btn_shortcuts = gr.Button("Shortcuts", size="sm", elem_classes="top-bar-btn")

        shortcuts_info = gr.Markdown(
            "**Keyboard Shortcuts:**\n"
            "- **Mark Audible:** `A`\n"
            "- **Mark Visible:** `V`\n"
            "- **Mark Both:** `B`\n"
            "- **Mark Neither:** `N`\n"
            "- **Play / Pause Video:** `Space`\n"
            "- **Next Video / Save:** `Enter` or `Right Arrow`\n"
            "- **Previous Video:** `Left Arrow`\n"
            "- **Toggle Shortcuts:** `?`\n"
            "*(Requires clicking outside input fields to focus)*",
            visible=False
        )

        # Nav Bar (Center)
        with gr.Row():
            with gr.Column(scale=1): pass
            with gr.Column(scale=1, elem_classes="center-nav"):
                with gr.Row():
                    btn_prev = gr.Button("← Prev", elem_classes="nav-btn", size="sm")
                    jump_input = gr.Number(value=1, minimum=1, maximum=max(len(backend.tasks), 1), label="", container=False, scale=0, min_width=80)
                    btn_jump = gr.Button("Go", elem_classes="nav-btn", size="sm")
                    btn_next = gr.Button("Next →", elem_classes="nav-btn", size="sm")
                    btn_skip_to_new = gr.Button("Skip to new", elem_classes="top-bar-btn", size="sm")
            with gr.Column(scale=1): pass

        # Progress
        total = len(backend.tasks)
        progress_text = gr.Markdown(f"**1** / {total} &nbsp; | &nbsp; Session: 0 new")

        # Main Layout
        with gr.Row():
            # Left: Video Player
            with gr.Column(scale=6):
                video_player = gr.Video(label="", autoplay=True, height=450, interactive=False, show_label=False)

            # Right: Labeling
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
                            lbl_txt = gr.Markdown(f"label {i}", elem_classes="no-margin")
                        with gr.Column(scale=1, min_width=0):
                            aud = gr.Checkbox(label="\u200b", container=False, show_label=False, elem_classes="hide-checkbox-text")
                        with gr.Column(scale=1, min_width=0):
                            vis = gr.Checkbox(label="\u200b", container=False, show_label=False, elem_classes="hide-checkbox-text")
                        # We store the state mapping inside the component identity conceptually
                        label_rows.append((r, lbl_txt, aud, vis))


        # Footer
        gr.HTML("<hr>")
        with gr.Row():
            with gr.Column(scale=9): pass
            with gr.Column(scale=1):
                btn_submit = gr.Button("Submit", elem_classes="submit-btn")

        # --- Logic ---
        all_outputs = [video_player]
        for r, lbl, a, v in label_rows:
            all_outputs.extend([r, lbl, a, v])

        def load_task_ui(idx):
            task = backend.get_task(idx)
            if not task:
                # EOF
                res = [None]
                for _ in range(MAX_LABELS):
                    res.extend([gr.update(visible=False), "", False, False])
                return res

            vid_path = backend.get_video_path(task['video_id'])
            candidates = task['candidate_labels']
            saved_state = backend.saved_states.get(task['video_id'], {})

            res = [
                vid_path
            ]
            
            for i in range(MAX_LABELS):
                if i < len(candidates):
                    lbl = candidates[i]
                    mod = saved_state.get(lbl, "")
                    is_a = "A" in mod if mod else False
                    is_v = "V" in mod if mod else False

                    # Show row
                    res.extend([
                        gr.update(visible=True),   # Row
                        lbl,                       # Markdown text
                        is_a,                      # Aud
                        is_v                       # Vis
                    ])
                else:
                    # Hide row
                    res.extend([
                        gr.update(visible=False),
                        "",
                        False,
                        False
                    ])
            return res



        def make_progress_text(idx):
            total = len(backend.tasks)
            new_in_session = len(backend.session_annotated - backend.initial_annotated)
            return f"**{idx + 1}** / {total} &nbsp; | &nbsp; Session: {new_in_session} new"

        def process_state_and_save(idx, *args):
            task = backend.get_task(idx)
            if not task:
                return
            
            results = {}
            for i in range(MAX_LABELS):
                lbl_val = args[i*3]
                a_val = args[i*3 + 1]
                v_val = args[i*3 + 2]

                if lbl_val:
                    mod = ""
                    if a_val and v_val: mod = "AV"
                    elif a_val: mod = "A"
                    elif v_val: mod = "V"
                    results[lbl_val] = mod
            
            backend.save_verdict(task['video_id'], results)

        def on_prev(idx, *args):
            process_state_and_save(idx, *args)
            new_idx = max(0, idx - 1)
            return new_idx, new_idx + 1, make_progress_text(new_idx)

        def on_next(idx, *args):
            process_state_and_save(idx, *args)
            new_idx = idx + 1
            return new_idx, new_idx + 1, make_progress_text(new_idx)

        def on_jump(idx, jump_val, *args):
            process_state_and_save(idx, *args)
            new_idx = max(0, min(int(jump_val) - 1, len(backend.tasks) - 1))
            return new_idx, new_idx + 1, make_progress_text(new_idx)

        def on_skip_to_new(idx, *args):
            process_state_and_save(idx, *args)
            for i in range(len(backend.tasks)):
                vid_id = backend.tasks[i]['video_id']
                if vid_id not in backend.saved_states:
                    return i, i + 1, make_progress_text(i)
            # All annotated — stay where we are
            return idx, idx + 1, make_progress_text(idx)

        # Wiring
        demo.load(fn=load_task_ui, inputs=[current_idx], outputs=all_outputs)
        
        # Let's cleanly separate inputs to avoid Row issues:
        clean_inputs = [current_idx]
        for r, lbl, a, v in label_rows:
            clean_inputs.extend([lbl, a, v])
            
        def on_submit_clean(idx, *args):
            process_state_and_save(idx, *args)
            new_idx = idx + 1
            return new_idx, new_idx + 1, make_progress_text(new_idx)

        jump_inputs = [current_idx, jump_input] + clean_inputs[1:]

        nav_outputs = [current_idx, jump_input, progress_text]

        btn_submit.click(fn=on_submit_clean, inputs=clean_inputs, outputs=nav_outputs).then(
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

        btn_skip_to_new.click(fn=on_skip_to_new, inputs=clean_inputs, outputs=nav_outputs).then(
            fn=load_task_ui, inputs=[current_idx], outputs=all_outputs
        )

        # Shortcuts Toggle
        shortcuts_state = gr.State(False)
        def toggle_shortcuts(visible):
            return gr.update(visible=not visible), not visible
            
        btn_shortcuts.click(fn=toggle_shortcuts, inputs=[shortcuts_state], outputs=[shortcuts_info, shortcuts_state])

    return demo

if __name__ == "__main__":
    app = create_app()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860
    )

