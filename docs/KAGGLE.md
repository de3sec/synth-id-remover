# Running on Kaggle

Notebook: [kaggle_notebook.ipynb](kaggle_notebook.ipynb)

## Setup
1. Create a Kaggle notebook and upload `kaggle_notebook.ipynb` (File → Import).
2. Settings: **Accelerator: GPU**, **Internet: On**.
3. (Recommended) Add a `HF_TOKEN` in Kaggle Secrets for faster Hugging Face downloads.
4. Run the cells top to bottom. First run downloads ~21 GB of models.

## Why scratch disk
`/kaggle/working` is capped at ~20 GB and is saved as notebook output, so models would not fit.
The notebook probes `/kaggle/temp`, `/kaggle/tmp` and `/tmp` (sources disagree on the exact name; Kaggle's
official docs were not verified) and uses the one with the most free space. It requires >= 30 GB free.
Scratch is wiped when the session ends, so models re-download every session.

The app reads these environment variables (set by the notebook):

| Variable | Effect |
|---|---|
| `SYNTH_DATA_DIR` | Where ComfyUI and the models are stored (default: repo directory) |
| `COMFYUI_REF` | ComfyUI git tag to check out (default `v0.4.0`) |
| `SYNTH_MAX_POLL` | Seconds to wait for ComfyUI to finish, incl. model loading (default `120`; the notebook sets `1800`) |
| `SKIP_PERF_OPTS=1` | Skip building `flash-attn`/`sageattention` (very slow on Kaggle) |
| `GRADIO_SHARE=1` | Launch Gradio with a public share link |

## Why ComfyUI v0.4.0 + ComfyUI-dype v2.3.0
- The workflow runs the `DyPE_FLUX` node on a Z-Image (`NextDiT`) model. `ComfyUI-dype` only supports Z-Image from
  v2.2.0 (fixed in v2.3.0, commit `3b96d85`). Older versions (e.g. v2.1.0) look for `pe_embedder` and fail with
  `'NextDiT' object has no attribute 'pe_embedder'` / `not a compatible FLUX/Qwen model structure`.
  (`NextDiT` has always used `rope_embedder`; it was never renamed.)
- dype v2.3.0 overrides `NextDiT.patchify_and_embed` with the signature used by ComfyUI v0.4.0; it is untested on
  newer ComfyUI, so the two are pinned together.
- `v0.3.77` is too old: it lacks the Z-Image Fun ControlNet loader (`UnboundLocalError: ... 'model'` in `load_model_patch`).

## Known issues
- Kaggle T4/P100 GPUs have no native bf16; processing is slow and may exceed the default 120 s limit in `app.py` (set `SYNTH_MAX_POLL`).
- If a workflow fails, ComfyUI's error is raised immediately instead of waiting for the timeout.
- Switching ComfyUI versions with local patches needs `git checkout -f` (the notebook does this; models are untracked and kept).
