# SynthID-Bypass v2 on Kaggle

Notebook: [kaggle_notebook_v2.ipynb](kaggle_notebook_v2.ipynb) · runner: [../v2/synthid_v2.py](../v2/synthid_v2.py)

Independent of v1 (`app.py`, `kaggle_notebook.ipynb`): own ComfyUI, own custom nodes, own models, own data dir (`synth_v2`).

## Setup
1. Kaggle: File → Import the notebook. Settings: **Accelerator: GPU**, **Internet: On**.
2. Push this repo to GitHub first: the notebook clones `de3sec/synth-id-remover` (cell 2) to get `v2/synthid_v2.py`.
3. Run the cells in order. Cell 3 downloads ~30 GB (needs ~38 GB free scratch). Cell 4 is a dry run that starts ComfyUI,
   checks every custom node loaded and converts the workflow, without needing an image.

## What it runs
The workflow `Synthid-Bypass-v2.0.json` is taken from [cebeuq/Synthid-Bypass](https://github.com/cebeuq/Synthid-Bypass)
(pinned to `c45d3bd`, cloned at setup, not copied into this repo: that repo has no license file).
It is a UI-format workflow, so `ui_to_api()` converts it using the running server's `/object_info`
(handles widget order, `control_after_generate`, bypassed nodes, and the rgthree Power Lora widget).

Only the nodes that lead to `SaveImage` run. `controlnet_aux`/MediaPipe and Inpaint-CropAndStitch are in the graph but
their outputs feed nothing that reaches `SaveImage`, so they are not installed.

## Pinned versions (and why)
Everything is pinned to what the workflow was authored against (node metadata in the JSON + commit dates), not "latest":

| Component | Pin |
|---|---|
| ComfyUI | `v0.9.2` (2026-01-15; frontend 1.36.14, workflow saved with 1.36.13). `comfy-kitchen==0.2.6` |
| ComfyUI-GGUF | `795e451` (the workflow's version) |
| ComfyUI-Impact-Pack | `6a517eb` (the workflow's newest Impact version; the bundled node calls `impact.core.enhance_detail` internals) |
| ComfyUI-Impact-Subpack | `50c7b71` (= 1.3.5, the workflow's version) |
| rgthree-comfy / ComfyUI-KJNodes | `8ff50e4` / `eaa9dbf` (the workflow's versions) |
| masquerade-nodes-comfyui | `432cb4d` (latest; unchanged since 2024) |
| RES4LYF | `0dc91c0` (last commit before the v2 release; provides `res_2s` + `bong_tangent`) |

Checked in the sources:
- The bundled node's `DifferentialDiffusion().execute(model)[0]` works on v0.9.2 (V3 node, `NodeOutput.__getitem__` exists).
- The `Qwen2.5-VL` GGUF is loaded with CLIP type `lumina2` in the workflow; in v0.9.2 `load_text_encoder_state_dicts` only
  special-cases `hunyuan_image` for that model, so it gets the Qwen-Image encoder either way.
- `ModelPatchLoader` reads `models/model_patches/`. The workflow's own Note says `diffusion_model_patches/`, which ComfyUI does not use.
- The v2 README's `ae.safetensors` link (`Comfy-Org/Z-Image-ComfyUI`) returned HTTP 401; the same file is served by `Comfy-Org/z_image_turbo`.

## Not verified
No GPU was available when this was written, so the whole pipeline has **not been run end to end**. What was checked:
the workflow conversion against the pinned node sources (widget order for every executed node) and the compatibility
points above. Unknowns: whether `pip install` of the node requirements resolves cleanly on Kaggle's Python 3.13 / torch image,
and speed/memory on a T4 (the Qwen-Image Q4 GGUF is 13.3 GB on a 15 GB card, so ComfyUI offloads part of it).
If something fails, `process()` raises with the failing node and the ComfyUI log tail (`synth_v2/comfyui.log`).

## Tuning
- `adaptive_level` (1-10, default 6 as saved in the workflow): the workflow's Note says use 7-8 for higher-resolution inputs.
- `max_megapixels`: optional downscale of big inputs first (default: keep size).
