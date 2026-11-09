"""SynthID-Bypass v2 runner: headless ComfyUI driving the v2 workflow from cebeuq/Synthid-Bypass.

Everything (ComfyUI, custom nodes, the workflow, ~30 GB of models) lives under one data directory
(``SYNTH_V2_DIR``, default ``./synth_v2_data``) so it can sit on Kaggle's scratch disk.

Usage (see docs/kaggle_notebook_v2.ipynb):
    import synthid_v2 as v2
    v2.setup()                       # clone pinned repos, pip install, download models
    out = v2.process(PIL_image)      # starts ComfyUI on first use, returns a PIL image
"""
import atexit
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid

# Must be set before huggingface_hub is imported; only enable if hf_transfer is installed.
if importlib.util.find_spec("hf_transfer") is not None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import requests
from PIL import Image

# --------------------------------------------------------------------------------------
# Pins. These are the versions the v2 workflow was authored against (its node metadata and
# the dates of the repos' commits), not "latest": ComfyUI, ComfyUI-GGUF and the Impact Pack
# change internal APIs often and the bundled Synthid node calls Impact Pack internals.
# --------------------------------------------------------------------------------------
COMFYUI_URL = "https://github.com/comfyanonymous/ComfyUI"
# v0.9.2 (2026-01-15) ships frontend 1.36.14; the workflow was saved with frontend 1.36.13.
COMFYUI_REF = os.environ.get("COMFYUI_REF", "v0.9.2")
COMFY_KITCHEN_PIN = "comfy-kitchen==0.2.6"  # newest release when v0.9.2 shipped (it requires >=0.2.6)

V2_REPO_URL = "https://github.com/cebeuq/Synthid-Bypass"
V2_REPO_COMMIT = "c45d3bddc33d71ef925521f65e8ca9439c8a7efa"
V2_WORKFLOW_FILE = "Synthid-Bypass-v2.0.json"
V2_NODEPACK_DIR = "custom_nodes/Comfyui-SynthidBypass"  # inside the v2 repo

# (folder name, git url, commit, requirements file or None). Only packs that the executed part of
# the workflow needs. (controlnet_aux/MediaPipe and Inpaint-CropAndStitch appear in the graph but
# their outputs feed nothing that reaches SaveImage, so ComfyUI never runs them.)
CUSTOM_NODES = [
    ("ComfyUI-GGUF", "https://github.com/city96/ComfyUI-GGUF",
     "795e45156ece99afbc3efef911e63fcb46e6a20d", "requirements.txt"),
    ("ComfyUI-Impact-Pack", "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
     "6a517ebe06fea2b74fc41b3bd089c0d7173eeced", "requirements.txt"),
    ("ComfyUI-Impact-Subpack", "https://github.com/ltdrdata/ComfyUI-Impact-Subpack",
     "50c7b71a6a224734cc9b21963c6d1926816a97f1", "requirements.txt"),
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy",
     "8ff50e4521881eca1fe26aec9615fc9362474931", None),
    ("ComfyUI-KJNodes", "https://github.com/kijai/ComfyUI-KJNodes",
     "eaa9dbf239996ad575b5b276b745a2410c9e1ffc", "requirements.txt"),
    ("masquerade-nodes-comfyui", "https://github.com/BadCafeCode/masquerade-nodes-comfyui",
     "432cb4d146a391b387a0cd25ace824328b5b61cf", None),
    ("RES4LYF", "https://github.com/ClownsharkBatwing/RES4LYF",
     "0dc91c00c4c3fb38e7874fcd7a2a327765e8882c", "requirements.txt"),
]

# (hf repo or None, file in repo or direct URL, destination under ComfyUI/models, approx GB)
MODELS = [
    ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q4_K_M.gguf",
     "diffusion_models/qwen-image-2512-Q4_K_M.gguf", 13.3),
    ("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
     "text_encoders/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4.7),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors",
     "vae/qwen_image_vae.safetensors", 0.25),
    ("Comfy-Org/Qwen-Image-DiffSynth-ControlNets",
     "split_files/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors",
     "model_patches/qwen_image_canny_diffsynth_controlnet.safetensors", 2.3),
    ("lightx2v/Qwen-Image-2512-Lightning", "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
     "loras/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors", 1.7),
    ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K_M.gguf",
     "diffusion_models/z_image_turbo-Q4_K_M.gguf", 5.0),
    ("worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF", "Qwen_3_4b-imatrix-IQ4_XS.gguf",
     "text_encoders/Qwen_3_4b-imatrix-IQ4_XS.gguf", 2.3),
    # The v2 README links Comfy-Org/Z-Image-ComfyUI for ae.safetensors, but that URL returns 401
    # (checked 2026-11); the same file is served by Comfy-Org/z_image_turbo.
    ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", "vae/ae.safetensors", 0.34),
    ("Bingsu/adetailer", "face_yolov8n.pt", "ultralytics/bbox/yolov8n-face.pt", 0.01),
    (None, "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
     "sams/sam_vit_b_01ec64.pth", 0.38),
]
MODELS_TOTAL_GB = sum(m[3] for m in MODELS)

# Node ids in Synthid-Bypass-v2.0.json that this module sets / reads.
NODE_LOAD_IMAGE = "11"
NODE_KSAMPLER = "3"
NODE_ADAPTIVE = "595"
NODE_FACE_DETAILER = "592"
NODE_SAVE = "62"
EXPECTED_CLASS = {
    NODE_LOAD_IMAGE: "LoadImage",
    NODE_KSAMPLER: "KSampler",
    NODE_ADAPTIVE: "Synthid-Bypass-AdaptiveDenoise",
    NODE_FACE_DETAILER: "SEGSDetailerModelSwap",
    NODE_SAVE: "SaveImage",
}
DEFAULT_ADAPTIVE_LEVEL = 6  # value stored in the workflow; its Note suggests 7-8 for high resolutions

DATA_DIR = os.path.abspath(os.environ.get("SYNTH_V2_DIR", "synth_v2_data"))
COMFYUI_DIR = os.path.join(DATA_DIR, "ComfyUI")
V2_REPO_DIR = os.path.join(DATA_DIR, "synthid_bypass_v2_repo")
PORT = int(os.environ.get("SYNTH_V2_PORT", "8188"))
LOG_FILE = os.path.join(DATA_DIR, "comfyui.log")


def _run(cmd, **kw):
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def _git_checkout(path, ref):
    """Checkout ``ref``; -f drops local edits. Fetches only if the ref is not present yet."""
    have = subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], cwd=path,
                          capture_output=True).returncode == 0
    if not have:
        _run(["git", "fetch", "-q", "--tags", "origin"], cwd=path)
    _run(["git", "checkout", "-q", "-f", ref], cwd=path)


def _clone_pinned(url, dest, ref):
    if not os.path.isdir(os.path.join(dest, ".git")):
        shutil.rmtree(dest, ignore_errors=True)
        _run(["git", "clone", "-q", url, dest])
    _git_checkout(dest, ref)


def _pip(*args):
    _run([sys.executable, "-m", "pip", "install", "-q", *args])


def _pip_requirements(path):
    """pip install a requirements file with two filters: drop the optional SAM2 git dependency of the
    Impact Pack (not used by this workflow) and prefer headless OpenCV (no GUI libs on Kaggle)."""
    lines = []
    for line in open(path).read().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "facebookresearch/sam2" in s:
            continue
        if s.split("=")[0].split(">")[0].split("<")[0].strip() == "opencv-python":
            s = "opencv-python-headless"
        lines.append(s)
    filtered = os.path.join(DATA_DIR, "_req_" + os.path.basename(os.path.dirname(path)) + ".txt")
    with open(filtered, "w") as f:
        f.write("\n".join(lines) + "\n")
    _pip("-r", filtered)


def _download_http(url, dest):
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)


def _download_models():
    from huggingface_hub import hf_hub_download
    staging = os.path.join(DATA_DIR, "_dl")
    for i, (repo, name, rel_dest, gb) in enumerate(MODELS, 1):
        dest = os.path.join(COMFYUI_DIR, "models", rel_dest)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  [{i}/{len(MODELS)}] {os.path.basename(dest)} (~{gb:.1f} GB)", flush=True)
        if repo is None:
            _download_http(name, dest)
        else:
            got = hf_hub_download(repo_id=repo, filename=name, local_dir=staging)
            shutil.move(got, dest)
    shutil.rmtree(staging, ignore_errors=True)


def setup(skip_models=False):
    """Idempotent: clones/pins ComfyUI + custom nodes, installs pip deps, downloads models."""
    os.makedirs(DATA_DIR, exist_ok=True)
    free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
    need = MODELS_TOTAL_GB + 8  # + ComfyUI, repos, pip cache headroom
    print(f"Data dir: {DATA_DIR} ({free_gb:.0f} GB free, need ~{need:.0f} GB incl. ~{MODELS_TOTAL_GB:.0f} GB models)")
    already = os.path.isdir(COMFYUI_DIR) and os.path.exists(
        os.path.join(COMFYUI_DIR, "models", MODELS[0][2]))
    if not already and free_gb < need:
        raise RuntimeError(f"Only {free_gb:.0f} GB free in {DATA_DIR}; need ~{need:.0f} GB.")

    print("== ComfyUI", COMFYUI_REF)
    _clone_pinned(COMFYUI_URL, COMFYUI_DIR, COMFYUI_REF)
    _pip("-r", os.path.join(COMFYUI_DIR, "requirements.txt"))
    _pip(COMFY_KITCHEN_PIN)

    print("== Synthid-Bypass v2 repo", V2_REPO_COMMIT[:7])
    _clone_pinned(V2_REPO_URL, V2_REPO_DIR, V2_REPO_COMMIT)
    custom_dir = os.path.join(COMFYUI_DIR, "custom_nodes")
    os.makedirs(custom_dir, exist_ok=True)
    pack_dst = os.path.join(custom_dir, "Comfyui-SynthidBypass")
    shutil.rmtree(pack_dst, ignore_errors=True)
    shutil.copytree(os.path.join(V2_REPO_DIR, V2_NODEPACK_DIR), pack_dst)

    print("== Custom nodes")
    for name, url, commit, req in CUSTOM_NODES:
        dest = os.path.join(custom_dir, name)
        _clone_pinned(url, dest, commit)
        if req and os.path.exists(os.path.join(dest, req)):
            _pip_requirements(os.path.join(dest, req))
        print(f"  {name} @ {commit[:7]}")
    _pip("huggingface_hub", "hf_transfer")

    if not skip_models:
        print(f"== Models (~{MODELS_TOTAL_GB:.0f} GB)")
        _download_models()
    print("[OK] setup complete")


# --------------------------------------------------------------------------------------
# UI-format workflow -> API prompt
# --------------------------------------------------------------------------------------
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
BYPASS, MUTE = 4, 2


def _is_widget(spec):
    t = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if opts.get("forceInput"):
        return False
    return isinstance(t, list) or t in _WIDGET_TYPES


def _input_order(info):
    names = []
    order = info.get("input_order")
    if order:
        for group in ("required", "optional"):
            names += [(n, (info["input"].get(group) or {}).get(n)) for n in order.get(group, [])]
    else:
        for group in ("required", "optional"):
            names += list((info["input"].get(group) or {}).items())
    return [(n, s) for n, s in names if s is not None]


def ui_to_api(workflow, object_info, output_types=("SaveImage",)):
    """Convert a ComfyUI UI-format workflow into an API prompt, keeping only what the SaveImage
    node(s) depend on. Widget names come from the running server's /object_info, so it follows
    whatever node versions are installed. Handles bypassed (mode 4) nodes by rewiring."""
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {l[0]: l for l in workflow["links"]}  # id -> [id, from_node, from_slot, to_node, to_slot, type]

    def origin_of(link_id):
        _, src, slot = links[link_id][:3]
        return resolve(src, slot)

    def resolve(node_id, slot):
        """Follow bypassed nodes (and Reroutes) to the real producer of this output."""
        node = nodes[node_id]
        if node["type"] == "Reroute" or node.get("mode") == BYPASS:
            out_type = node["outputs"][slot]["type"] if node["type"] != "Reroute" else None
            for inp in node.get("inputs", []):
                if inp.get("link") is not None and (out_type is None or inp["type"] == out_type):
                    return origin_of(inp["link"])
            raise ValueError(f"Bypassed node {node_id} ({node['type']}) has no input to pass through")
        if node.get("mode") == MUTE:
            raise ValueError(f"Node {node_id} ({node['type']}) is muted but is needed by the graph")
        return node_id, slot

    prompt = {}

    def build(node_id):
        key = str(node_id)
        if key in prompt:
            return
        node = nodes[node_id]
        ctype = node["type"]
        if ctype not in object_info:
            raise ValueError(f"Node {node_id}: class '{ctype}' is not installed in ComfyUI")
        info = object_info[ctype]
        values = list(node.get("widgets_values") or [])
        linked = {i["name"]: i["link"] for i in node.get("inputs", []) if i.get("link") is not None}
        inputs = {}

        if ctype == "Power Lora Loader (rgthree)":
            n = 0
            for v in values:
                if isinstance(v, dict) and "lora" in v:
                    n += 1
                    inputs[f"lora_{n}"] = dict(v)
        else:
            vi = 0
            for name, spec in _input_order(info):
                if _is_widget(spec):
                    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                    if vi < len(values):
                        inputs[name] = values[vi]
                        vi += 1
                        if opts.get("control_after_generate"):
                            vi += 1  # the UI stores "randomize"/"fixed" right after the seed
                # link inputs (and widgets converted to links) are filled below
        for name, link_id in linked.items():
            src, slot = origin_of(link_id)
            build(src)
            inputs[name] = [str(src), slot]
        prompt[key] = {"class_type": ctype, "inputs": inputs, "_meta": {"title": node.get("title") or ctype}}

    roots = [n["id"] for n in workflow["nodes"] if n["type"] in output_types and n.get("mode", 0) == 0]
    if not roots:
        raise ValueError(f"No active {output_types} node in the workflow")
    for r in roots:
        build(r)
    return prompt


# --------------------------------------------------------------------------------------
# ComfyUI server
# --------------------------------------------------------------------------------------
_proc = None
_object_info = None
_base = f"http://127.0.0.1:{PORT}"


def _log_tail(n=40):
    try:
        with open(LOG_FILE, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"


def start_server(timeout=600):
    global _proc, _object_info
    if _proc is not None and _proc.poll() is None:
        return
    _object_info = None
    log = open(LOG_FILE, "w")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    _proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch"],
        cwd=COMFYUI_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
    atexit.register(stop_server)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with code {_proc.returncode} during startup:\n{_log_tail()}")
        try:
            if requests.get(f"{_base}/system_stats", timeout=2).status_code == 200:
                print(f"ComfyUI ready ({time.time() - t0:.0f}s)")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    stop_server()
    raise RuntimeError(f"ComfyUI did not start within {timeout}s:\n{_log_tail()}")


def stop_server():
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def _get_object_info():
    global _object_info
    if _object_info is None:
        r = requests.get(f"{_base}/object_info", timeout=120)
        r.raise_for_status()
        _object_info = r.json()
    return _object_info


def _import_failures():
    try:
        lines = open(LOG_FILE, errors="replace").read().splitlines()
    except OSError:
        return ""
    bad = [l for l in lines if "IMPORT FAILED" in l or "Cannot import" in l or "Traceback" in l]
    return "\n".join(bad[:15])


def load_workflow():
    with open(os.path.join(V2_REPO_DIR, V2_WORKFLOW_FILE)) as f:
        return json.load(f)


def build_prompt(adaptive_level=DEFAULT_ADAPTIVE_LEVEL, seed=None, input_name="input.png"):
    """UI workflow -> API prompt with our per-run settings applied. Needs the server running."""
    wf = load_workflow()
    info = _get_object_info()
    try:
        prompt = ui_to_api(wf, info)
    except ValueError as e:
        raise RuntimeError(f"{e}\nCustom-node import problems from the ComfyUI log:\n{_import_failures()}") from e
    for nid, cls in EXPECTED_CLASS.items():
        if nid not in prompt or prompt[nid]["class_type"] != cls:
            raise RuntimeError(f"Workflow changed: node {nid} is not a {cls}")
    prompt[NODE_LOAD_IMAGE]["inputs"]["image"] = input_name
    prompt[NODE_ADAPTIVE]["inputs"]["adaptive_level"] = int(adaptive_level)
    for nid in (NODE_KSAMPLER, NODE_FACE_DETAILER):  # both are "randomize" in the saved workflow
        prompt[nid]["inputs"]["seed"] = random.randint(0, 2**50) if seed is None else int(seed)
    return prompt


def _explain_node_errors(resp_json):
    parts = []
    err = resp_json.get("error") or {}
    if err:
        parts.append(f"{err.get('message')}: {err.get('details', '')}")
    for nid, ne in (resp_json.get("node_errors") or {}).items():
        for e in ne.get("errors", []):
            parts.append(f"node {nid} ({ne.get('class_type')}): {e.get('message')} - {e.get('details')}")
    return "\n".join(parts) or json.dumps(resp_json)[:2000]


def process(image, adaptive_level=DEFAULT_ADAPTIVE_LEVEL, seed=None, timeout=3600, max_megapixels=None):
    """Run the v2 workflow on a PIL image and return the result as a PIL image.

    adaptive_level: 1-10 (workflow default 6; its Note says 7-8 for higher resolutions).
    max_megapixels: optionally downscale larger inputs first (T4 memory/time); None keeps the size.
    """
    if image is None:
        return None
    image = image.convert("RGB")
    if max_megapixels and image.width * image.height > max_megapixels * 1e6:
        s = (max_megapixels * 1e6 / (image.width * image.height)) ** 0.5
        image = image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)
        print(f"Downscaled input to {image.width}x{image.height}")

    start_server()
    input_name = f"synthid_v2_{uuid.uuid4().hex[:8]}.png"
    input_path = os.path.join(COMFYUI_DIR, "input", input_name)
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    image.save(input_path)
    try:
        prompt = build_prompt(adaptive_level, seed, input_name)
        r = requests.post(f"{_base}/prompt", json={"prompt": prompt, "client_id": uuid.uuid4().hex})
        if r.status_code != 200:
            raise RuntimeError("ComfyUI rejected the workflow:\n" + _explain_node_errors(r.json()))
        pid = r.json()["prompt_id"]
        print(f"Queued {pid[:8]} ({image.width}x{image.height}, adaptive_level={adaptive_level})")

        t0, last_report = time.time(), 0
        while True:
            if _proc is None or _proc.poll() is not None:
                raise RuntimeError("ComfyUI process died (often out of RAM/VRAM). Log tail:\n" + _log_tail())
            h = requests.get(f"{_base}/history/{pid}", timeout=30).json()
            if pid in h:
                entry = h[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    errs = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
                    if errs:
                        e = errs[0]
                        raise RuntimeError(
                            f"ComfyUI failed in node {e.get('node_id')} ({e.get('node_type')}): "
                            f"{e.get('exception_type')}: {e.get('exception_message')}\n"
                            + "".join(e.get("traceback", [])[-6:]))
                    raise RuntimeError(f"ComfyUI workflow failed: {status}")
                imgs = (entry.get("outputs", {}).get(NODE_SAVE) or {}).get("images")
                if imgs:
                    break
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timed out after {timeout}s. Log tail:\n{_log_tail(15)}")
            if time.time() - last_report >= 30:
                last_report = time.time()
                tail = [l for l in _log_tail(3).splitlines() if l.strip()]
                print(f"  {time.time() - t0:.0f}s ... {tail[-1][-110:] if tail else ''}", flush=True)
            time.sleep(2)

        meta = imgs[0]
        v = requests.get(f"{_base}/view", params={"filename": meta["filename"],
                         "subfolder": meta.get("subfolder", ""), "type": meta.get("type", "output")}, timeout=120)
        v.raise_for_status()
        import io
        out = Image.open(io.BytesIO(v.content)).convert("RGB")
        print(f"Done in {time.time() - t0:.0f}s -> {out.width}x{out.height}")
        return out
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


def launch_ui(share=True):
    """Small Gradio front-end around process()."""
    import gradio as gr
    demo = gr.Interface(
        fn=lambda img, level, mp: process(img, adaptive_level=int(level), max_megapixels=(mp or None)),
        inputs=[gr.Image(type="pil", label="Input"),
                gr.Slider(1, 10, value=DEFAULT_ADAPTIVE_LEVEL, step=1, label="Adaptive level (raise for high-res inputs)"),
                gr.Number(value=0, label="Downscale inputs above N megapixels (0 = keep size)")],
        outputs=gr.Image(type="pil", label="Output"),
        title="SynthID-Bypass v2", flagging_mode="never")
    demo.queue().launch(share=share)
    return demo
