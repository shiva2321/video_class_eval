import base64
import gc
import io
import json
import os
import subprocess
from pathlib import Path

# Add torch\lib to DLL directory for CTranslate2 on Windows
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

import httpx
import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


def main():
    torch_version = torch.__version__
    cuda_available = bool(torch.cuda.is_available())

    if cuda_available:
        free_bytes, _ = torch.cuda.mem_get_info()
        vram_free_mb = int(free_bytes // (1024 * 1024))
    else:
        vram_free_mb = 0

    # ffmpeg check
    try:
        ffmpeg_dir = Path(r"F:\tools\ffmpeg\bin")
        if ffmpeg_dir.exists() and str(ffmpeg_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
        p = subprocess.run(["ffmpeg", "-version"], check=False, capture_output=True, text=True, timeout=5)
        ffmpeg_str = p.stdout.splitlines()[0] if p.returncode == 0 and p.stdout else "error"
    except (subprocess.SubprocessError, OSError) as e:
        ffmpeg_str = f"error: {e}"

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Whisper CUDA load check
    try:
        from faster_whisper import WhisperModel
        sr = 16000
        t = np.linspace(0, 3, sr * 3, endpoint=False, dtype=np.float32)
        tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        whisper_dir = cfg.get("models", {}).get("whisper_dir", r"F:\models\whisper")
        whisper_model_name = cfg.get("models", {}).get("whisper", "large-v3-turbo")
        local_turbo = Path(whisper_dir) / "large-v3-turbo"
        model_path = str(local_turbo) if (local_turbo / "model.bin").exists() else whisper_model_name

        model = WhisperModel(model_path, device="cuda", compute_type="int8_float16", download_root=whisper_dir)
        segments, _ = model.transcribe(tone, beam_size=1)
        list(segments)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        whisper_res = "ok"
    except Exception as e:  # noqa: BLE001
        whisper_res = f"error: {e}"

    # CLIP check
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_id = cfg.get("models", {}).get("clip", "openai/clip-vit-base-patch32")

        clip_model = CLIPModel.from_pretrained(clip_id).to("cuda")
        clip_processor = CLIPProcessor.from_pretrained(clip_id)

        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, 32):
            for j in range(0, 224, 32):
                if ((i // 32) + (j // 32)) % 2 == 0:
                    arr[i : i + 32, j : j + 32] = 255
        img = Image.fromarray(arr)

        inputs = clip_processor(text=["checkerboard", "cat"], images=img, return_tensors="pt", padding=True).to("cuda")
        outputs = clip_model(**inputs)
        _ = outputs.logits_per_image
        del clip_model, clip_processor, inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()
        clip_res = "ok"
    except Exception as e:  # noqa: BLE001
        clip_res = f"error: {e}"

    # Ollama check
    base_url = cfg.get("ollama", {}).get("base_url", "http://127.0.0.1:11434")
    model_tag = cfg.get("models", {}).get("llm", "qwen3-vl:8b")
    try:
        img_buf = io.BytesIO()
        Image.new("RGB", (256, 256), color=(73, 109, 137)).save(img_buf, format="PNG")
        img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }
        payload = {
            "model": model_tag,
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with key 'status' set to 'ok'.",
                    "images": [img_b64],
                }
            ],
            "format": schema,
            "stream": False,
            "keep_alive": 0,
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            resp_data = resp.json()
            content = resp_data.get("message", {}).get("content", "")
            parsed = json.loads(content)
            if "status" in parsed:
                ollama_res = {"model": model_tag, "json_roundtrip": "ok"}
            else:
                ollama_res = {"model": model_tag, "json_roundtrip": f"unexpected: {parsed}"}
    except Exception as e:  # noqa: BLE001
        ollama_res = {"model": model_tag, "json_roundtrip": f"error: {e}"}

    # Pyannote check
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or not hf_token.strip():
        pyannote_res = "skipped: HF_TOKEN not set"
    else:
        try:
            import huggingface_hub
            orig_download = huggingface_hub.hf_hub_download

            def _compat_download(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                return orig_download(*args, **kwargs)

            huggingface_hub.hf_hub_download = _compat_download

            from pyannote.audio import Pipeline
            pyannote_id = cfg.get("models", {}).get("pyannote", "pyannote/speaker-diarization-3.1")
            pipeline = Pipeline.from_pretrained(pyannote_id, use_auth_token=hf_token.strip())
            if pipeline is not None:
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()
                pyannote_res = "ok"
            else:
                pyannote_res = "error: GatedRepoError: access denied or terms not accepted on Hugging Face"
        except Exception as e:  # noqa: BLE001
            first_line = str(e).strip().splitlines()[0] if str(e).strip() else "unknown"
            pyannote_res = f"error: {type(e).__name__}: {first_line}"

    # ECAPA check
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
        ecapa_id = cfg.get("models", {}).get("ecapa", "speechbrain/spkrec-ecapa-voxceleb")
        cache_dir = Path(".cache/ecapa")
        cache_dir.mkdir(parents=True, exist_ok=True)
        ecapa = SpeakerRecognition.from_hparams(
            source=ecapa_id,
            savedir=str(cache_dir),
            run_opts={"device": "cuda:0"},
        )
        del ecapa
        gc.collect()
        torch.cuda.empty_cache()
        ecapa_res = "ok"
    except Exception as e:  # noqa: BLE001
        ecapa_res = f"error: {e}"

    result = {
        "torch": torch_version,
        "cuda": cuda_available,
        "vram_free_mb": vram_free_mb,
        "ffmpeg": ffmpeg_str,
        "whisper_cuda_load": whisper_res,
        "clip": clip_res,
        "ollama": ollama_res,
        "pyannote": pyannote_res,
        "ecapa": ecapa_res,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
