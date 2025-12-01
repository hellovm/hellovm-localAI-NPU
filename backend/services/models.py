import os
import shutil
from pathlib import Path

def models_root(base: Path) -> Path:
    p = base / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p

RECOMMENDED_MODELS = {
    "chat": [
        {"id": "qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen2.5-0.5B-Instruct", "desc": "通义千问超轻量级指令微调模型，适合低显存设备 (0.5B)"},
        {"id": "qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen2.5-1.5B-Instruct", "desc": "通义千问轻量级模型，平衡性能与速度 (1.5B)"},
        {"id": "qwen/Qwen2.5-3B-Instruct", "name": "Qwen2.5-3B-Instruct", "desc": "通义千问中等规模模型，逻辑推理能力强 (3B)"},
        {"id": "qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B-Instruct", "desc": "通义千问标准版，通用能力优秀 (7B)"},
        {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "name": "DeepSeek-R1-Distill-Qwen-1.5B", "desc": "DeepSeek R1 蒸馏版，极高性价比的推理模型 (1.5B)"},
        {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "name": "DeepSeek-R1-Distill-Qwen-7B", "desc": "DeepSeek R1 蒸馏版，强大的思维链能力 (7B)"},
        {"id": "ZhipuAI/glm-4-9b-chat", "name": "GLM-4-9B-Chat", "desc": "智谱AI第四代开源模型，对话流畅 (9B)"},
    ],
    "t2i": [
        {"id": "Tongyi-MAI/Z-Image-Turbo", "name": "Z-Image-Turbo (Tongyi-MAI)", "desc": "阿里达摩院最新文生图模型，8步快速生成，支持双语提示词，低显存友好"},
        {"id": "AI-ModelScope/stable-diffusion-v1-5", "name": "Stable Diffusion v1.5", "desc": "经典文生图模型，生态丰富，兼容性好"},
        {"id": "AI-ModelScope/stable-diffusion-2-1", "name": "Stable Diffusion v2.1", "desc": "生成质量更高的升级版SD模型"},
        {"id": "stabilityai/stable-diffusion-xl-base-1.0", "name": "SDXL Base 1.0", "desc": "高质量大分辨率文生图模型 (需较大显存)"},
    ],
    "t2v": [
        {"id": "damo/text-to-video-synthesis", "name": "ModelScope T2V (Damo)", "desc": "阿里达摩院文生视频模型，生成短视频"},
        {"id": "ali-vilab/text-to-video-ms-1.7b", "name": "ZeroScope T2V", "desc": "轻量级无水印视频生成模型"},
    ]
}

def get_recommended_models():
    return RECOMMENDED_MODELS

def list_models(base: Path):
    root = models_root(base)
    items = []
    for d in root.iterdir():
        if d.is_dir():
            size = 0
            for path, _, files in os.walk(d):
                for f in files:
                    fp = Path(path) / f
                    try:
                        size += fp.stat().st_size
                    except Exception:
                        pass
            kind = None
            try:
                if (d / "model_index.json").exists():
                    kind = "t2i"
                else:
                    for sub in ("openvino", "ov", "runtime", "openvino_model"):
                        if (d / sub / "model_index.json").exists():
                            kind = "t2i"
                            break
                if not kind:
                    if (d / "openvino_model.xml").exists():
                        kind = "llm"
                if not kind:
                    nm = d.name
                    if nm.endswith("_t2v_fp16") or nm.endswith("_t2v_int8"):
                        kind = "t2v"
            except Exception:
                pass
            prec = None
            name = d.name
            try:
                if ("_ov_int8" in name) or ("_quant_int8" in name):
                    prec = "int8"
                elif "_ov_fp16" in name:
                    prec = "fp16"
                elif "_ov_fp32" in name:
                    prec = "fp32"
            except Exception:
                pass
            src = None
            try:
                marks = [".msc", ".mv", ".mdl"]
                for m in marks:
                    if (d / m).exists():
                        src = "modelscope"
                        break
            except Exception:
                pass
            items.append({
                "id": d.name,
                "path": str(d),
                "size_bytes": size,
                "type": kind or "unknown",
                "precision": prec,
                "source": src,
            })
    return items

def delete_model(base: Path, model_id: str):
    root = models_root(base)
    target = root / model_id
    if target.exists():
        shutil.rmtree(target)
        return True
    return False
