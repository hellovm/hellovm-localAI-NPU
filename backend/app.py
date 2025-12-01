import json
import threading
import subprocess
import shlex
import shutil
import os
import queue
from pathlib import Path
from apiflask import APIFlask
from flask import request, jsonify, send_from_directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import warnings
try:
    from torch.onnx import TracerWarning
    warnings.filterwarnings("ignore", category=TracerWarning)
except Exception:
    pass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ... (rest of imports)
from backend.services.system import get_info
from backend.services.models import list_models, delete_model, models_root, get_recommended_models
from backend.services.inference import load_pipeline, generate, quantize_model, is_model_in_use, release_model, is_model_loaded
from backend.utils.tasks import task_store

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = models_root(BASE_DIR)
PERF = {"lat": {"CPU": [], "GPU": [], "NPU": [], "NVIDIA": []}, "ttft": {"CPU": [], "GPU": [], "NPU": [], "NVIDIA": []}, "tpot": {"CPU": [], "GPU": [], "NPU": [], "NVIDIA": []}, "throughput": {"CPU": [], "GPU": [], "NPU": [], "NVIDIA": []}, "gen": {"CPU": [], "GPU": [], "NPU": [], "NVIDIA": []}, "last": {}, "warn": None}
PMODE_STATE = {"CPU": {"mode": "CUMULATIVE_THROUGHPUT", "stable": 0}, "GPU": {"mode": "CUMULATIVE_THROUGHPUT", "stable": 0}, "NPU": {"mode": "CUMULATIVE_THROUGHPUT", "stable": 0}, "NVIDIA": {"mode": "CUMULATIVE_THROUGHPUT", "stable": 0}}

def _choose_perf_mode(config, device):
    key = device if device in PERF["lat"] else ("NPU" if "NPU" in device else ("GPU" if "GPU" in device else ("CPU" if "CPU" in device else "CPU")))
    tt = PERF["ttft"].get(key) or []
    th = PERF["throughput"].get(key) or []
    lat = PERF["lat"].get(key) or []
    def _avg(a):
        return float(sum(a)/len(a)) if a else None
    def _var(a):
        if not a:
            return None
        m = _avg(a)
        return float(sum((x-m)*(x-m) for x in a)/len(a))
    tt_m = _avg(tt) or 0.0
    tt_v = _var(tt) or 0.0
    th_m = _avg(th) or 0.0
    lat_m = _avg(lat) or 0.0
    max_new = int(config.get("max_new_tokens")) if config.get("max_new_tokens") is not None else 512
    num_req = int(config.get("num_requests")) if config.get("num_requests") is not None else 1
    want_thr = (max_new >= 512 or num_req > 1) or (th_m >= 1.0 and tt_m >= 800)
    want_lat = tt_m >= 1500 and th_m < 2.0
    st = PMODE_STATE.get(key)
    cur = st["mode"]
    if want_thr and cur != "THROUGHPUT":
        st["stable"] += 1
        if st["stable"] >= 3:
            st["mode"] = "THROUGHPUT"
            st["stable"] = 0
    elif want_lat and cur != "LATENCY":
        st["stable"] += 1
        if st["stable"] >= 3:
            st["mode"] = "LATENCY"
            st["stable"] = 0
    else:
        st["stable"] = max(0, st["stable"]-1)
    return st["mode"]

app = APIFlask(
    __name__,
    title="AI Funland",
    version="V0.0.1 Dev",
    static_folder=str(BASE_DIR / "web"),
    static_url_path="/static",
)

@app.get("/")
def index():
    return app.send_static_file("index.html")

 

@app.get("/api/system/info")
def api_system_info():
    return jsonify(get_info())

def _pick_default_model_id():
    import os
    env_id = os.environ.get("AIFUNLAND_DEFAULT_MODEL_ID")
    if env_id:
        return env_id
    items = list_models(BASE_DIR)
    if not items:
        return None
    try:
        items.sort(key=lambda x: ((1 if ("_quant_int8" in str(x.get("id"))) else 0), int(x.get("size_bytes") or 0)), reverse=True)
    except Exception:
        pass
    return items[0]["id"] if items else None

def _preload_on_start():
    try:
        import os
        mid = _pick_default_model_id()
        if not mid:
            return
        dev = os.environ.get("AIFUNLAND_DEFAULT_DEVICE") or "HETERO:NPU,GPU"
        cfg = {"perf_mode": "LATENCY", "hetero_enable": True, "max_prompt_len": 512, "min_response_len": 8, "auto_multi": True, "prefill_igpu_decode_npu": True}
        model_dir = MODELS_DIR / mid
        def _bg():
            try:
                pipe = load_pipeline(model_dir, dev, cfg)
                try:
                    g2 = pipe.get_generation_config()
                    g2.max_new_tokens = 1
                    _ = pipe.generate("warmup", g2)
                except Exception:
                    pass
            except Exception:
                pass
        threading.Thread(target=_bg, daemon=True).start()
    except Exception:
        pass

@app.get("/api/models/list")
def api_models_list():
    return jsonify({"items": list_models(BASE_DIR)})

@app.get("/api/models/recommend")
def api_models_recommend():
    return jsonify(get_recommended_models())

@app.get("/api/models/is_loaded")
def api_models_is_loaded():
    model_id = request.args.get("model_id")
    device = request.args.get("device", "CPU")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    model_dir = MODELS_DIR / model_id.replace("/", "__")
    ok = is_model_loaded(model_dir, device)
    return jsonify({"loaded": bool(ok)})

def _get_cache_dir():
    import os
    v = os.environ.get("AIFUNLAND_CACHE_DIR")
    return Path(v) if v else (BASE_DIR / "tmp")

def _run_modelscope_download(task_id, model_id, local_dir, include=None, exclude=None, revision=None):
    import sys as _sys
    import re as _re
    exe = str((Path(_sys.executable).parent / "Scripts" / "modelscope.exe"))
    pyexe = str(_sys.executable)
    use_exe = Path(exe).exists()
    base_cmd = ([exe] if use_exe else [pyexe, "-m", "modelscope"]) + ["download", "--model", model_id, "--local_dir", str(local_dir)]
    if revision:
        base_cmd += ["--revision", revision]
    if include:
        base_cmd += ["--include"] + include
    if exclude:
        base_cmd += ["--exclude"] + exclude
    try:
        cache_dir = _get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        task_store.update(task_id, status="running", progress=1, message=f"starting: cache={cache_dir}")
        proc = subprocess.Popen(base_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR), env={**_os_environ(cache_dir)})
        for line in proc.stdout:
            s = line.strip()
            if not s:
                continue
            if ("uvicorn" in s) or ("UvicornWorker" in s):
                continue
            task_store.update(task_id, message=s)
            m = _re.search(r"(\d{1,3})%", s)
            if m:
                try:
                    pct = int(m.group(1))
                    task_store.update(task_id, progress=max(0, min(100, pct)))
                except Exception:
                    pass
        code = proc.wait()
        if code == 0:
            task_store.complete(task_id, result=str(local_dir))
            try:
                from backend.services.inference import export_model_ir
                save_dir = MODELS_DIR / (local_dir.name + "_ov_fp32")
                threading.Thread(target=lambda: export_model_ir(local_dir, save_dir), daemon=True).start()
            except Exception:
                pass
        else:
            parts = model_id.split("/")
            if len(parts) == 2 and parts[0] != parts[0].lower():
                alt_id = parts[0].lower() + "/" + parts[1]
                alt_cmd = base_cmd.copy()
                i = alt_cmd.index("--model")
                alt_cmd[i+1] = alt_id
                task_store.update(task_id, message="retry")
                proc2 = subprocess.Popen(alt_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR), env={**_os_environ(cache_dir)})
                for line in proc2.stdout:
                    s2 = line.strip()
                    if not s2:
                        continue
                    if ("uvicorn" in s2) or ("UvicornWorker" in s2):
                        continue
                    task_store.update(task_id, message=s2)
                    m2 = _re.search(r"(\d{1,3})%", s2)
                    if m2:
                        try:
                            pct2 = int(m2.group(1))
                            task_store.update(task_id, progress=max(0, min(100, pct2)))
                        except Exception:
                            pass
                code2 = proc2.wait()
                if code2 == 0:
                    task_store.complete(task_id, result=str(local_dir))
                    try:
                        from backend.services.inference import export_model_ir
                        save_dir = MODELS_DIR / (local_dir.name + "_ov_fp32")
                        threading.Thread(target=lambda: export_model_ir(local_dir, save_dir), daemon=True).start()
                    except Exception:
                        pass
                    return
            try:
                from modelscope import snapshot_download
                task_store.update(task_id, message="api_download")
                def _run_api():
                    p = snapshot_download(model_id, cache_dir=str(cache_dir), revision=revision)
                    src = Path(p)
                    if local_dir.exists():
                        try:
                            shutil.rmtree(local_dir)
                        except Exception:
                            pass
                    shutil.move(str(src), str(local_dir))
                t = threading.Thread(target=_run_api, daemon=True)
                t.start()
                def _monitor_progress():
                    import time
                    pct = 3
                    while t.is_alive():
                        try:
                            task_store.update(task_id, status="running", progress=pct, message="api_download")
                            pct = pct + 2 if pct < 40 else (pct + 3 if pct < 75 else (pct + 1))
                            if pct > 95:
                                pct = 95
                        except Exception:
                            pass
                        time.sleep(1)
                mon = threading.Thread(target=_monitor_progress, daemon=True)
                mon.start()
                t.join()
                task_store.complete(task_id, result=str(local_dir))
                try:
                    from backend.services.inference import export_model_ir
                    save_dir = MODELS_DIR / (local_dir.name + "_ov_fp32")
                    threading.Thread(target=lambda: export_model_ir(local_dir, save_dir), daemon=True).start()
                except Exception:
                    pass
            except Exception as e2:
                task_store.update(task_id, status="error", error=f"exit {code}: {str(e2)}")
    except Exception as e:
        task_store.update(task_id, status="error", error=str(e))

def _os_environ(cache_dir: Path = None):
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if cache_dir is None:
        cache_dir = _get_cache_dir()
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    env["MODELSCOPE_CACHE"] = str(cache_dir)
    env["MS_CACHE_HOME"] = str(cache_dir)
    return env

def _run_modelscope_t2i_download_and_convert(task_id: str, model_id: str, precision: str):
    try:
        cache_dir = _get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = MODELS_DIR / model_id.replace("/", "__")
        out_dir = MODELS_DIR / (raw_dir.name + ("_ov_int8" if precision == "int8" else "_ov_fp16"))
        env = _os_environ(cache_dir)
        try:
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            env["HF_HUB_ENABLE_DOWNLOAD_ACCELERATION"] = "1"
        except Exception:
            pass
        exe = str((Path(sys.executable).parent / "Scripts" / "modelscope.exe"))
        pyexe = str(sys.executable)
        use_exe = Path(exe).exists()
        ok = False
        for attempt in range(3):
            cmd = ([exe] if use_exe else [pyexe, "-m", "modelscope"]) + ["download", "--model", model_id, "--local_dir", str(raw_dir)]
            task_store.update(task_id, status="running", progress=max(1, 3*attempt+1), message=f"download_cli_{attempt+1}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR), env=env)
            pct = 3
            for line in proc.stdout:
                s = (line or "").strip()
                if not s:
                    continue
                task_store.update(task_id, message=s)
                pct = min(60, pct + (2 if pct < 40 else 1))
                task_store.update(task_id, progress=pct)
            code = proc.wait()
            if code == 0:
                ok = True
                break
        if not ok:
            try:
                from modelscope import snapshot_download
                for attempt2 in range(3):
                    task_store.update(task_id, message=f"api_download_{attempt2+1}")
                    p = snapshot_download(model_id, cache_dir=str(cache_dir))
                    src = Path(p)
                    if raw_dir.exists():
                        shutil.rmtree(raw_dir)
                    shutil.move(str(src), str(raw_dir))
                    ok = True
                    break
            except Exception:
                ok = False
        if not ok:
            task_store.update(task_id, status="error", error="modelscope_download_failed")
            return
        task_store.update(task_id, message="convert")
        task_store.update(task_id, progress=65)
        try:
            exe2 = sys.executable
            env2 = _os_environ(cache_dir)
            cmd2 = [exe2, "-m", "optimum.exporters.openvino.convert", "--model", str(raw_dir), "--output", str(out_dir), "--trust-remote-code", "--weight-format", ("int8" if precision == "int8" else "fp16")]
            proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR), env=env2)
            pct2 = 70
            for line2 in proc2.stdout:
                s2 = (line2 or "").strip()
                if not s2:
                    continue
                task_store.update(task_id, message=s2)
                pct2 = min(95, pct2 + (2 if pct2 < 85 else 1))
                task_store.update(task_id, progress=pct2)
            code2 = proc2.wait()
            if code2 != 0:
                task_store.update(task_id, status="error", error="convert_failed")
                return
        except Exception as e3:
            task_store.update(task_id, status="error", error=str(e3))
            return
        try:
            idx = out_dir / "model_index.json"
            if not idx.exists():
                raise RuntimeError("model_index_missing")
        except Exception as e4:
            task_store.update(task_id, status="error", error=str(e4))
            return
        task_store.complete(task_id, result=str(out_dir))
    except Exception as e:
        task_store.update(task_id, status="error", error=str(e))

@app.post("/api/image/ms_download_and_convert")
def api_image_ms_download_and_convert():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    precision = str(data.get("precision") or "fp16").lower()
    if precision not in ("fp16", "int8"):
        return jsonify({"error": "invalid_precision", "message": "Precision must be fp16 or int8"}), 400
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    task_id = task_store.create("t2i_ms_export")
    t = threading.Thread(target=_run_modelscope_t2i_download_and_convert, args=(task_id, model_id, precision), daemon=True)
    t.start()
    return jsonify({"task_id": task_id})

@app.post("/api/models/download")
def api_models_download():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    include = data.get("include")
    exclude = data.get("exclude")
    revision = data.get("revision")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    dest = MODELS_DIR / model_id.replace("/", "__")
    task_id = task_store.create("download")
    t = threading.Thread(target=_run_modelscope_download, args=(task_id, model_id, dest, include, exclude, revision), daemon=True)
    t.start()
    return jsonify({"task_id": task_id})

@app.post("/api/models/export_ir")
def api_models_export_ir():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    src = MODELS_DIR / model_id.replace("/", "__")
    if not src.exists():
        return jsonify({"error": "model_not_found"}), 404
    dest = MODELS_DIR / (src.name + "_ov_fp32")
    task_id = task_store.create("export_ir")
    def _bg():
        try:
            from backend.services.inference import export_model_ir
            task_store.update(task_id, status="running", progress=1, message="exporting")
            result = export_model_ir(src, dest)
            task_store.complete(task_id, result=result)
        except Exception as e:
            task_store.update(task_id, status="error", error=str(e))
    threading.Thread(target=_bg, daemon=True).start()
    return jsonify({"task_id": task_id})

@app.get("/api/tasks/<task_id>")
def api_task_status(task_id):
    t = task_store.get(task_id)
    if not t:
        return jsonify({"error": "not_found"}), 404
    return jsonify(t)

@app.get("/api/tasks/stream/<task_id>")
def api_task_stream(task_id):
    def _stream():
        q = task_store.subscribe(task_id)
        try:
            while True:
                try:
                    # Wait for update (timeout to keep connection alive)
                    data = q.get(timeout=15)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("status") in ("completed", "error"):
                        break
                except queue.Empty:
                    # Keep-alive comment
                    yield ": keep-alive\n\n"
                    # check if task still exists/valid
                    curr = task_store.get(task_id)
                    if not curr:
                        break
        except GeneratorExit:
            pass
        finally:
            task_store.unsubscribe(task_id, q)

    return app.response_class(_stream(), mimetype="text/event-stream")

@app.post("/api/models/quantize")
def api_models_quantize():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    mode = data.get("mode", "int8")
    params = data.get("params")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    src = MODELS_DIR / model_id.replace("/", "__")
    if not src.exists():
        return jsonify({"error": "model_not_found"}), 404
    out = MODELS_DIR / (src.name + f"_quant_{mode}")
    out.mkdir(parents=True, exist_ok=True)
    task_id = task_store.create("quantize")

    def _run():
        task_store.update(task_id, status="running", message="quantizing")
        try:
            result = quantize_model(src, out, mode, params)
            try:
                release_model(src)
            except Exception:
                pass
            try:
                import shutil
                shutil.rmtree(src)
            except Exception:
                pass
            task_store.complete(task_id, result=result)
        except Exception as e:
            msg = str(e)
            if "int4_disabled" in msg:
                msg = "系统已禁用 INT4 量化。仅支持 INT8 作为默认方案。"
            if ("does not recognize this architecture" in msg) or ("model type" in msg and "qwen3" in msg.lower()):
                msg = "当前模型架构(qwen3)不受当前 Transformers/Optimum 版本支持，建议：1) 升级 Transformers/Optimum 至支持 qwen3 的版本 2) 更换为兼容模型 (如 qwen/Qwen2.5-0.5B/1.5B/3B-Instruct)。"
            if ("Calibration dataset is required" in msg) or ("requires dataset" in msg):
                msg = "当前量化策略需要校准数据或数据感知选项，请提供 dataset 或选择仅权重量化（INT8）。"
            task_store.update(task_id, status="error", error=msg)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"task_id": task_id})

@app.delete("/api/models/delete")
def api_models_delete():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    target = MODELS_DIR / model_id.replace("/", "__")
    if is_model_in_use(target):
        return jsonify({
            "error_code": "model_in_use",
            "friendly": True,
            "message": "模型正在使用，无法删除。请先释放模型后再删除。",
            "action": "release_then_delete"
        }), 409
    try:
        ok = delete_model(BASE_DIR, model_id.replace("/", "__"))
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({
            "error_code": "delete_failed",
            "friendly": True,
            "message": "删除失败，可能由于文件占用。请释放模型后重试。",
            "detail": str(e)
        }), 409

@app.post("/api/models/release")
def api_models_release():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    target = MODELS_DIR / model_id.replace("/", "__")
    release_model(target)
    return jsonify({"ok": True})

def _validate_chat_config(config):
    try:
        if "max_new_tokens" in config:
            mnt = int(config["max_new_tokens"])
            if mnt < 1 or mnt > 8192:
                return "max_new_tokens must be between 1 and 8192"
        if "temperature" in config:
            temp = float(config["temperature"])
            if temp < 0.0 or temp > 2.0:
                return "temperature must be between 0.0 and 2.0"
        if "top_p" in config:
            tp = float(config["top_p"])
            if tp < 0.0 or tp > 1.0:
                return "top_p must be between 0.0 and 1.0"
        if "top_k" in config:
            tk = int(config["top_k"])
            if tk < 0:
                return "top_k must be non-negative"
        if "repetition_penalty" in config:
            rp = float(config["repetition_penalty"])
            if rp < 0.1 or rp > 10.0: # reasonable range
                 return "repetition_penalty must be between 0.1 and 10.0"
    except ValueError:
        return "Invalid numeric parameters in config"
    return None

@app.post("/api/infer/chat")
def api_infer_chat():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    device = data.get("device", "CPU")
    prompt = data.get("prompt")
    config = data.get("config", {})
    
    if not model_id or not prompt:
        return jsonify({"error": "model_id and prompt required"}), 400

    # Validation for config parameters
    err_msg = _validate_chat_config(config)
    if err_msg:
        return jsonify({"error": "invalid_parameter", "message": err_msg}), 400

    if "hetero_enable" not in config:
        config["hetero_enable"] = True
    
    model_dir = MODELS_DIR / model_id.replace("/", "__")
    if not model_dir.exists():
        return jsonify({"error": "model_not_found"}), 404
    
    try:
        import time
        t0 = time.time()
        if str(config.get("perf_mode", "")).upper() == "AUTO":
            config["perf_mode"] = _choose_perf_mode(config, device)
        if config.get("web_search"):
            try:
                from backend.services.inference import web_search, augment_with_sources
                q = config.get("search_query") or prompt
                sources = web_search(q, max_results=5)
                prompt = augment_with_sources(prompt, sources, lang="zh")
            except Exception:
                pass
        pipe = load_pipeline(model_dir, device, config)
        cur_dev = getattr(pipe, "_af_device", device)
        cur_real = getattr(pipe, "_af_device_real", cur_dev)
        output, metrics = generate(pipe, prompt, config)
        try:
            s = str(output)
            p = s.lower().find("</think>")
            if p != -1:
                output = s[p+8:].strip()
        except Exception:
            pass
        dt = int((time.time() - t0) * 1000)
        
        # Performance tracking
        key = cur_dev if cur_dev in PERF["lat"] else ("NPU" if "NPU" in cur_dev else ("GPU" if "GPU" in cur_dev else ("CPU" if "CPU" in cur_dev else None)))
        arr = PERF["lat"].get(key)
        if arr is not None:
            arr.append(dt)
            if len(arr) > 30:
                del arr[:len(arr)-30]
        
        # Fallback detection
        fb = False
        try:
            sd = str(cur_dev)
            sr = str(cur_real)
            fb = (sd.startswith("HETERO") and (not sr.startswith("HETERO"))) and (sd != sr)
        except Exception:
            fb = False
        PERF["last"] = {"device": cur_dev, "real_device": cur_real, "fallback": fb, "latency_ms": dt, "metrics": metrics}
        
        if metrics and key:
            try:
                if metrics.get("ttft_ms"):
                    PERF["ttft"][key].append(metrics["ttft_ms"]) 
                if metrics.get("tpot_ms"):
                    PERF["tpot"][key].append(metrics["tpot_ms"]) 
                if metrics.get("throughput_tps"):
                    PERF["throughput"][key].append(metrics["throughput_tps"]) 
                if metrics.get("generate_ms"):
                    PERF["gen"][key].append(metrics["generate_ms"]) 
                for k in ("ttft","tpot","throughput","gen"):
                    if len(PERF[k][key]) > 30:
                        del PERF[k][key][:len(PERF[k][key])-30]
            except Exception:
                pass
        
        cpu_avg = sum(PERF["lat"]["CPU"]) / len(PERF["lat"]["CPU"]) if PERF["lat"]["CPU"] else None
        npu_avg = sum(PERF["lat"]["NPU"]) / len(PERF["lat"]["NPU"]) if PERF["lat"]["NPU"] else None
        PERF["warn"] = None
        if (device == "NPU" or ("NPU" in device)) and cpu_avg is not None and npu_avg is not None and npu_avg > cpu_avg * 1.2:
            PERF["warn"] = "npu_slower_than_cpu"
            
        logger.info(f"Chat generation successful: model={model_id}, device={cur_dev}, dt={dt}ms")
        return jsonify({"output": output, "metrics": metrics})
    except Exception as e:
        msg = str(e)
        logger.error(f"Chat generation failed: {msg}", exc_info=True)
        if ("No more devices are left" in msg) or ("Failed to compile" in msg and "devices" in msg):
            return jsonify({
                "error_code": "npu_no_device_left",
                "friendly": True,
                "message": "NPU 编译失败或设备不可用。建议：1) 在设置页将 Streams/并发请求设为 1 2) 关闭协同或改用 GPU 3) 点击‘释放模型’后重试 4) 更新 Intel NPU 驱动与 OpenVINO 版本",
                "recommended": [
                    "设置 OV_NUM_STREAMS=1",
                    "设置 OV_HINT_NUM_REQUESTS=1",
                    "切换设备为 GPU 或 CPU",
                    "使用 MULTI:GPU,CPU",
                    "释放模型后重试"
                ],
                "device": device
            }), 200
        if ("does not recognize this architecture" in msg) or ("model type" in msg and "qwen3" in msg.lower()):
            return jsonify({
                "error_code": "unsupported_architecture",
                "friendly": True,
                "message": "当前模型架构(qwen3)不受当前 Transformers/Optimum 版本支持，建议：1) 升级 Transformers/Optimum 至支持 qwen3 的版本 2) 更换为兼容模型 (如 qwen/Qwen2.5-0.5B/1.5B/3B-Instruct)",
                "device": device
            }), 200
        if device in ("NPU", "GPU") and ("Could not find a model" in msg or "Jenkins" in msg or "not supported" in msg):
            return jsonify({
                "error_code": "incompatible_acceleration",
                "friendly": True,
                "message": "当前模型不支持硬件加速功能，建议：1) 更换为兼容模型 2) 使用CPU模式运行",
                "docs_link": "https://github.com/openvinotoolkit/openvino.genai",
                "recommended": [
                    "qwen/Qwen2.5-0.5B-Instruct",
                    "qwen/Qwen2.5-1.5B-Instruct",
                    "qwen/Qwen2.5-3B-Instruct"
                ],
                "device": device
            }), 200
        if "bad allocation" in msg or "Memory" in msg:
             return jsonify({
                "error_code": "memory_error",
                "friendly": True,
                "message": "内存/显存不足，无法加载或运行模型。请尝试：1) 关闭其他占用内存的应用 2) 使用较小的模型 (如 INT4 量化版) 3) 减小 max_new_tokens 参数",
                "device": device
            }), 200
        return jsonify({"error": "internal_error", "message": msg}), 500

@app.post("/api/infer/preload")
def api_infer_preload():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    device = data.get("device", "CPU")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    model_dir = MODELS_DIR / model_id.replace("/", "__")
    if not model_dir.exists():
        return jsonify({"error": "model_not_found"}), 404
    cfg = data.get("config", {})
    if "hetero_enable" not in cfg:
        cfg["hetero_enable"] = True
    if "auto_multi" not in cfg:
        cfg["auto_multi"] = True
    if "prefill_igpu_decode_npu" not in cfg:
        cfg["prefill_igpu_decode_npu"] = True
    if str(cfg.get("perf_mode", "")).upper() == "AUTO":
        try:
            cfg["perf_mode"] = _choose_perf_mode(cfg, device)
        except Exception:
            pass
    def _bg():
        try:
            pipe = load_pipeline(model_dir, device, cfg)
            try:
                g2 = pipe.get_generation_config()
                g2.max_new_tokens = 1
                _ = pipe.generate("warmup", g2)
            except Exception:
                pass
        except Exception:
            pass
    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return jsonify({"ok": True, "async": True})
def _encode_bmp(arr):
    import numpy as np, struct
    h = int(arr.shape[0])
    w = int(arr.shape[1])
    if arr.shape[-1] != 3:
        raise ValueError("unsupported_channels")
    bgr = arr.astype('uint8')[:, :, ::-1]
    row_size = ((24 * w + 31) // 32) * 4
    pad = row_size - w * 3
    buf = bytearray()
    for y in range(h):
        buf.extend(bgr[y].tobytes())
        if pad:
            buf.extend(b"\x00" * pad)
    pixel_bytes = bytes(buf)
    file_size = 14 + 40 + len(pixel_bytes)
    header = bytearray()
    header.extend(b"BM")
    header.extend(struct.pack('<IHHI', file_size, 0, 0, 14 + 40))
    header.extend(struct.pack('<IiiHHIIiiII', 40, w, -h, 1, 24, 0, len(pixel_bytes), 0, 0, 0, 0))
    return bytes(header) + pixel_bytes
@app.post("/api/image/generate")
def api_image_generate():
    data = request.get_json(force=True)
    model_path = data.get("model_path")
    device = data.get("device")
    prompt = data.get("prompt")
    
    # Determine defaults based on model
    is_turbo = "turbo" in str(model_path).lower() or "z-image" in str(model_path).lower()
    default_guidance = 0.0 if is_turbo else 7.5
    default_steps = 8 if is_turbo else 30

    # Validation
    try:
        width = int(data.get("width") or 512)
        height = int(data.get("height") or 512)
        steps = int(data.get("steps") or default_steps)
        guidance = float(data.get("guidance_scale") if data.get("guidance_scale") is not None else default_guidance)
    except ValueError:
        return jsonify({"error": "invalid_parameters", "message": "Invalid numeric parameters"}), 400

    if width % 8 != 0 or height % 8 != 0:
        return jsonify({"error": "invalid_dimensions", "message": "Width and height must be divisible by 8"}), 400
    
    if width < 256 or height < 256:
        return jsonify({"error": "dimension_too_small", "message": "Min dimension is 256"}), 400

    if width > 2048 or height > 2048:
         return jsonify({"error": "dimension_too_large", "message": "Max dimension is 2048"}), 400

    if steps < 1 or steps > 100:
        return jsonify({"error": "invalid_steps", "message": "Steps must be between 1 and 100"}), 400
        
    if guidance < 0.0 or guidance > 20.0:
        return jsonify({"error": "invalid_guidance", "message": "Guidance scale must be between 0.0 and 20.0"}), 400

    # heterogeneous devices
    te_dev = data.get("text_encoder_device")
    unet_dev = data.get("unet_device")
    vae_dev = data.get("vae_decoder_device") or data.get("vae_device")
    if not model_path or not prompt:
        return jsonify({"error": "model_path and prompt required"}), 400
    mdir = MODELS_DIR / model_path if (model_path and ("/" not in model_path) and ("\\" not in model_path)) else Path(model_path)
    if not mdir.exists():
        try:
            alt = MODELS_DIR / model_path.replace("/", "__").replace("\\", "__")
        except Exception:
            alt = None
        if alt and alt.exists():
            mdir = alt
        else:
            cand_names = []
            try:
                base = model_path.replace("/", "__").replace("\\", "__")
                cand_names = [base + s for s in ("_ov_fp16", "_ov_int8", "_ov_fp32", "")]
            except Exception:
                pass
            found = None
            for name in cand_names:
                p = MODELS_DIR / name
                try:
                    if p.exists():
                        found = p
                        break
                except Exception:
                    pass
            if found is not None:
                mdir = found
            else:
                return jsonify({"error": "model_not_found"}), 404
    # ensure Text2ImagePipeline dir contains model_index.json; if not, try fallback to sibling raw dir
    try:
        idx = mdir / "model_index.json"
        if not idx.exists():
            # prefer sibling converted dirs
            sibs = [mdir.parent / (mdir.name + s) for s in ("_ov_fp16", "_ov_int8", "_ov_fp32")]
            for s in sibs:
                if (s / "model_index.json").exists():
                    mdir = s
                    break
            # if still missing, try common subdirectories
            if not (mdir / "model_index.json").exists():
                for sub in ("openvino", "ov", "runtime", "openvino_model"):
                    cand = mdir / sub
                    if (cand / "model_index.json").exists():
                        mdir = cand
                        break
            # final fallback: find first model_index.json under tree
            if not (mdir / "model_index.json").exists():
                try:
                    found = next(mdir.glob("**/model_index.json"), None)
                    if found:
                        mdir = found.parent
                except Exception:
                    pass
    except Exception:
        pass
    try:
        from backend.services.inference import load_t2i_pipeline, t2i_generate
        devs = None
        if te_dev or unet_dev or vae_dev:
            devs = {"text_encoder": te_dev or "CPU", "unet": unet_dev or (te_dev or "CPU"), "vae_decoder": vae_dev or (unet_dev or te_dev or "CPU")}
        else:
            devs = device or "CPU"
        props = {"CACHE_DIR": str((Path(os.environ.get("AIFUNLAND_CACHE_DIR") or (Path.cwd() / "tmp")) / "ov_cache").resolve())}
        idx_final = mdir / "model_index.json"
        if not idx_final.exists():
            return jsonify({"error": "model_index_missing"}), 404
        pipe = load_t2i_pipeline(mdir, devs, props)
        image_tensor = t2i_generate(pipe, prompt, width=width, height=height, steps=steps, guidance_scale=guidance)
        import base64
        arr = getattr(image_tensor, 'data', None)
        if arr is None:
            return jsonify({"error": "no_image_data"}), 500
        img = arr[0]
        bmp = _encode_bmp(img)
        b64 = base64.b64encode(bmp).decode('ascii')
        return jsonify({"mime": "image/bmp", "image_b64": b64, "width": int(img.shape[1]), "height": int(img.shape[0])})
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}", exc_info=True)
        return jsonify({"error": "internal_error", "message": str(e)}), 500
def _slugify_model_id(mid: str) -> str:
    return mid.replace('/', '__')
def _run_ov_export(task_id, hf_id, out_dir, precision):
    try:
        exe = sys.executable
        env = _os_environ()
        cmd = [exe, "-m", "optimum.exporters.openvino.convert", "--model", hf_id, "--output", str(out_dir), "--trust-remote-code", "--weight-format", ("int8" if precision=="int8" else "fp16")]
        task_store.update(task_id, status="running", progress=1, message="starting")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR), env=env)
        pct = 3
        for line in proc.stdout:
            s = (line or "").strip()
            if not s:
                continue
            task_store.update(task_id, message=s)
            try:
                pct = min(95, pct + (2 if pct < 40 else (3 if pct < 75 else 1)))
                task_store.update(task_id, progress=pct)
            except Exception:
                pass
        code = proc.wait()
        if code == 0:
            task_store.complete(task_id, result=str(out_dir))
        else:
            task_store.update(task_id, status="error", error=f"exit {code}")
    except Exception as e:
        task_store.update(task_id, status="error", error=str(e))

@app.post("/api/image/download_model")
def api_image_download_model():
    data = request.get_json(force=True)
    hf_id = data.get("hf_id")
    precision = str(data.get("precision") or "fp16").lower()
    if precision not in ("fp16", "int8"):
        return jsonify({"error": "invalid_precision", "message": "Precision must be fp16 or int8"}), 400
    if not hf_id:
        return jsonify({"error": "hf_id required"}), 400
    slug = _slugify_model_id(hf_id)
    out_name = f"sd__{slug}_ov_{'int8' if precision=='int8' else 'fp16'}"
    out_dir = MODELS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = task_store.create("ov_export")
    t = threading.Thread(target=_run_ov_export, args=(task_id, hf_id, out_dir, precision), daemon=True)
    t.start()
    return jsonify({"task_id": task_id})

def _video_root():
    p = BASE_DIR / "tmp" / "videos"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

def _run_modelscope_t2v_download_and_convert(task_id: str, model_id: str, precision: str):
    try:
        cache_dir = _get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = MODELS_DIR / model_id.replace("/", "__")
        out_dir = MODELS_DIR / (raw_dir.name + ("_t2v_int8" if precision == "int8" else "_t2v_fp16"))
        env = _os_environ(cache_dir)
        exe = str((Path(sys.executable).parent / "Scripts" / "modelscope.exe"))
        pyexe = str(sys.executable)
        use_exe = Path(exe).exists()
        ok = False
        for attempt in range(3):
            cmd = ([exe] if use_exe else [pyexe, "-m", "modelscope"]) + ["download", "--model", model_id, "--local_dir", str(raw_dir)]
            task_store.update(task_id, status="running", progress=max(1, 3*attempt+1), message=f"download_cli_{attempt+1}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR), env=env)
            pct = 3
            for line in proc.stdout:
                s = (line or "").strip()
                if not s:
                    continue
                task_store.update(task_id, message=s)
                pct = min(60, pct + (2 if pct < 40 else 1))
                task_store.update(task_id, progress=pct)
            code = proc.wait()
            if code == 0:
                ok = True
                break
        if not ok:
            try:
                from modelscope import snapshot_download
                for attempt2 in range(3):
                    task_store.update(task_id, message=f"api_download_{attempt2+1}")
                    p = snapshot_download(model_id, cache_dir=str(cache_dir))
                    src = Path(p)
                    if raw_dir.exists():
                        shutil.rmtree(raw_dir)
                    shutil.move(str(src), str(raw_dir))
                    ok = True
                    break
            except Exception:
                ok = False
        if not ok:
            task_store.update(task_id, status="error", error="modelscope_download_failed")
            return
        task_store.update(task_id, message="convert")
        task_store.update(task_id, progress=65)
        converted = False
        try:
            exe2 = sys.executable
            env2 = _os_environ(cache_dir)
            cmd2 = [exe2, "-m", "optimum.exporters.openvino.convert", "--model", str(raw_dir), "--output", str(out_dir), "--trust-remote-code", "--weight-format", ("int8" if precision == "int8" else "fp16")]
            proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR), env=env2)
            pct2 = 70
            for line2 in proc2.stdout:
                s2 = (line2 or "").strip()
                if not s2:
                    continue
                task_store.update(task_id, message=s2)
                pct2 = min(95, pct2 + (2 if pct2 < 85 else 1))
                task_store.update(task_id, progress=pct2)
            code2 = proc2.wait()
            if code2 == 0:
                converted = True
        except Exception:
            converted = False
        if converted:
            try:
                idx = out_dir / "model_index.json"
                if not idx.exists():
                    raise RuntimeError("model_index_missing")
            except Exception:
                pass
            task_store.complete(task_id, result=str(out_dir))
        else:
            task_store.update(task_id, message="convert_failed")
            task_store.complete(task_id, result=str(raw_dir))
    except Exception as e:
        task_store.update(task_id, status="error", error=str(e))

@app.post("/api/video/ms_download_and_convert")
def api_video_ms_download_and_convert():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    precision = str(data.get("precision") or "fp16").lower()
    if precision not in ("fp16", "int8"):
        return jsonify({"error": "invalid_precision", "message": "Precision must be fp16 or int8"}), 400
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    task_id = task_store.create("t2v_ms_export")
    t = threading.Thread(target=_run_modelscope_t2v_download_and_convert, args=(task_id, model_id, precision), daemon=True)
    t.start()
    return jsonify({"task_id": task_id})

@app.get("/api/video/get/<name>")
def api_video_get(name):
    root = _video_root()
    f = root / name
    if not f.exists():
        return jsonify({"error": "not_found"}), 404
    return send_from_directory(str(root), name)

@app.post("/api/video/generate")
def api_video_generate():
    data = request.get_json(force=True)
    model_path = data.get("model_path")
    prompt = data.get("prompt")
    seconds = int(data.get("seconds") or 4)
    fps = int(data.get("fps") or 8)

    if seconds < 1 or seconds > 60:
        return jsonify({"error": "invalid_seconds", "message": "Seconds must be between 1 and 60"}), 400
    if fps < 1 or fps > 60:
        return jsonify({"error": "invalid_fps", "message": "FPS must be between 1 and 60"}), 400

    if not model_path or not prompt:
        return jsonify({"error": "model_path and prompt required"}), 400
    mdir = MODELS_DIR / model_path if (model_path and ("/" not in model_path) and ("\\" not in model_path)) else Path(model_path)
    if not mdir.exists():
        alt = MODELS_DIR / model_path.replace("/", "__").replace("\\", "__")
        if alt.exists():
            mdir = alt
        else:
            return jsonify({"error": "model_not_found"}), 404
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        p = pipeline(task=Tasks.text_to_video_synthesis, model=str(mdir))
        out = p({"text": prompt})
        vid = out.get("output_video") or out.get("video")
        if not vid:
            return jsonify({"error": "no_video"}), 500
        src = Path(vid)
        if not src.exists():
            return jsonify({"error": "no_video_file"}), 500
        root = _video_root()
        import uuid
        name = f"{uuid.uuid4().hex}{src.suffix or '.mp4'}"
        dst = root / name
        try:
            shutil.copyfile(str(src), str(dst))
        except Exception:
            return jsonify({"error": "copy_failed"}), 500
        return jsonify({"video_url": f"/api/video/get/{name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.get("/api/infer/stream")
def api_infer_stream():
    model_id = request.args.get("model_id")
    device = request.args.get("device", "CPU")
    prompt = request.args.get("prompt")
    cfg_s = request.args.get("config")
    try:
        config = json.loads(cfg_s) if cfg_s else {}
    except Exception:
        config = {}
    
    # Validation
    err_msg = _validate_chat_config(config)
    if err_msg:
        def _err_param():
            yield "event: server_error\n"
            yield "data: " + json.dumps({"error": "invalid_parameter", "message": err_msg}) + "\n\n"
        return app.response_class(_err_param(), mimetype="text/event-stream")

    if "hetero_enable" not in config:
        config["hetero_enable"] = True
    if "auto_multi" not in config:
        config["auto_multi"] = True
    if "prefill_igpu_decode_npu" not in config:
        config["prefill_igpu_decode_npu"] = True
    if not model_id or not prompt:
        def _err():
            yield "event: error\n"
            yield "data: {\"error\": \"model_id and prompt required\"}\n\n"
        return app.response_class(_err(), mimetype="text/event-stream")
    model_dir = MODELS_DIR / model_id.replace("/", "__")
    if not model_dir.exists():
        def _err2():
            yield "event: error\n"
            yield "data: {\"error\": \"model_not_found\"}\n\n"
        return app.response_class(_err2(), mimetype="text/event-stream")
    def _gen():
        import time, queue, threading
        try:
            yield "event: start\n"
            yield "data: {}\n\n"
            t0 = time.time()
            if str(config.get("perf_mode", "")).upper() == "AUTO":
                config["perf_mode"] = _choose_perf_mode(config, device)
            
            try:
                pipe = load_pipeline(model_dir, device, config)
            except Exception as e:
                msg = str(e)
                if "bad allocation" in msg or "Memory" in msg:
                    msg = "内存/显存不足，无法加载模型。请尝试关闭其他应用或使用较小的模型。"
                yield "event: error\n"
                yield "data: " + json.dumps({"error": "load_failed", "message": msg}) + "\n\n"
                return

            cur_dev = getattr(pipe, "_af_device", device)
            cur_real = getattr(pipe, "_af_device_real", cur_dev)
            q = queue.Queue()
            buf = []
            done = {"v": False}
            first = {"t": None}
            gate = {"open": False, "buf": ""}

            # Detect if model is likely a thinking model (e.g. DeepSeek R1)
            # If not, open the gate immediately to avoid hiding output
            mid_lower = str(model_id).lower()
            is_thinking = "deepseek" in mid_lower and ("r1" in mid_lower or "reasoner" in mid_lower)
            if not is_thinking:
                gate["open"] = True

            def sink(x):
                try:
                    xs = str(x)
                    if not gate["open"]:
                        gate["buf"] += xs
                        s = gate["buf"].lower()
                        k = s.find("</think>")
                        if k != -1:
                            gate["open"] = True
                            post = gate["buf"][k+8:]
                            gate["buf"] = ""
                            if first["t"] is None:
                                first["t"] = time.time()
                            if post:
                                q.put(post)
                    else:
                        if first["t"] is None:
                            first["t"] = time.time()
                        q.put(xs)
                except Exception:
                    pass
            def streamer(subword):
                try:
                    sink(subword)
                except Exception:
                    pass
                import openvino_genai as ov_genai
                return ov_genai.StreamingStatus.RUNNING
            out = {"text": None, "metrics": None, "error": None}
            sources = None
            if config.get("web_search"):
                try:
                    from backend.services.inference import web_search, augment_with_sources
                    qtext = config.get("search_query") or prompt
                    sources = web_search(qtext, max_results=5)
                    try:
                        yield "event: sources\n"
                        yield "data: " + json.dumps({"sources": sources}) + "\n\n"
                    except Exception:
                        pass
                    prompt_aug = augment_with_sources(prompt, sources, lang="zh")
                except Exception:
                    prompt_aug = prompt
            else:
                prompt_aug = prompt
            def run_gen():
                try:
                    from backend.services.inference import generate_stream
                    text, metrics = generate_stream(pipe, prompt_aug, config, streamer)
                    out["text"] = text
                    out["metrics"] = metrics
                except Exception as e:
                    out["text"] = ""
                    out["metrics"] = None
                    out["error"] = str(e)
                finally:
                    done["v"] = True
                    try:
                        q.put(None)
                    except Exception:
                        pass
            th = threading.Thread(target=run_gen, daemon=True)
            th.start()
            key = cur_dev if cur_dev in PERF["lat"] else ("NPU" if "NPU" in cur_dev else ("GPU" if "GPU" in cur_dev else ("CPU" if "CPU" in cur_dev else None)))
            while True:
                try:
                    item = q.get(timeout=0.2)
                except Exception:
                    item = None
                if item is None:
                    if done["v"]:
                        break
                    else:
                        continue
                try:
                    buf.append(item)
                    yield "event: token\n"
                    yield "data: " + json.dumps({"text": item}) + "\n\n"
                except Exception:
                    pass
            
            if out["error"]:
                msg = out["error"]
                if "bad allocation" in msg or "Memory" in msg:
                    msg = "生成过程中内存不足。请尝试缩短输入或使用更小的模型。"
                yield "event: error\n"
                yield "data: " + json.dumps({"error": "generation_failed", "message": msg}) + "\n\n"
                return

            dt = int((time.time() - t0) * 1000)
            arr = PERF["lat"].get(key)
            if arr is not None:
                arr.append(dt)
                if len(arr) > 30:
                    del arr[:len(arr)-30]
            metrics = out["metrics"]
            if first["t"] is not None and key:
                try:
                    ttft_ms = float((first["t"] - t0) * 1000.0)
                    PERF["ttft"][key].append(ttft_ms)
                    if len(PERF["ttft"][key]) > 30:
                        del PERF["ttft"][key][:len(PERF["ttft"][key])-30]
                except Exception:
                    pass
            fb = False
            try:
                sd = str(cur_dev)
                sr = str(cur_real)
                fb = (sd.startswith("HETERO") and (not sr.startswith("HETERO"))) and (sd != sr)
            except Exception:
                fb = False
            PERF["last"] = {"device": cur_dev, "real_device": cur_real, "fallback": fb, "latency_ms": dt, "metrics": metrics}
            if metrics and key:
                try:
                    if metrics.get("tpot_ms"):
                        PERF["tpot"][key].append(metrics["tpot_ms"]) 
                    if metrics.get("throughput_tps"):
                        PERF["throughput"][key].append(metrics["throughput_tps"]) 
                    if metrics.get("generate_ms"):
                        PERF["gen"][key].append(metrics["generate_ms"]) 
                    for k in ("tpot","throughput","gen"):
                        if len(PERF[k][key]) > 30:
                            del PERF[k][key][:len(PERF[k][key])-30]
                except Exception:
                    pass
            
            cpu_avg = sum(PERF["lat"]["CPU"]) / len(PERF["lat"]["CPU"]) if PERF["lat"]["CPU"] else None
            npu_avg = sum(PERF["lat"]["NPU"]) / len(PERF["lat"]["NPU"]) if PERF["lat"]["NPU"] else None
            PERF["warn"] = None
            if (device == "NPU" or ("NPU" in device)) and cpu_avg is not None and npu_avg is not None and npu_avg > cpu_avg * 1.2:
                PERF["warn"] = "npu_slower_than_cpu"
            
            s = out["text"] or "" 
            try:
                ss = str(s)
                pp = ss.lower().find("</think>")
                if pp != -1:
                    s = ss[pp+8:].strip()
            except Exception:
                pass
            yield "event: final\n"
            yield "data: " + json.dumps({"text": s, "metrics": metrics}) + "\n\n"
        except Exception as e:
            msg = str(e)
            if "bad allocation" in msg or "Memory" in msg:
                msg = "系统内存不足，无法完成生成。"
            yield "event: error\n"
            yield "data: " + json.dumps({"error": "internal_error", "message": msg}) + "\n\n"
    return app.response_class(_gen(), mimetype="text/event-stream")
@app.get("/api/perf")
def api_perf():
    def avg(a):
        return float(sum(a)/len(a)) if a else None
    usage = {}
    try:
        import psutil
        usage["cpu_percent"] = float(psutil.cpu_percent(interval=0))
    except Exception:
        usage["cpu_percent"] = None
    try:
        from openvino.runtime import Core
        c = Core()
        try:
            tot = c.get_property("NPU", "DEVICE_TOTAL_MEM_SIZE")
            alloc = c.get_property("NPU", "DEVICE_ALLOC_MEM_SIZE")
            usage["npu_mem_total"] = int(tot) if isinstance(tot, (int, float)) else None
            usage["npu_mem_used"] = int(alloc) if isinstance(alloc, (int, float)) else None
        except Exception:
            pass
        try:
            gtot = c.get_property("GPU", "DEVICE_TOTAL_MEM_SIZE")
            gallon = c.get_property("GPU", "DEVICE_ALLOC_MEM_SIZE")
            usage["gpu_mem_total"] = int(gtot) if isinstance(gtot, (int, float)) else None
            usage["gpu_mem_used"] = int(gallon) if isinstance(gallon, (int, float)) else None
        except Exception:
            pass
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], stderr=subprocess.STDOUT, shell=True, text=True)
        rows = []
        for ln in out.strip().splitlines():
            xs = [x.strip() for x in ln.split(",")]
            if len(xs) >= 3:
                rows.append({"util": xs[0], "mem_used": xs[1], "mem_total": xs[2]})
        usage["nvidia"] = rows
    except Exception:
        pass
    # derive hetero participation from memory usage percentages
    hp = {}
    try:
        if isinstance(usage.get("gpu_mem_total"), int) and isinstance(usage.get("gpu_mem_used"), int) and usage["gpu_mem_total"] > 0:
            hp["GPU"] = float(usage["gpu_mem_used"]) / float(usage["gpu_mem_total"]) * 100.0
    except Exception:
        pass
    try:
        if isinstance(usage.get("npu_mem_total"), int) and isinstance(usage.get("npu_mem_used"), int) and usage["npu_mem_total"] > 0:
            hp["NPU"] = float(usage["npu_mem_used"]) / float(usage["npu_mem_total"]) * 100.0
    except Exception:
        pass
    return jsonify({
        "avg": {k: (int(avg(v)) if avg(v) is not None else None) for k, v in PERF["lat"].items()},
        "avg_details": {
            "ttft": {k: avg(v) for k, v in PERF["ttft"].items()},
            "tpot": {k: avg(v) for k, v in PERF["tpot"].items()},
            "throughput": {k: avg(v) for k, v in PERF["throughput"].items()},
            "generate": {k: avg(v) for k, v in PERF["gen"].items()},
        },
        "last": PERF["last"],
        "warn": PERF["warn"],
        "usage": usage,
        "hetero_participation": hp
    })

@app.post("/api/system/clear_cache")
def api_system_clear_cache():
    try:
        base = _get_cache_dir()
        ov = base / "ov_cache"
        try:
            import shutil as _sh
            if ov.exists():
                _sh.rmtree(ov)
        except Exception:
            pass
        try:
            ov.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            for k in ("CPU","GPU","NPU","NVIDIA"):
                PERF["lat"][k].clear()
                PERF["ttft"][k].clear()
                PERF["tpot"][k].clear()
                PERF["throughput"][k].clear()
                PERF["gen"][k].clear()
        except Exception:
            pass
        PERF["last"] = {}
        PERF["warn"] = None
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

def run():
    try:
        _preload_on_start()
    except Exception:
        pass
    app.run(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    run()
