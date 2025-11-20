<div align="center">

<h1>AI Funland 🎡🤖</h1>

Local AI Q&A platform powered by OpenVINO, optimized for Intel Ultra NPU. | 基于 OpenVINO 的本地 AI 问答平台，针对 Intel Ultra 系列 NPU 优化。

<p>
📦 Version: <b>V0.0.2 Dev</b> · 🗓️ Release Date: <b>2025-11-20</b>
</p>

</div>

## Highlights / 亮点

- ⚡ 一键“下载 + INT8量化”，开箱即用
- ⬇️ ModelScope 下载：实时进度（不定进度→精确百分比），友好文案替换“API”
- 🧩 INT8/INT4 权重量化，量化完成自动清理原始模型，仅保留量化版本；Tokenizer IR 自动编译
- 🖥️ 加速器选择：CPU · Intel GPU · Intel Ultra NPU · NVIDIA · 协同选项（NPU+GPU、NPU+GPU+CPU）
- 🎯 默认优先 Intel NPU+GPU 协同；首次加载自动预热，启用编译缓存（`OV_CACHE_DIR`），缩短 TTFT
- 🧠 Intel Ultra NPU 优化：`OV_PERFORMANCE_HINT=LATENCY`、可调并行度 `streams`，低延迟优先
- 🛡️ 友好错误提示：不兼容加速器与模型占用删除，前端弹窗与“释放模型”按钮
- 💬 现代化 UI（Material 风格）：消息气泡与历史搜索、Tooltip/配置向导、三档响应式布局
- 📈 性能面板与告警：`/api/perf` 监控延迟；NPU 低于 CPU 自动提示；系统信息“当前加速器”不重复显示
- 🚀 一键启动 `start.bat`，无需 Node.js；项目缓存：`tmp/`（可用 `AIFUNLAND_CACHE_DIR` 定制）
- 🧱 模块化架构，预留扩展接口（文生图/视频等）

## Release Notes · V0.0.2 Dev（2025/11/20）

### 1) 错误提示优化
- 模型加载失败提示优化：当检测到模型不兼容 Intel NPU/GPU 加速时，前端显示友好信息：
  - “当前模型不支持硬件加速功能，请更换兼容模型或使用 CPU 模式”
- 模型删除失败提示优化：当模型被占用时显示：
  - “模型正在使用中，请先停止相关任务后再尝试删除”

### 2) 交互体验改进
- 首次加载等待提示：
  - “系统正在初始化模型，这可能需要 1-2 分钟，请稍候...”
- 实时下载进度：
  - 准确反映后端下载状态，避免仅显示“API”，支持不定进度到精确百分比的切换

### 3) 硬件加速优化
- 加速器选择优先级更新：
  - 1) Intel NPU → 2) Intel GPU → 3) CPU → 4) Nvidia GPU
- 多硬件协同加速模式：
  - 提供 NPU+GPU+CPU 与 NPU+GPU 混合计算选项，提升硬件利用率
- NPU 推理效率优化：
  - 量化参数与并行度配置优化，优先低延迟模式（可调 `Latency/Throughput` 与 `streams`）

### 4) 性能优化
- NPU 推理效率专项优化，缩短响应时间与首帧时间（TTFT）
- 前端界面响应速度优化与动画反馈统一，提升整体体验

版本状态：开发版（Dev）

---

### Previous Releases / 历史版本

<p>
📦 Version: <b>V0.0.1 Dev</b> · 🗓️ Release Date: <b>2025-11-19</b>
</p>

## Quick Start（Windows）

1. 双击 `start.bat`
   - 自动安装/检查嵌入式 Python 与依赖
   - 启动后端并打开浏览器 `http://127.0.0.1:8000/`
2. 在 “Models” 输入框使用默认模型：`qwen/Qwen2.5-0.5B-Instruct`
3. 点击 “Download” 或 “Download+INT8” 一键体验
4. 在 “Chat” 区选择模型，输入问题，点击 “Generate”

Tips：如需自定义缓存目录，设置环境变量 `AIFUNLAND_CACHE_DIR`（默认 `d:\codes\AI Funland\tmp`）。

## Features（English）

- Model selection, chat UI, bilingual interface
- Model management: download (ModelScope), quantize (INT8/INT4), delete
- Hardware info and accelerator selection: CPU, Intel GPU, Intel Ultra NPU, NVIDIA GPU
- Responsive, modern UI; no Node.js required; one-click `start.bat`
- Project-scoped cache (`tmp/`), robust download with retries and API fallback

## 功能（中文）

- 模型选择与对话，中英双语界面
- 模型管理：ModelScope 下载、INT8/INT4 量化、删除
- 系统硬件信息与加速器选择：CPU / Intel GPU / Intel NPU / NVIDIA GPU
- 响应式现代化 UI；无需 Node.js；一键 `start.bat`
- 项目级缓存（`tmp/`）；下载重试与 API 回退，稳健可靠

## Architecture / 架构

- Backend（Python）：
  - `backend/app.py` · 路由与服务集成（系统信息、下载、量化、推理、任务进度）
  - `backend/services/inference.py` · OpenVINO GenAI 管线、量化与 Tokenizer IR 自动编译
  - `backend/services/models.py` · 模型列表/删除；`backend/services/system.py` · 设备检测
  - `backend/utils/tasks.py` · 任务状态存储与轮询
- Frontend（纯静态）：
  - `web/index.html` · 结构与控件
  - `web/styles.css` · 响应式样式与进度动画
  - `web/app.js` · 交互逻辑、i18n、API 调用、下载与量化按钮
- Startup：
  - `start.bat` · 一键启动、pip 检查与缓存路径注入（`AIFUNLAND_CACHE_DIR`）

## Recommended Models / 推荐模型

- `qwen/Qwen2.5-0.5B-Instruct` · 适合 CPU 快速验证；建议先 INT8
- `qwen/Qwen2.5-1.5B-Instruct` · 更优质量；适合 CPU/Intel NPU/Intel GPU/NVIDIA GPU（量化后）
- `qwen/Qwen2.5-3B-Instruct` · 中端显卡/核显可用；注意显存与 IR 分片

## Directories / 目录约定

- `models/` · 下载与量化后模型：`<org__model>`、`<org__model>_quant_int8`
- `tmp/` · ModelScope 缓存（可用 `AIFUNLAND_CACHE_DIR` 自定义）

## Troubleshooting / 常见问题

- ModelScope 下载失败（`exit 1`）：组织名大小写问题，已自动回退与小写重试；必要时使用 API 下载。
- 进度不显示：初期显示不定进度动画，解析到百分比后转为精确进度。
- PyTorch ONNX 符号错误（`_attention_scale`）：已固定 `torch==2.4.1`，兼容 `optimum[openvino]==1.27.0`。
- Tokenizer IR 缺失（`openvino_tokenizer.xml`）：量化后自动编译；推理加载时若缺失也会自动补齐。
- 静态资源 404：已修正静态目录指向 `web/`，首页使用 `send_static_file('index.html')`。

## Roadmap / 未来更新计划

- 0.0.2 · 流式输出、对话历史、系统健康检查（端口/防火墙）
- 0.1.x · 更强的设备自动选择与性能档位；更精细的量化策略（混合精度）
- 0.2.x · 文生图 / 文生视频 / 图生视频 / 视频换脸等扩展模块
- 0.3.x · 插件化架构、预训练模型管理、配置持久化与导入导出
- 0.4.x · 跨平台打包（Windows/macOS/Linux）、离线安装包
- 0.5.x · 测试完善（单测/集成测试）、CI/CD、错误可观测性与日志

## Dependencies / 依赖

```
apiflask==2.4.0
openvino==2025.3.0
openvino-genai==2025.3.0.0
openvino-tokenizers==2025.3.0.0
langchain_community==0.3.29
optimum[openvino]==1.27.0
modelscope==1.12.0
torch==2.4.1
transformers==4.42.4
```

## Credits / 致谢

- OpenVINO · OpenVINO GenAI · Optimum OpenVINO · ModelScope
- Qwen2.5 系列模型（用于功能验证）

---

<div align="center">

❤️ If you find AI Funland useful, please star the repo.

</div>