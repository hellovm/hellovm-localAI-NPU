# HelloVM-AI-Funland 开发文档
# HelloVM-AI-Funland Development Documentation

## 📋 文档概述 / Documentation Overview

本文档为 HelloVM-AI-Funland 多硬件加速大语言模型交互平台的完整开发指南，涵盖架构设计、开发规范、API 接口、测试策略等各个方面。

This document provides a comprehensive development guide for HelloVM-AI-Funland multi-hardware accelerated large language model interaction platform, covering architecture design, development standards, API interfaces, testing strategies, and more.

## 🎯 项目目标 / Project Objectives

### 核心目标 / Core Objectives

1. **多硬件加速支持 / Multi-Hardware Acceleration Support**
   - 支持 CPU、Intel GPU、Intel NPU、NVIDIA GPU 等多种硬件
   - 实现硬件自动检测与性能优化
   - 提供统一的加速接口抽象层

2. **智能模型管理 / Intelligent Model Management**
   - 基于 Modelscope API 的模型下载与管理
   - 支持 GGUF/GGML 等多种模型格式
   - 实现多线程断点续传与完整性校验

3. **现代化用户界面 / Modern User Interface**
   - Vue 3 + TypeScript 技术栈
   - 国际化支持（中英文）
   - 响应式设计，适配多种设备

4. **插件扩展架构 / Plugin Extension Architecture**
   - 支持文本生成图像、视频等 AI 功能扩展
   - 热加载插件机制
   - 依赖隔离与版本管理

## 🏗️ 架构设计 / Architecture Design

### 系统架构 / System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│                (Vue 3 + TypeScript + Tailwind)             │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway Layer                        │
│                    (FastAPI + WebSocket)                    │
├─────────────────────────────────────────────────────────────┤
│                   Business Logic Layer                      │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Hardware  │   Model     │  Download   │   Plugin    │  │
│  │   Manager   │   Manager   │   Manager   │   Manager   │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  Acceleration Layer                         │
│         (OpenVINO + CUDA + Native Acceleration)           │
├─────────────────────────────────────────────────────────────┤
│                    Runtime Layer                            │
│              (LLM Runtime + Model Formats)                  │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈 / Technology Stack

#### 后端技术栈 / Backend Stack
- **语言 / Language**: Python 3.13+
- **Web 框架 / Web Framework**: FastAPI
- **异步处理 / Async Processing**: asyncio, aiohttp
- **硬件加速 / Hardware Acceleration**: OpenVINO, CUDA, PyTorch
- **模型管理 / Model Management**: Modelscope API
- **数据存储 / Data Storage**: SQLite (本地), PostgreSQL (可选)
- **日志系统 / Logging**: loguru
- **配置管理 / Configuration**: pydantic-settings

#### 前端技术栈 / Frontend Stack
- **框架 / Framework**: Vue 3.3+
- **语言 / Language**: TypeScript 5.0+
- **构建工具 / Build Tool**: Vite 5.0+
- **状态管理 / State Management**: Pinia
- **UI 框架 / UI Framework**: Tailwind CSS
- **图标库 / Icons**: Heroicons
- **国际化 / Internationalization**: vue-i18n
- **图表库 / Charts**: Recharts (可选)

### 模块设计 / Module Design

#### 1. 硬件管理模块 / Hardware Management Module

```python
# 硬件检测接口 / Hardware Detection Interface
class HardwareDetector:
    def detect_cpu(self) -> CPUInfo
    def detect_gpu(self) -> GPUInfo
    def detect_npu(self) -> NPUInfo
    def get_acceleration_capabilities(self) -> AccelerationInfo

# 加速器基类 / Accelerator Base Class
class BaseAccelerator(ABC):
    @abstractmethod
    def is_available(self) -> bool
    @abstractmethod
    def get_performance_metrics(self) -> PerformanceMetrics
    @abstractmethod
    def load_model(self, model_path: str) -> Model
    @abstractmethod
    def infer(self, model: Model, input_data: Any) -> InferenceResult
```

#### 2. 模型管理模块 / Model Management Module

```python
# 模型管理器 / Model Manager
class ModelManager:
    def search_models(self, query: str) -> List[ModelInfo]
    def download_model(self, model_id: str, format: str) -> DownloadTask
    def load_model(self, model_id: str, accelerator: str) -> LoadedModel
    def validate_model(self, model_path: str) -> ValidationResult

# 下载管理器 / Download Manager
class DownloadManager:
    def create_download_task(self, url: str, output_path: str) -> DownloadTask
    def pause_download(self, task_id: str) -> bool
    def resume_download(self, task_id: str) -> bool
    def get_download_progress(self, task_id: str) -> DownloadProgress
```

#### 3. 插件系统模块 / Plugin System Module

```python
# 插件基类 / Plugin Base Class
class BasePlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str
    @property
    @abstractmethod
    def version(self) -> str
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool
    @abstractmethod
    def execute(self, input_data: Any) -> PluginResult

# 插件管理器 / Plugin Manager
class PluginManager:
    def load_plugin(self, plugin_path: str) -> bool
    def unload_plugin(self, plugin_name: str) -> bool
    def get_loaded_plugins(self) -> List[PluginInfo]
    def execute_plugin(self, plugin_name: str, input_data: Any) -> PluginResult
```

## 📋 开发规范 / Development Standards

### 代码规范 / Code Standards

#### Python 代码规范 / Python Code Standards
- **代码风格 / Code Style**: 遵循 PEP 8 规范
- **类型注解 / Type Annotations**: 必须使用类型注解
- **文档字符串 / Docstrings**: 使用 Google 风格文档字符串
- **异常处理 / Exception Handling**: 使用自定义异常类
- **日志记录 / Logging**: 使用结构化的日志记录

```python
from typing import Dict, List, Optional
from loguru import logger

class ModelManager:
    """模型管理器 / Model Manager
    
    负责模型的搜索、下载、加载和管理 / Responsible for model search, download, loading and management
    """
    
    def search_models(self, query: str, limit: int = 10) -> List[ModelInfo]:
        """搜索模型 / Search models
        
        Args:
            query: 搜索关键词 / Search keyword
            limit: 返回结果数量限制 / Result limit
            
        Returns:
            模型信息列表 / List of model information
            
        Raises:
            ModelSearchError: 搜索失败时抛出 / Raised when search fails
        """
        try:
            logger.info(f"Searching models with query: {query}")
            # 实现搜索逻辑 / Implement search logic
            return models
        except Exception as e:
            logger.error(f"Model search failed: {e}")
            raise ModelSearchError(f"Search failed: {e}")
```

#### TypeScript 代码规范 / TypeScript Code Standards
- **代码风格 / Code Style**: 遵循 ESLint 配置
- **组件设计 / Component Design**: 使用 Composition API
- **类型定义 / Type Definitions**: 定义清晰的接口和类型
- **错误处理 / Error Handling**: 使用 try-catch 块处理异步操作

```typescript
// 类型定义 / Type Definitions
export interface ModelInfo {
  id: string
  name: string
  size: number
  format: ModelFormat
  quantization: QuantizationType
  status: ModelStatus
}

// 组件实现 / Component Implementation
export default defineComponent({
  name: 'ModelCard',
  props: {
    model: {
      type: Object as PropType<ModelInfo>,
      required: true
    }
  },
  setup(props) {
    const { t } = useI18n()
    
    const formatFileSize = (bytes: number): string => {
      // 文件大小格式化 / File size formatting
      return formattedSize
    }
    
    return {
      formatFileSize
    }
  }
})
```

### 命名规范 / Naming Conventions

#### Python 命名规范 / Python Naming Conventions
- **类名 / Class Names**: 使用 PascalCase (例如: `ModelManager`)
- **函数名 / Function Names**: 使用 snake_case (例如: `search_models`)
- **常量名 / Constants**: 使用 UPPER_SNAKE_CASE (例如: `MAX_DOWNLOAD_THREADS`)
- **模块名 / Module Names**: 使用 snake_case (例如: `model_manager.py`)

#### TypeScript 命名规范 / TypeScript Naming Conventions
- **类名 / Class Names**: 使用 PascalCase (例如: `DownloadManager`)
- **函数名 / Function Names**: 使用 camelCase (例如: `startDownload`)
- **常量名 / Constants**: 使用 UPPER_SNAKE_CASE (例如: `MAX_CONCURRENT_DOWNLOADS`)
- **接口名 / Interface Names**: 使用 PascalCase (例如: `DownloadTask`)

---

<div align="center">
  <p><strong>HelloVM-AI-Funland 开发文档</strong></p>
  <p>版本 / Version: 1.0.0 | 更新日期 / Last Updated: 2024-11-14</p>
</div>