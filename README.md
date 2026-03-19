# Agentic AI Art Studio (智能视觉资产渲染中枢)

这是一个基于 **Agentic Workflow（智能体工作流）** 的多模态 AI 图像处理系统。通过引入大语言模型（LLM）作为“中枢大脑”，系统能够精准理解用户的自然语言指令（如“把天空中的云换成飞鸟，并加上梵高油画风格”），自动编译调度底层的多套前沿计算机视觉模型（CV），实现一站式的图像语义分割、局部重绘、全局风格迁移与超分增强。

本项目深度适配 **Apple Silicon (M系列芯片，如 M4 Pro)**，充分压榨 MPS (Metal Performance Shaders) 统一内存算力，实现全链路本地极速推理。

---

## 核心特性 (Key Features)

* **LLM 智能体调度中心 (Agentic Brain)**
    * 内置 **Qwen 2.5 (7B-Instruct)** 作为逻辑大脑，具备强大的视觉上下文理解和空间推理能力。
    * 支持自然语言直接下发极其复杂的“增删改查”图像指令，自动拒绝“幻觉”，精准输出 JSON 调度工作流。
* **视觉前置感知与语义切割 (Vision Perception)**
    * **BLIP**: 提前对原始上传图像进行全景视觉扫描，提取环境语义，打破大模型的“视觉盲区”。
    * **CLIPSeg**: 基于文本的零样本图像分割，告别手动涂抹掩膜，实现像素级的物体与背景精准分离。
* **百万像素级无痕重绘 (Seamless Inpainting)**
    * 挂载 **SDXL 1.0 Inpainting** 专用九通道模型，结合 DPM++ 2M Karras 高阶调度器，在 20 步内榨干潜空间画质。
    * 独创 **“百万像素等面积动态靶心”** 算法，完美支持 `16:9`、`9:16` 等任意画幅，彻底解决重绘比例畸变与画风割裂问题。
    * 自动应用掩膜羽化（Gaussian Blur）与环境继承（Style Anchoring），实现新老元素的完美光影交融。
* **4K 工业级画质增强 (Super Resolution)**
    * 集成 **Swin2SR** 现实世界图像超分大模型，作为流水线最后一道质检关卡，一键输出 4K 级超清巨幅海报。
* **极度优雅的丝滑交互 (Streamlit UI)**
    * 提供“宏观阶段 + 微观步数”双轨动态进度播报。
    * 支持细粒度的高级参数调节（LLM Temperature, CFG Scale, 重绘 Strength）。

---

## 系统架构 (Architecture)

系统由前端交互层、LLM 编译层和 CV 渲染管线三部分组成：

1.  **用户指令** -> `Streamlit Front-End`
2.  **图像预审** -> `BLIP` (提取场景描述)
3.  **编译引擎** -> `Qwen 2.5 7B` (结合场景描述与用户指令，生成包含掩膜目标、替换词的 JSON DAG)
4.  **渲染管线** -> `ArtRestorationSystem`:
    * *节点 A: 图像超分* (`Swin2SR`)
    * *节点 B: 掩膜切割* (`CLIPSeg`)
    * *节点 C: 局部重绘* (`SDXL Inpainting`)
    * *节点 D: 风格迁移* (`SDXL Base`)

---

## 安装与运行 (Installation & Usage)

### 环境依赖
请确保你的机器拥有 24GB 及以上的统一内存 (推荐 M3/M4 Pro 及以上机型) 或同等显存的 Nvidia GPU。

```bash
# 1. 克隆项目并进入目录
git clone https://github.com/yourusername/Agentic-AI-Art-Studio.git
cd Agentic-AI-Art-Studio

# 2. 创建 conda 虚拟环境
conda create -n ai_art python=3.11 -y
conda activate ai_art

# 3. 安装核心依赖包 (PyTorch MPS/CUDA 均兼容)
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate peft
pip install streamlit modelscope pillow numpy
```

### 启动服务
```bash
# 启动 Streamlit 本地服务
streamlit run app.py
```
首次运行时，系统会自动从 ModelScope 和 Hugging Face 拉取所需的 LLM 和 CV 模型权重，请保持网络畅通（已内置国内镜像源优化）。

---

## 提示词魔法 (Prompt Examples)

在左侧的“Agent 指令”框中，你可以尝试以下咒语来激发系统的潜力：

* **常规局部添加**："帮我高清修复一下，在画面里的路上加一辆跑车。"
* **定向逻辑替换**："把脸上的苹果换成桃子。" *(注意：进行替换操作时，建议展开高级参数，将「局部重绘强度」拉至 0.95 以上)*
* **全局风格覆写**："将这张风景照转换成赛博朋克风格，霓虹灯闪烁。"
* **全链路终极压榨**："帮我把这张画高清修复，在天上加一个巨大的飞碟，最后整体套一个新海诚动漫风格。"

---

## 目录结构 (Directory Structure)
```text
.
├── app.py                  # Streamlit 前端交互主入口
├── agent_brain.py          # LLM 调度大脑，负责自然语言解析与 JSON 路由
├── art_system.py           # 底层多模态视觉管线，封装所有模型前向推理
├── inputs/                 # 默认的测试物料输入目录
└── outputs/                # 各阶段流水线的渲染产出目录
```
