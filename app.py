import os
import json
import streamlit as st
from PIL import Image

# ==========================================
# 页面基础配置与 CSS 视觉优化
# ==========================================
st.set_page_config(page_title="AI 艺术资产中枢", page_icon="🎨", layout="wide")

st.markdown("""
<style>
    h1 { font-size: 1.8rem !important; margin-bottom: 0rem; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.1rem !important; }
    h4 { font-size: 1.0rem !important; color: #4A90E2;}
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="正在唤醒 M4 Pro 统一内存与百亿参数模型...")
def load_ai_systems():
    from art_system import ArtRestorationSystem
    from agent_brain import ArtAgentBrain
    return ArtRestorationSystem(), ArtAgentBrain()

vision_system, agent_brain = load_ai_systems()

# ==========================================
# 左侧控制区：紧凑排版与多模态参数矩阵
# ==========================================
with st.sidebar:
    st.markdown("### 📥 上传图片")
    uploaded_file = st.file_uploader("请选择图片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    st.markdown("### 🗣️ 下达指令")
    user_instruction = st.text_area(
        label="Agent 指令",
        label_visibility="collapsed",
        placeholder="例如：帮我高清修复，在画面里加一个骑车的人...",
        height=100
    )
    start_button = st.button("🚀 启动智能体工作流", type="primary", width='stretch')
    with st.expander("⚙️ 展开高级核心参数", expanded=False):
        st.markdown("**🧠 LLM 逻辑层**")
        llm_temp = st.slider("大脑发散度 (Temperature)", 0.1, 1.0, 0.3, 0.1, help="越大越有想象力，越小越严谨")
        
        st.markdown("**👁️ Vision 视觉层**")
        vision_cfg = st.slider("提示词贴合度 (CFG Scale)", 4.0, 15.0, 8.5, 0.5, help="越小AI发挥越自由，越大越死板贴合文字")
        vision_strength = st.slider("局部重绘强度 (Strength)", 0.5, 1.0, 0.85, 0.05, help="越大对原图背景的覆盖越彻底")
    
    st.divider()
    json_container = st.empty() # 预留放 JSON 的空位

# ==========================================
# 右侧主工作区：完美对齐的画廊
# ==========================================
st.markdown("<h1>🎨 AI 视觉资产渲染看板</h1>", unsafe_allow_html=True)
# 更新了底下的架构说明，匹配你现在强大的 7B 和 SDXL 模型
st.markdown("<span style='color:gray; font-size:0.9rem;'>底层架构：Qwen 2.5 7B + SDXL + BLIP + Swin2SR | 算力支撑：Apple M4 Pro</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 提前划分好两列，确保标题永远在同一高度完美对齐
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 🖼️ 原始画作")
    img_in_placeholder = st.empty() 

with col2:
    st.markdown("#### ✨ 渲染产物")
    img_out_placeholder = st.empty() 

if not uploaded_file:
    img_in_placeholder.info("👈 请在左侧控制台上传图片。")
    img_out_placeholder.info("等待 Agent 调度...")
else:
    # 修正了 width='stretch' 为 width='stretch'
    img_in_placeholder.image(uploaded_file, width='stretch')
    
    if not start_button:
        img_out_placeholder.info("等待 Agent 调度...")
    else:
        if not user_instruction:
            st.error("请输入指令！")
        else:
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            input_path = os.path.join("inputs", "temp_input.jpg")
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # ======= 【架构升级：视觉感知前置】 =======
            current_image = input_path
            base_name = "temp_input.jpg"
            
            # 1. 第一步：先让 BLIP 看图提取特征！
            progress_bar = st.progress(10, text="👁️ BLIP 正在进行全景视觉扫描...")
            base_desc = vision_system.analyze_and_describe(current_image)
            
            # 2. 第二步：将老板指令和视觉特征一起喂给 7B 大脑
            progress_bar.progress(25, text="🤖 大脑正在结合视觉上下文编译 JSON...")
            try:
                config_dict = agent_brain.parse_instruction(
                    user_prompt=user_instruction, 
                    temperature=llm_temp,
                    image_context=base_desc # 【核心注入点】让大脑不再是瞎子！
                )
                with json_container.container():
                    st.markdown("### 🧠 实时调度配置 (JSON)")
                    st.json(config_dict)
            except Exception as e:
                progress_bar.empty()
                st.error(f"解析异常: {e}")
                st.stop()

            # ======= 将 Pipeline 逻辑继续执行 =======
            # 1. 优先执行局部重绘
            if config_dict.get("inpaint"):
                target = config_dict['inpaint']['mask_target']
                progress_bar.progress(50, text=f"🖌️ 正在分离 '{target}' 掩膜并注入新元素...")
                sub_progress = st.empty()
                def update_render_progress(step, total):
                    # 计算百分比 (0.0 ~ 1.0)，并防止极其罕见的溢出报错
                    pct = min(step / total, 1.0)
                    # 实时刷新小进度条
                    sub_progress.progress(pct, text=f"SDXL 重绘渲染中: 第 {step} / {total} 步")
                current_image = vision_system.auto_inpaint(
                    image_path=current_image,
                    mask_target_text=target,
                    addition_prompt=config_dict['inpaint']['addition_prompt'],
                    output_path=os.path.join(output_dir, f"1_inpainted_{base_name}"),
                    base_desc=base_desc, 
                    guidance_scale=vision_cfg,
                    strength=vision_strength,
                    step_callback=update_render_progress
                )
                sub_progress.empty()
                
            # 2. 然后执行全局风格统一
            if config_dict.get("style_transfer"):
                style = config_dict['style_transfer']['target_style']
                progress_bar.progress(70, text=f"🎨 正在应用全局渲染风格: {style}...")
                sub_progress = st.empty()
                def update_render_progress(step, total):
                    # 计算百分比 (0.0 ~ 1.0)，并防止极其罕见的溢出报错
                    pct = min(step / total, 1.0)
                    # 实时刷新小进度条
                    sub_progress.progress(pct, text=f"全局风格渲染中: 第 {step} / {total} 步")

                final_prompt = f"{base_desc}, {style}" if base_desc else style
                current_image = vision_system.style_transfer(
                    image_path=current_image,
                    prompt=final_prompt,
                    output_path=os.path.join(output_dir, f"2_stylized_{base_name}"),
                    guidance_scale=vision_cfg,
                    strength=config_dict['style_transfer'].get("strength", 0.6),
                    step_callback=update_render_progress
                )
                sub_progress.empty()

            # 3. 【核心修复】：最后一步执行 4x 超分！直接输出 4K 巨幅画作！
            if config_dict.get("enhance"):
                progress_bar.progress(90, text="🔍 Swin2SR 正在执行终极 4x 超分画质增强...")
                current_image = vision_system.restore_and_enhance(
                    current_image, 
                    os.path.join(output_dir, f"3_final_enhanced_{base_name}")
                )
            progress_bar.progress(100, text="✨ 渲染完毕！")
            
            img_out_placeholder.image(current_image, width='stretch')
            st.toast('处理完毕！', icon='✅')