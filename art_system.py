import os
# 开启 PyTorch 底层兼容模式，允许不支持的 MPS 算子安全回退至 CPU 执行
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from PIL import Image, ImageFilter
import numpy as np
import math
# 引入 HuggingFace 相关的模型与预处理器库
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler

class ArtRestorationSystem:
    """
    Art Restoration & Analysis System (Dynamic Pipeline Edition).
    
    模块涵盖：
    1. 图像高清修复 (Swin2SR)
    2. 视觉语义理解 (BLIP)
    3. 跨风格全局重绘 (SD Img2Img)
    4. 基于文本掩膜的局部重绘 (CLIPSeg + SD Inpainting)
    """
    
    def __init__(self):
        print("==================================================")
        print("[SYSTEM] Initializing AI Art System on Apple Silicon")
        print("==================================================\n")

        # 1. 硬件计算设备配置与分配
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")          # 主算力分配给 Apple Silicon 的 MPS 核心 (GPU)
            self.cpu_device = torch.device("cpu")      # 辅助算力分配给 CPU，用于处理容易溢出的不稳定算子
            print("[INFO] Compute Device Configured: Apple MPS (GPU)")
        else:
            self.device = torch.device("cpu")          # 全局降级至 CPU 运行
            self.cpu_device = torch.device("cpu")
            print("[WARN] MPS not detected. Falling back to CPU.")
            
        # 2. 加载超分辨率增强模型
        print("[INFO] [1/5] Loading Super-Resolution Model (Swin2SR-x4)...")
        sr_model_id = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
        self.sr_processor = Swin2SRImageProcessor.from_pretrained(sr_model_id) # 实例化图像特征提取器
        self.sr_model = Swin2SRForImageSuperResolution.from_pretrained(sr_model_id).to(self.device) # 加载模型权重并推入主设备显存
        
        # 3. 加载视觉语言多模态模型
        print("[INFO] [2/5] Loading Visual-Language Model (BLIP)...")
        vlm_model_id = "Salesforce/blip-image-captioning-large"
        self.vlm_processor = BlipProcessor.from_pretrained(vlm_model_id) # 实例化图文多模态特征处理器
        self.vlm_model = BlipForConditionalGeneration.from_pretrained(vlm_model_id).to(self.device) # 加载文本生成模型并推入显存

        # 4. 加载文本驱动的图像分割模型
        print("[INFO] [3/5] Loading Text-Segmentation Model (CLIPSeg)...")
        clipseg_model_id = "CIDAS/clipseg-rd64-refined"
        self.seg_processor = CLIPSegProcessor.from_pretrained(clipseg_model_id)
        self.seg_model = CLIPSegForImageSegmentation.from_pretrained(clipseg_model_id).to(self.device) # 轻量级模型，锁定在 CPU 确保精度

        # 5. 加载全局图生图扩散模型 (全面升级为 SDXL 1.0)
        print("[INFO] [4/5] Loading Img2Img Pipeline (SDXL 1.0)...")
        self.sd_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, # 必须用半精度，否则容易爆内存
            variant="fp16",            # 直接下载半精度精简版权重，节省极大的硬盘空间和下载时间
            use_safetensors=True
        ).to(self.device)

        # 6. 加载专用于局部重绘的 9通道 UNet 模型
        print("[INFO] [5/5] Configuring Shared Inpainting Pipeline...")
        self.inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True
        ).to(self.device)

        print("[INFO] Enabling VAE slicing to optimize Apple Silicon memory...")
        self.sd_pipeline.enable_vae_slicing()

        # ==========================================
        # 挂载 DPM++ 2M Karras 高阶调度器
        # ==========================================
        print("[INFO] Injecting DPM++ 2M Karras Scheduler for high-quality fast inference...")
        
        # 启用 Karras 噪声分布，这是目前能让 SDXL 在 20 步内达到极高画质的唯一真神
        dpm_scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipeline.scheduler.config,
            use_karras_sigmas=True
        )
        self.sd_pipeline.scheduler = dpm_scheduler
        self.inpaint_pipeline.scheduler = dpm_scheduler
        
        print("\n[SUCCESS] Initialization complete. All models ready.\n")

    def _get_optimal_sdxl_size(self, image: Image.Image) -> tuple:
        """基于 SDXL 黄金像素总数 (~1 Megapixel) 动态计算最佳宽高比"""
        w, h = image.size
        target_area = 896 * 896  # SDXL 的标准百万像素靶心
        
        # 计算原图的宽高比例
        aspect_ratio = w / h
        
        # 按等面积公式推导目标长宽
        new_w = math.sqrt(target_area * aspect_ratio)
        new_h = math.sqrt(target_area / aspect_ratio)
        
        # 这是 SDXL 潜空间 (Latent Space) 的底层硬性要求
        new_w = max(64, round(new_w / 64) * 64)
        new_h = max(64, round(new_h / 64) * 64)
        
        return (int(new_w), int(new_h))

    def _generate_text_mask(self, image: Image.Image, text_prompt: str, target_size=(1024, 1024)) -> Image.Image:
        """内部辅助方法：根据自然语言指令生成二值化目标掩膜 (Mask)"""
        print(f"[PROCESS] Segmenting masking region for: '{text_prompt}'...")
        # 将原始图像与自然语言转化为模型特征张量，并送入 CPU 设备
        inputs = self.seg_processor(text=[text_prompt], images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad(): # 阻断梯度反向传播以节省内存
            outputs = self.seg_model(**inputs) # 执行前向推理，获取分割热力图
        
        preds = outputs.logits.unsqueeze(1) # 提取预测 Logits 并扩展通道维度
        preds = torch.nn.functional.interpolate(
            preds, size=image.size[::-1], mode="bilinear", align_corners=False # 利用双线性插值将特征图还原至原始图像尺寸
        )
        
        mask_tensor = torch.sigmoid(preds[0][0]) > 0.5 # 应用 Sigmoid 激活并通过 0.5 阈值进行二值化切分
        mask_pil = Image.fromarray(mask_tensor.byte().cpu().numpy() * 255, mode="L") # 转换为单通道 (L模式) 的 8 位图像对象
        mask_resized = mask_pil.resize(target_size, Image.LANCZOS)
        return mask_resized.filter(ImageFilter.GaussianBlur(radius=16)) # 强制尺寸缩放至默认 512x512，并采用最近邻插值确保边缘黑白分明

    # ==========================================
    # 原子功能模块 (Atomic Modules)
    # ==========================================
    def restore_and_enhance(self, image_path: str, output_path: str) -> str:
        """模块 1：执行图像画质清洗与 4x 超分辨放大"""
        print(f"[PROCESS] Enhancing image quality: {image_path}")
        raw_image = Image.open(image_path).convert("RGB")
        
        w, h = raw_image.size
        # 即使不用官方 processor，底层 Swin2SR 依然要求输入必须是 8 的倍数
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        raw_image = raw_image.resize((new_w, new_h))
        
        # 【终极修复】：彻底抛弃 HuggingFace 官方的 Processor！它内部有隐蔽的强制裁切 1:1 Bug！
        # 手动将 PIL 图像转化为神经网络需要的 Tensor (B, C, H, W)，并归一化到 0~1
        img_array = np.array(raw_image).astype(np.float32) / 255.0
        pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 直接投喂纯净的 Tensor，再也没有任何中间商能压扁你的图了
            outputs = self.sr_model(pixel_values=pixel_values) 
            
        output_tensor = outputs.reconstruction.cpu().squeeze(0).clamp(0, 1) 
        output_array = (output_tensor.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8) 
        Image.fromarray(output_array).save(output_path) 
        print(f"[SUCCESS] Enhanced image saved to: {output_path}")
        return output_path

    def analyze_and_describe(self, image_path: str) -> str:
        """模块 2：执行视觉内容到自然语言的语义映射"""
        print(f"[PROCESS] Analyzing image semantics...")
        image = Image.open(image_path).convert("RGB")
        inputs = self.vlm_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            ids = self.vlm_model.generate(**inputs, max_new_tokens=50) # 触发自回归生成机制，控制最大产出长度为 50 Token
            
        desc = self.vlm_processor.batch_decode(ids, skip_special_tokens=True)[0] # 批量解码 Token 序列，剥离控制字符提取纯净文本
        print(f"[RESULT] Extracted semantics: '{desc}'")
        return desc

    def style_transfer(
            self, 
            image_path: str, 
            prompt: str, 
            output_path: str, 
            guidance_scale: float = 8.5, 
            strength: float = 0.6,
            step_callback=None
        ) -> str:
        """模块 3：执行全局扩散生图与跨风格迁移"""
        print(f"[PROCESS] Executing global style transfer...")
        print(f"[INFO] Applied Prompt: '{prompt}' | Strength: {strength}")
        
        init_image_raw = Image.open(image_path).convert("RGB")
        target_size = self._get_optimal_sdxl_size(init_image_raw)
        init_image = init_image_raw.resize(target_size)
        generator = torch.Generator(device=self.device).manual_seed(42)

        num_steps = 20
        actual_steps = max(1, int(num_steps * strength))

        def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
            if step_callback is not None:
                step_callback(step_index + 1, actual_steps)
            return callback_kwargs
        
        with torch.no_grad():
            image = self.sd_pipeline(
                prompt=prompt, 
                image=init_image, 
                strength=strength, 
                generator=generator,
                num_inference_steps=20, 
                guidance_scale=guidance_scale,
                width=target_size[0],
                height=target_size[1],
                callback_on_step_end=diffusers_callback
            ).images[0]
            
        image.save(output_path)
        print(f"[SUCCESS] Stylized image saved to: {output_path}")
        return output_path

    def auto_inpaint(
            self, 
            image_path: str, 
            mask_target_text: str, 
            addition_prompt: str, 
            output_path: str, 
            base_desc: str = "", 
            guidance_scale: float = 8.5, 
            strength: float = 0.9,
            step_callback=None
        ) -> str:
        """模块 4：执行基于语义掩膜的局部内容创造性注入"""
        print(f"[PROCESS] Executing targeted auto-inpainting...")
        print(f"[INFO] Auto-Masking: '{mask_target_text}' | Painting: '{addition_prompt}'")
        
        init_image_raw = Image.open(image_path).convert("RGB")
        target_size = self._get_optimal_sdxl_size(init_image_raw)
        init_image_dynamic = init_image_raw.resize(target_size)
        init_image_1024 = init_image_raw.resize((1024, 1024)) # 准备基础画布张量
        
        # 动态请求 CLIPSeg 模块输出目标区域的二值化隔离区
        mask_image = self._generate_text_mask(init_image_raw, mask_target_text, target_size=target_size)
        
        # context_prompt = f"in the exact style of ({base_desc})" if base_desc else "matching original art style"
        style_anchor = "perfectly matching the original image's artistic style, color palette, lighting, and aesthetic"
        full_prompt = f"{addition_prompt}, {style_anchor}, seamless integration, masterpiece" 
        
        num_steps = 20
        actual_steps = max(1, int(num_steps * strength))

        def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
            if step_callback is not None:
                # 告诉前端：当前是第几步，总共几步
                step_callback(step_index + 1, actual_steps)
            return callback_kwargs # 必须返回 kwargs，这是底层的死规矩

        # 恢复通用的、治本的负面提示词，专注打击质量问题和风格冲突
        neg_prompt = "ugly, deformed, artifacts, poorly drawn, out of context, mismatching style, photorealistic (if original is art)"
        generator = torch.Generator(device="cpu").manual_seed(42)

        with torch.no_grad():
            result = self.inpaint_pipeline(
                prompt=full_prompt,
                negative_prompt=neg_prompt,
                image=init_image_dynamic,  # 传入基础图像
                mask_image=mask_image, # 传入黑白掩膜 (黑区锁定不变，白区重新计算生成)
                num_inference_steps=20, # 提升去噪步数以获取更精密的局部结构
                guidance_scale=guidance_scale,    # 拉高对文本提示的遵循权重
                strength=strength,          # 提高局部重绘的重构强度，彻底覆盖旧区域
                generator=generator,
                width=target_size[0],
                height=target_size[1],
                callback_on_step_end=diffusers_callback
            )
            
        result.images[0].save(output_path)
        print(f"[SUCCESS] Inpainted image saved to: {output_path}")
        return output_path

    # ==========================================
    # 动态流水线接口 (Dynamic Pipeline)
    # ==========================================
    def run_dynamic_pipeline(self, input_image: str, options: dict, output_dir: str = "outputs") -> str:
        """
        核心调度层：基于外部传入的参数字典，动态构建并串联处理图 (DAG)。
        系统将自动追踪上一节点的输出指针，并安全交接给下一节点作为输入源。
        """
        print("\n" + "="*50)
        print("[START] Executing Dynamic Art Pipeline")
        print("="*50)
        
        os.makedirs(output_dir, exist_ok=True) # 初始化文件系统的输出沙箱
        current_image_path = input_image # 设置流水线的活动数据流指针
        base_name = os.path.basename(input_image) # 剥离路径提取原始文件名，用于日志和派生文件命名
        
        # 节点 A: 修复增强层拦截与处理
        if options.get("enhance", False):
            print("\n>>> [Pipeline Step] Quality Enhancement")
            enhanced_path = os.path.join(output_dir, f"1_enhanced_{base_name}")
            current_image_path = self.restore_and_enhance(current_image_path, enhanced_path) # 指针覆写为高清图像

        # 节点 B: 语义抽取层拦截与处理
        base_desc = ""
        if options.get("analyze", False):
            print("\n>>> [Pipeline Step] Semantic Analysis")
            base_desc = self.analyze_and_describe(current_image_path) # 捕获当前图像的上下文映射

        # 节点 C: 局部重绘层拦截与处理
        inpaint_config = options.get("inpaint", None)
        if inpaint_config:
            print("\n>>> [Pipeline Step] Auto-Inpainting")
            inpainted_path = os.path.join(output_dir, f"2_inpainted_{base_name}")
            current_image_path = self.auto_inpaint(
                image_path=current_image_path,
                mask_target_text=inpaint_config["mask_target"], # 提取掩膜探测词
                addition_prompt=inpaint_config["addition_prompt"], # 提取新元素生成指令
                output_path=inpainted_path,
                base_desc=base_desc
            ) # 指针覆写为局部合成图像

        # 节点 D: 全局风格滤镜层拦截与处理
        style_config = options.get("style_transfer", None)
        if style_config:
            print("\n>>> [Pipeline Step] Global Style Transfer")
            target_style = style_config.get("target_style", "masterpiece")
            strength = style_config.get("strength", 0.6)
            
            # 语义融合算法：拼接原图客观骨架描述与目标主观风格特征，实现高保真度重绘
            final_prompt = f"{base_desc}, {target_style}" if base_desc else target_style
            stylized_path = os.path.join(output_dir, f"3_stylized_{base_name}")
            current_image_path = self.style_transfer(
                image_path=current_image_path,
                prompt=final_prompt,
                output_path=stylized_path,
                strength=strength
            ) # 指针覆写为最终艺术图像

        print("\n" + "="*50)
        print(f"[FINISHED] Pipeline complete. Final Masterpiece: {current_image_path}") # 播报生命周期结束，释放最终资产句柄
        print("="*50)
        return current_image_path


# ==========================================
# 本地测试与执行主入口
# ==========================================
if __name__ == "__main__":
    art_sys = ArtRestorationSystem() # 拉起常驻服务实例
    test_image = "inputs/test.jpg"   # 绑定测试物料
    
    if os.path.exists(test_image): # 执行 I/O 探活
        
        # 场景定义 1：全链路高阶渲染测试 (修复 -> 解析 -> 局部注物 -> 全局滤镜)
        config_full = {
            "enhance": True,
            "analyze": True,
            "inpaint": {
                "mask_target": "sky",
                "addition_prompt": "add an airplane flying in the sky"
            },
            "style_transfer": {
                "target_style": "in the style of Vincent van Gogh, thick brushstrokes",
                "strength": 0.3 # 限制破坏权重以保护前置步骤注入的细微结构
            }
        }
        
        # 场景定义 2：极简增量渲染测试 (修复 -> 局部注物 -> 无损输出)
        config_simple = {
            "enhance": True,
            "analyze": False,
            "inpaint": {
                "mask_target": "grass",
                "addition_prompt": "a cute golden retriever running"
            },
            "style_transfer": None # 绕过渲染层级，强制硬切保留原始肌理
        }
        
        # 执行调度：下发配置负载并挂起主线程直至管线全生命周期回调
        art_sys.run_dynamic_pipeline(input_image=test_image, options=config_full)
        
    else:
        print(f"[ERROR] Required input file not found: {test_image}") # 抛出运行时文件失联异常