import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

# 导入视觉底层系统
from art_system import ArtRestorationSystem 

class ArtAgentBrain:
    """
    智能体大脑 (LLM Agent Brain)
    负责理解用户的自然语言指令，并将其编译为底层视觉系统所需的 JSON 配置表。
    """
    def __init__(self):
        print("==================================================")
        print("[AGENT] 正在唤醒本地大语言模型 (LLM Brain)...")
        print("==================================================\n")
        
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        # 选用 Qwen2.5-1.5B-Instruct
        model_id = "qwen/Qwen2.5-7B-Instruct" 
        
        print(f"[DOWNLOADING] 正在从阿里 ModelScope 极速拉取模型: {model_id} ...")
        # 让 ModelScope 把模型下载到本地，并返回绝对路径
        local_model_dir = snapshot_download(model_id)
        
        # 让 Transformers 直接从本地路径加载，不再去连国外的 Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_dir, 
            torch_dtype=torch.float16,
        ).to(self.device)
        
        print("\n[SUCCESS] 语言大脑已就绪！\n")

    def parse_instruction(self, user_prompt: str, temperature: float = 0.3, image_context: str = "") -> dict:
        """核心方法：将自然语言转化为 JSON 配置"""
        print(f"[THINKING] AI 正在思考您的指令: '{user_prompt}'")
        
        # 极具技术含量的 System Prompt 设计 (Prompt Engineering)
        system_prompt = f"""你是一个智能艺术处理中枢。
        【当前画面的视觉描述 (非常重要)】：{image_context if image_context else "未知场景"}""" + """
        
        请仔细阅读用户的自然语言指令，结合上方提供的【画面视觉描述】，将其翻译为一个严格的 JSON 配置文件。
        绝不要输出任何解释性文字、分析或多余的标点符号。

        【绝对致命规则】：
        1. 所有的值必须翻译为纯英文！
        2. 局部重绘与替换 (inpaint)：如果指令中包含“添加”、“换成”、“替换为”等明确的修改要求，`inpaint` 绝对不能为 null！你必须：
           - `addition_prompt`: 填入最终想要生成的新物体（纯英文）。
           - `mask_target`: 填入要被遮盖或替换的区域（纯英文）。
             【逻辑分支 A - 添加】：如果是“添加”（如“加一辆车”），根据常识推断背景填入 `mask_target`（如 "road", "sky"）。
             【逻辑分支 B - 替换】：如果是“替换”（如“把狗换成猫”），`mask_target` 必须是被替换的原物体本身（如 "dog"）！绝不能是背景！
              极其重要：在替换场景下，`addition_prompt` 绝对不能只写一个孤立的名词（如 "peach"）！你必须结合原图语境，强行补充详细的【空间位置描述】和【遮挡关系】，以对抗绘图模型的常识脑补！例如：必须写成 "a large peach floating in front of the face, covering the face completely"。
        3. 如果用户没有明确要求改变画风，`style_transfer` 必须严格设为 null！绝不允许擅自编造艺术家！
        4. 如果指令中明确要求了**特定的数量**（如“三个太阳”），必须在英文中极力强调它，如 "exactly three distinct suns"。
        
        可用的 JSON 格式和字段定义如下：
        {
            "enhance": true 或 false,  
            "analyze": true 或 false,  
            "inpaint": {               
                "mask_target": "被替换的背景或被替换的原物体(纯英文)",
                "addition_prompt": "要添加或替换成的新物体(纯英文)"
            } 或 null,
            "style_transfer": {        
                "target_style": "目标艺术风格(纯英文)",
                "strength": 0.6        
            } 或 null
        }"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 组装对话模板并推入设备计算
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200, temperature=temperature) # 极低的温度保证 JSON 输出的严谨性
            
        # 截取大模型新生成的回复部分
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 使用正则表达式强行提取 JSON 块（防御性编程，防止 LLM 话多）
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                config_dict = json.loads(match.group(0))
                print("\n[AGENT] 思考完毕，生成的调度配置如下：")
                print(json.dumps(config_dict, indent=4, ensure_ascii=False))
                return config_dict
            else:
                raise ValueError("未找到合法的 JSON 结构")
        except Exception as e:
            print(f"[ERROR] LLM 编译 JSON 失败，返回安全默认配置。错误信息: {e}")
            print(f"原始大模型输出：{response_text}")
            # 兜底的安全配置
            return {"enhance": True, "analyze": False, "inpaint": None, "style_transfer": None}

# ==========================================
# 终极路演测试入口
# ==========================================
if __name__ == "__main__":
    # 下载 LLM 权重)
    agent_brain = ArtAgentBrain()
    
    # 加载 Stable Diffusion 等视觉系统
    vision_system = ArtRestorationSystem()
    
    test_image = "inputs/test.jpg"
    if os.path.exists(test_image):
        
        # 输入指令
        boss_instruction = "帮我把这张画高清修复一下，再在天空上添加一个太阳，最后整体套一个卡通效果。"
        
        # 将自然语言翻译成机器配置
        dynamic_config = agent_brain.parse_instruction(boss_instruction)
        
        # 底层视觉引擎根据配置干活
        vision_system.run_dynamic_pipeline(input_image=test_image, options=dynamic_config)
        
    else:
        print(f"[ERROR] 找不到测试图片: {test_image}")