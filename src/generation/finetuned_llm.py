# =====================================================================
# ⚠️ 注意：此模块为云端 GPU 环境专用的可选模块。
# 本地开发环境（仅 CPU 或无独立显卡）请勿直接导入或运行此文件。
# 依赖库：transformers, peft, torch, accelerate (详见 requirements-gpu.txt)
# =====================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

class FinetunedLLM:
    """LoRA 微调后的模型封装，用于生产环境推理"""
    def __init__(self, base_model_path: str, lora_weights_path: str):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.base_model, lora_weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)