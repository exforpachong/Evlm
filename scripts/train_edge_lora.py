#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
边缘设备训练脚本：Qwen3-VL-2B-Instruct LoRA 微调
适配 8GB 显存的边缘设备
- 限制图像分辨率 (max_pixels=262144, 约512x512)
- 使用 gradient checkpointing
- 小 batch size + 梯度累积
"""

import json
import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import transformers
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info
from PIL import Image

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "finetune_output_v2"

# 边缘设备配置
MAX_PIXELS = 262144  # 512x512，适合 8GB 显存


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=str(ROOT_DIR / "models" / "Qwen3-VL-2B-Instruct"),
        metadata={"help": "Model path"}
    )
    use_peft: bool = field(default=True, metadata={"help": "Use PEFT/LoRA"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    train_file: str = field(
        default=str(DATA_DIR / "dataset_train.json"),
        metadata={"help": "Training data file"}
    )
    val_file: str = field(
        default=str(DATA_DIR / "dataset_val.json"),
        metadata={"help": "Validation data file"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max training samples (for debug)"}
    )


class DisasterDataset(torch.utils.data.Dataset):
    """灾害数据集 - 适配边缘设备"""
    
    def __init__(self, data_file: str, processor, max_samples: int = None, max_pixels: int = MAX_PIXELS):
        self.processor = processor
        self.root_dir = Path(data_file).parent.parent
        self.max_pixels = max_pixels
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        self.data = data
        print(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        
        # 获取图片路径
        image_path = self.root_dir / record['images'][0]
        
        # 获取对话
        conversations = record.get('conversations', [])
        user_msg = ""
        assistant_msg = ""
        
        for conv in conversations:
            if conv.get('from') == 'user':
                user_msg = conv.get('value', '').replace('<image>\n', '').replace('<image>', '')
            elif conv.get('from') == 'assistant':
                assistant_msg = conv.get('value', '')
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path), "resized_height": 512, "resized_width": 512},
                    {"type": "text", "text": user_msg}
                ]
            },
            {
                "role": "assistant",
                "content": assistant_msg
            }
        ]
        
        return {"messages": messages}


class DataCollatorForVLM:
    """VLM 数据整理器"""
    
    def __init__(self, processor, max_pixels: int = MAX_PIXELS):
        self.processor = processor
        self.max_pixels = max_pixels
    
    def __call__(self, batch):
        texts = []
        images = []
        
        for item in batch:
            messages = item["messages"]
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            
            # 处理图像（带分辨率限制）
            image_inputs, _ = process_vision_info(messages)
            if image_inputs:
                images.extend(image_inputs)
        
        # 处理输入
        inputs = self.processor(
            text=texts,
            images=images if images else None,
            padding=True,
            return_tensors="pt",
            max_pixels=self.max_pixels
        )
        
        # 创建标签
        labels = inputs["input_ids"].clone()
        inputs["labels"] = labels
        
        return inputs


def main():
    print("="*60)
    print("Qwen3-VL-2B-Instruct LoRA Training for Edge Device")
    print("="*60)
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Max Pixels: {MAX_PIXELS} (512x512)")
    
    # 参数
    model_path = ROOT_DIR / "models" / "Qwen3-VL-2B-Instruct"
    output_dir = OUTPUT_DIR / "qwen3vl_2b_lora_edge"
    
    print(f"\nModel path: {model_path}")
    print(f"Output dir: {output_dir}")
    
    # 加载模型
    print("\nLoading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        max_pixels=MAX_PIXELS
    )
    
    # 配置 LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 启用 gradient checkpointing
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")
    
    # 加载数据
    print("\nLoading data...")
    train_dataset = DisasterDataset(
        str(DATA_DIR / "dataset_train.json"),
        processor,
        max_pixels=MAX_PIXELS
    )
    val_dataset = DisasterDataset(
        str(DATA_DIR / "dataset_val.json"),
        processor,
        max_pixels=MAX_PIXELS
    )
    
    # 数据整理器
    data_collator = DataCollatorForVLM(processor, max_pixels=MAX_PIXELS)
    
    # 训练参数 - 适配边缘设备
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # 有效 batch size = 16
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        max_grad_norm=1.0,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\nStarting training...")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    
    trainer.train()
    
    # 保存模型
    print("\nSaving model...")
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))
    
    # 保存训练配置
    config = {
        "model_name": "Qwen3-VL-2B-Instruct",
        "training_method": "LoRA",
        "lora_r": 16,
        "lora_alpha": 32,
        "max_pixels": MAX_PIXELS,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "timestamp": datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(0)
    }
    
    with open(output_dir / "training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Model saved to: {output_dir / 'final_adapter'}")
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
