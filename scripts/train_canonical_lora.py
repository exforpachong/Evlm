#!/usr/bin/env python3
"""
Train Qwen3-VL-2B LoRA on canonical dataset.
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

# Paths
MODEL_DIR = "D:/dataset/models_hf/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
DATA_DIR = "D:/dataset/data"
SAMPLE_DIR = "D:/dataset/sample_images"
OUTPUT_DIR = "D:/dataset/finetune/qwen3vl_canonical_lora"

# Training config
MAX_PIXELS = 262144  # 512x512
MAX_TRAIN_SAMPLES = None  # Use all

def load_dataset(split):
    """Load canonical dataset."""
    path = os.path.join(DATA_DIR, f"dataset_{split}_canonical.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} {split} samples")
    return data

class VLMDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor, sample_dir):
        self.data = data
        self.processor = processor
        self.sample_dir = sample_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get image path
        images = item.get("images", [])
        if not images:
            return None
        image_path = os.path.join(self.sample_dir, Path(images[0]).name)
        
        # Create messages
        messages = item.get("conversations", [])
        
        # Format for training
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Process image
        image_inputs, _ = process_vision_info([{"role": "user", "content": [{"type": "image", "image": image_path}]}])
        
        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
        }

def main():
    print("="*60)
    print("Training Qwen3-VL-2B LoRA on Canonical Dataset")
    print("="*60)
    
    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    processor.max_pixels = MAX_PIXELS
    
    # Load data
    print("\nLoading datasets...")
    train_data = load_dataset("train")
    eval_data = load_dataset("val")
    
    if MAX_TRAIN_SAMPLES:
        train_data = train_data[:MAX_TRAIN_SAMPLES]
        print(f"Using {len(train_data)} training samples")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = VLMDataset(train_data, processor, SAMPLE_DIR)
    eval_dataset = VLMDataset(eval_data, processor, SAMPLE_DIR)
    
    # Load model
    print("\nLoading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save adapter
    print("\nSaving adapter...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    
    print("\nTraining complete!")
    print(f"Adapter saved to: {OUTPUT_DIR}/final_adapter")

if __name__ == "__main__":
    main()
