"""
Train ablation LoRA adapters with canonical data
Usage: python train_ablation_canonical.py --scale [25|50|100]
"""
import json
import torch
import argparse
from pathlib import Path
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info


ROOT_DIR = Path(__file__).resolve().parent.parent
MAX_PIXELS = 262144

class AblationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, images_dir):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.images_dir = Path(images_dir)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get image path
        img_name = sample['images'][0]
        if '\\' in img_name:
            img_name = img_name.split('\\')[-1]
        img_path = str(self.images_dir / img_name)
        
        # Build messages in OpenAI format for Qwen processor.
        conversations = sample['conversations']
        messages = []
        for conv in conversations:
            source_role = conv.get("from")
            if source_role in {"human", "user"}:
                role = "user"
            elif source_role in {"gpt", "assistant"}:
                role = "assistant"
            else:
                raise ValueError(f"Unsupported conversation role: {source_role}")

            content = conv["value"]
            if role == "user" and "<image>" in content:
                messages.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": content.replace("<image>", "").strip()}
                    ]
                })
            else:
                messages.append({"role": role, "content": content})

        return {"messages": messages}


class DataCollatorForAblation:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = []
        images = []
        for item in batch:
            messages = item["messages"]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            image_inputs, _ = process_vision_info(messages)
            if image_inputs:
                images.extend(image_inputs)

        inputs = self.processor(
            text=texts,
            images=images if images else None,
            padding=True,
            return_tensors="pt",
            max_pixels=MAX_PIXELS,
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, required=True, choices=[25, 50, 100])
    parser.add_argument('--model_path', default=str(ROOT_DIR / "models" / "Qwen3-VL-2B-Instruct"))
    parser.add_argument('--data_dir', default=str(ROOT_DIR / "data"))
    parser.add_argument('--images_dir', default=str(ROOT_DIR / "sample_images"))
    parser.add_argument('--output_root', default=str(ROOT_DIR / "ablation_adapters"))
    parser.add_argument('--epochs', type=float, default=3.0)
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(args.data_dir)
    base_model = args.model_path
    train_data = data_dir / f"dataset_train_canonical_{args.scale}pct.json"
    val_data = data_dir / "dataset_val_canonical_clean.json"
    images_dir = args.images_dir
    output_dir = Path(args.output_root) / f"scale_{args.scale}pct"
    
    print(f'Training scale {args.scale}% with canonical data')
    print(f'Train data: {train_data}')
    print(f'Output: {output_dir}')
    
    # Load model
    print('Loading model...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, max_pixels=MAX_PIXELS)
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print('Loading data...')
    train_dataset = AblationDataset(train_data, images_dir)
    val_dataset = AblationDataset(val_data, images_dir)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForAblation(processor),
    )
    
    # Train
    print('Starting training...')
    trainer.train()
    
    # Save
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f'Saved adapter to {output_dir}')

if __name__ == '__main__':
    main()
