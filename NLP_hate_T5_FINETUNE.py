from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
import torch
import json
from torch.utils.data import Dataset

# 1. 加载数据
with open('data/train.json', encoding='utf-8') as f:
    train_data = json.load(f)

inputs = [x['content'] for x in train_data]
targets = [x['output'] for x in train_data]

# 2. 加载Tokenizer和Model
tokenizer = T5Tokenizer.from_pretrained('Langboat/mengzi-t5-base')
model = T5ForConditionalGeneration.from_pretrained('Langboat/mengzi-t5-base')

# 3. 定义Dataset类
class T5Dataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=128):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.inputs[idx]
        target = self.targets[idx]

        source_enc = self.tokenizer(source, padding='max_length', truncation=True,
                                    max_length=self.max_length, return_tensors='pt')
        target_enc = self.tokenizer(target, padding='max_length', truncation=True,
                                    max_length=self.max_length, return_tensors='pt')

        # 忽略label中的padding token
        labels = target_enc['input_ids'].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_enc['input_ids'].squeeze(),
            'attention_mask': source_enc['attention_mask'].squeeze(),
            'labels': labels
        }

# 4. 构建Dataset
train_dataset = T5Dataset(inputs, targets, tokenizer, max_length=128)

# 5. 自动数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir='./t5-hate',
    num_train_epochs=6,
    per_device_train_batch_size=8,
    save_strategy='epoch',
    evaluation_strategy='no',
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=5e-5,
    lr_scheduler_type='linear',
    fp16=True,
    save_total_limit=2,
    predict_with_generate=True,  # 支持生成时自动解码
    generation_max_length=128,
    report_to='none'
)

# 7. 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8. 开始训练并保存模型
trainer.train()
model.save_pretrained('./t5-hate')
tokenizer.save_pretrained('./t5-hate')
