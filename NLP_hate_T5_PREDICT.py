from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import torch
import re
from tqdm import tqdm

def clean_pred(text):
    # 合并多余空格，只留一个
    text = re.sub(r' +', ' ', text)
    # 规范 [SEP] 和 [END] 前的空格为一个
    text = re.sub(r'\s*\[SEP\]', ' [SEP]', text)
    text = re.sub(r'\s*\[END\]', ' [END]', text)
    # 保证结尾有[END]
    if not text.strip().endswith('[END]'):
        text = text.strip() + ' [END]'
    return text.strip()

# 加载模型和分词器
tokenizer = T5Tokenizer.from_pretrained('./t5-hate')
model = T5ForConditionalGeneration.from_pretrained('./t5-hate').to('cuda')
model.eval()

# 加载测试数据
with open('data/test1.json', encoding='utf-8') as f:
    test_data = json.load(f)

batch_size = 8  # 根据显存大小可调

# 批量预测
with open('result.txt', 'w', encoding='utf-8') as fout:
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]
        texts = [item['content'] for item in batch]
        ids = [item.get('id', i + idx) for idx, item in enumerate(batch)]

        # Tokenize
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                           max_length=128).to('cuda')

        # 推理
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=4,  # 与训练一致
                early_stopping=True
            )

        # 解码结果
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 写入文件
        for id_, content, pred in zip(ids, texts, decoded):
            pred = clean_pred(pred)
            fout.write(json.dumps({
                'id': id_,
                'content': content,
                'prediction': pred
            }, ensure_ascii=False) + '\n')

print("预测完成，已保存至 result.txt")
