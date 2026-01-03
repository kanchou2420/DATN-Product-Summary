from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

text = """San pham dung kha tot, chat luong on.
Thiet ke dep nhung gia hoi cao.
Su dung on dinh nhung pin nhanh het.
Giao hang nhanh nhung dong goi so sai.
Chat lieu ben nhung mau sac khong dep.
Gia re nhung hieu nang khong cao."""

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable")

inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=256)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)