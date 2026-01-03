from transformers import AutoTokenizer, AutoModel

text = """San pham dung kha tot, chat luong on.
Thiet ke dep nhung gia hoi cao.
Su dung on dinh nhung pin nhanh het.
Giao hang nhanh nhung dong goi so sai.
Chat lieu ben nhung mau sac khong dep.
Gia re nhung hieu nang khong cao."""

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

tokens = tokenizer(text, return_tensors="pt")
outputs = model(**tokens)
result = tokenizer.decode(tokens['input_ids'][0])
print(result)