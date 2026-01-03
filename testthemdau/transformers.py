from transformers import pipeline

text = """San pham dung kha tot, chat luong on.
Thiet ke dep nhung gia hoi cao.
Su dung on dinh nhung pin nhanh het.
Giao hang nhanh nhung dong goi so sai.
Chat lieu ben nhung mau sac khong dep.
Gia re nhung hieu nang khong cao."""

restorer = pipeline("text2text-generation", model="vinai/bartpho-syllable")
result = restorer(text, max_length=512)[0]['generated_text']
print(result)