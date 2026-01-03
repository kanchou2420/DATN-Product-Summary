from pyvncorenlp import VnCoreNLP

text = """San pham dung kha tot, chat luong on.
Thiet ke dep nhung gia hoi cao.
Su dung on dinh nhung pin nhanh het.
Giao hang nhanh nhung dong goi so sai.
Chat lieu ben nhung mau sac khong dep.
Gia re nhung hieu nang khong cao."""

rdrsegmenter = VnCoreNLP(address="http://127.0.0.1", port=9000)
result = rdrsegmenter.annotate(text)
print(result)