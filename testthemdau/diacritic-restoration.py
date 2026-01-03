from diacritic_restoration import DiacriticRestoration

text = """San pham dung kha tot, chat luong on.
Thiet ke dep nhung gia hoi cao.
Su dung on dinh nhung pin nhanh het.
Giao hang nhanh nhung dong goi so sai.
Chat lieu ben nhung mau sac khong dep.
Gia re nhung hieu nang khong cao."""

restorer = DiacriticRestoration()
result = restorer.restore(text)
print(result)