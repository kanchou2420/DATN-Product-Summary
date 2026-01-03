import stanza

text = """San pham dung kha tot, chat luong on.
Thiet ke dep nhung gia hoi cao.
Su dung on dinh nhung pin nhanh het.
Giao hang nhanh nhung dong goi so sai.
Chat lieu ben nhung mau sac khong dep.
Gia re nhung hieu nang khong cao."""

stanza.download('vi')
nlp = stanza.Pipeline('vi', processors='tokenize')
doc = nlp(text)
result = ' '.join([token.text for sent in doc.sentences for token in sent.tokens])
print(result)