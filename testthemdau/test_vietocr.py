from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

text = """San pham dung kha tot, chat luong on.
Thiet ke dep nhung gia hoi cao.
Su dung on dinh nhung pin nhanh het.
Giao hang nhanh nhung dong goi so sai.
Chat lieu ben nhung mau sac khong dep.
Gia re nhung hieu nang khong cao."""

config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu'
detector = Predictor(config)
print(text)