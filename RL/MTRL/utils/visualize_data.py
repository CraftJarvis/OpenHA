import pandas as pd

df = pd.read_parquet('~/data/verl-agent/text/train.parquet')
print(df.head())  # 查看前几行数据
