import pandas as pd

# 加载 parquet 文件
train_df = pd.read_parquet('~/data/geo3k/train.parquet')
import pdb
pdb.set_trace()
test_df = pd.read_parquet('~/data/geo3k/test.parquet')

# 打印行列信息
print("Train parquet:")
print(train_df.info())
print(train_df.head(1))  # 打印一行样例

print("\nTest parquet:")
print(test_df.info())
print(test_df.head(1))  # 打印一行样例
