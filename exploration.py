
from src.data_loader import load_dataframe

df = load_dataframe("./dataset/deterioro_cognitivo.sav")

print("\nSHAPE")
print(df.shape)

print("\nCOLUMNAS")
print(df.columns.tolist())

print("\nTIPOS")
print(df.dtypes)

print("\nNULOS")
print(df.isnull().sum())

print("\nTARGET")
print(df["GDS_R2"].value_counts())