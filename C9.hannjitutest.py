import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# データの読み込み
data = pd.read_csv("/kaggle/input/bank-csv/bank.csv", sep=";")
# 列名を適切に設定
data.columns = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]

# カテゴリカル変数のエンコーディング
label_encoder = LabelEncoder()
categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# 目的変数の変換
data["y"] = data["y"].map({"yes": 1, "no": 0})

# データの分割
X = data.drop("y", axis=1)
y = data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 前処理後のデータを確認
print(X_train.head())
print(y_train.head())
