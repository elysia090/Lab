import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import dowhy
from dowhy import CausalModel

# データの読み込み
data = pd.read_csv("/kaggle/input/bank-csv/bank.csv", sep=";")
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

# DoWhyのための因果モデルの作成
model = CausalModel(
    data=data,
    treatment="campaign",  # マーケティングキャンペーンの実施を治療として扱う
    outcome="y",  # 定期預金口座の開設を結果として扱う
    common_causes=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "pdays", "previous", "poutcome"]
)

# 因果推論を実行して因果効果を推定
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression", test_significance=True)

# 結果の表示
print(estimate)
