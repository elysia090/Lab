# アヤメのデータ分析
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Irisデータセットの読み込み
iris = load_iris()

# データフレームの作成
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

# 無限値をNaNに変換
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# サンプルデータの先頭を表示
print(df.head())

# データの基本統計量を表示
print(df.describe())

# データの欠損値の確認
print(df.isnull().sum())

# データの可視化
with pd.option_context('mode.use_inf_as_na', True):
    sns.pairplot(df, hue="target")
plt.show()

# データの分割
df.dropna(inplace=True)  # NaNを含む行を削除
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの構築と学習
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# モデルの評価
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
