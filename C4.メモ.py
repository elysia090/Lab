import re
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# テキストの前処理
def preprocess_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = re.sub("@\w+", '', text) # '@'で始まる文字列を削除
    text = re.sub("'\d+", '', text) # 数字を削除
    text = re.sub("\d+", '', text)  # 数字を削除
    text = re.sub("http\w+", '', text)  # URLを削除
    text = re.sub(r"\s+", " ", text)  # 連続した空白を1つの空白に置換
    text = re.sub(r"\.+", ".", text)  # 連続したピリオドを1つに置換
    text = re.sub(r"\,+", ",", text)  # 連続したカンマを1つに置換
    text = text.strip()  # 先頭と末尾の空白を削除
    return text

# HTMLタグの削除
def remove_html_tags(text):
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

# TfidfVectorizerのパラメータ設定
vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer='word',
            ngram_range=(1,4),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,
)

# TfidfVectorizerを使って特徴量を生成し、データに統合
def generate_and_merge_tfidf_features(data, vectorizer):
    train_tfid = vectorizer.fit_transform([preprocess_text(text) for text in data['full_text']])
    dense_matrix = train_tfid.toarray()
    df = pd.DataFrame(dense_matrix, columns=[f'tfid_{i}' for i in range(len(dense_matrix[0]))])
    df['essay_id'] = data['essay_id']
    return data.merge(df, on='essay_id', how='left')

# 特徴量の生成
def generate_features(data):
    data = preprocess_paragraphs(data)
    data = preprocess_sentences(data)
    data = preprocess_words(data)
    data = generate_and_merge_tfidf_features(data, vectorizer)
    return data

# 学習および評価
def train_and_evaluate(X_train, y_train, X_test):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1_scores = []
    kappa_scores = []
    models = []

    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
        print('fold', i)
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        model = lgb.LGBMRegressor(
                    objective=qwk_obj,
                    metrics='None',
                    learning_rate=0.1,
                    max_depth=5,
                    num_leaves=10,
                    colsample_bytree=0.5,
                    reg_alpha=0.1,
                    reg_lambda=0.8,
                    n_estimators=1024,
                    random_state=42,
                    extra_trees=True,
                    class_weight='balanced',
                    verbosity=-1
                )
        
        predictor = model.fit(
                        X_train_fold,
                        y_train_fold,
                        eval_names=['train', 'valid'],
                        eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                        eval_metric=quadratic_weighted_kappa,
                        callbacks=[log_evaluation(period=25), early_stopping(stopping_rounds=75, first_metric_only=True)]
                    )
        
        models.append(predictor)
        
        predictions_fold = predictor.predict(X_test_fold)
        predictions_fold = (predictions_fold + a).clip(1, 6).round()
        
        f1_fold = f1_score(y_test_fold, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)
        
        kappa_fold = cohen_kappa_score(y_test_fold, predictions_fold, weights='quadratic')
        kappa_scores.append(kappa_fold)
        
        cm = confusion_matrix(y_test_fold, predictions_fold, labels=[x for x in range(1, 7)])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[x for x in range(1, 7)])
        disp.plot()
        plt.show()
        
        print(f'F1 score across fold: {f1_fold}')
        print(f'Cohen kappa score across fold: {kappa_fold}')

    mean_f1_score = np.mean(f1_scores)
    mean_kappa_score = np.mean(kappa_scores)
    
    print(f'Mean F1 score across 5 folds: {mean_f1_score}')
    print(f'Mean Cohen kappa score across 5 folds: {mean_kappa_score}')
    
    return models

# quadratic weighted kappaの目的関数
def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-a)**2+b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))
    return grad, hess

# データ読み込みと特徴量生成
train_data = pl.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv")
test_data = pl.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")
train_feats = generate_features(train_data)
test_feats = generate_features(test_data)
feature_names = [col for col in train_feats.columns if col not in ['essay_id', 'score']]

# モデルの学習および評価
X_train = train_feats[feature_names].astype(np.float32).values
y_train = train_feats['score'].astype(np.float32).values - a
models = train_and_evaluate(X_train, y_train)

# テストデータに対する予測
X_test = test_feats[feature_names].astype(np.float32).values
probabilities = [model.predict(X_test) + a for model in models]
predictions = np.mean(probabilities, axis=0)
predictions = np.round(predictions.clip(1, 6))

# 提出用ファイルの作成
submission = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
submission['score'] = predictions.astype(int)
submission.to_csv("submission.csv", index=None)
display(submission.head())

