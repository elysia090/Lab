# プロジェクト概要

このリポジトリは、金融モデリング、機械学習、コンパイラ最適化、ゼロ知識証明など、多岐にわたる研究用スクリプトをまとめたものです。従来はルートディレクトリに多数のファイルが混在していましたが、用途に合わせて整理した高品質なディレクトリ構成に刷新しました。

## ディレクトリ構成

```
src/
├── finance/
│   ├── analytics/
│   ├── corporate_finance/
│   ├── econometrics/
│   └── market_simulation/
├── machine_learning/
│   ├── classical/
│   ├── deep_learning/
│   │   ├── experiments/
│   │   ├── generative/
│   │   ├── graph/
│   │   ├── quantization/
│   │   └── vision/
│   ├── nlp/
│   ├── state_space/
│   └── time_series/
├── attention/
├── compiler_and_systems/
│   ├── gpu_acceleration/
│   ├── llvm_ptx/
│   ├── native/
│   └── simulation/
├── cryptography/
│   └── plonk/
├── quantum/
└── utilities/
    └── fractals/
```

各ディレクトリには `__init__.py` を配置して Python パッケージとして利用できるようにしています。主な分類は以下の通りです。

- **finance/**: 市場シミュレーション、経済計量モデル、企業価値評価、指標分析など金融分野のスクリプト。
- **machine_learning/**: 古典的な機械学習、自然言語処理、カルマンフィルタや時系列分析、各種ディープラーニング実験をカテゴリ別に格納。
- **attention/**: 高速注意機構や SERA/SLUM など注意系研究の実装。
- **compiler_and_systems/**: LLVM→PTX 変換、PyCUDA 実験、仮想 CPU、ネイティブ資産 (C++/Rust/PowerShell) を含むシステムレベルのツール群。
- **cryptography/plonk/**: SymPLONK 系のゼロ知識証明実装と検証ユーティリティ。
- **quantum/**: 量子ゲートシミュレータなどの量子計算関連コード。
- **utilities/**: クリフォード代数メモ、ラーチアルゴリズム、ネットワーク解析、フラクタル生成などの汎用ツール。

## テスト

```
pytest
```

現在は `finance.market_simulation.market_model` を対象とした単体テストを提供しています。必要に応じて `src` ディレクトリを `PYTHONPATH` に追加して実行してください。

## 変更履歴

- ルート直下に散在していたスクリプトをカテゴリ別のサブディレクトリに移動。
- すべてのコードを Python パッケージ構造に整備し、再利用性を向上。
- `tests/test_market_model.py` を新しいパス構成に合わせて更新。
