# 技術コンテキスト

## 技術スタック

### 言語とランタイム
- **Python 3.12**: 最新の型ヒントと機能を活用

### 主要ライブラリ
- **LangChain**: LLMアプリケーション構築フレームワーク
  - langchain-core: 基本コンポーネント
  - langchain-community: コミュニティコネクタ
  - langchain-openai: OpenAI連携
- **FAISS**: 高速ベクトル検索ライブラリ
- **PyYAML**: YAML処理
- **python-dotenv**: 環境変数管理

### AI/LLM
- **OpenAI API**: GPT-4oモデルを使用
- **OpenAI Embeddings**: テキスト埋め込み

## 開発環境

### 環境標準化
- **devcontainer**: 開発環境の標準化
  - Python 3.12
  - uv パッケージマネージャー

### コード品質ツール
- **Ruff**: リンティングとフォーマット
  - 行の長さ: 88文字
  - 引用符スタイル: シングルクォート
  - インデント: 4スペース
- **mypy**: 静的型チェック
  - 厳格な型チェック設定

### テスト
- **pytest**: テストフレームワーク
- **pytest-cov**: カバレッジ測定

## 外部依存

### API キー
- **OpenAI API キー**: `.env`ファイルで管理

### 環境変数
```
OPENAI_API_KEY=your-api-key-here
```

## 開発プラクティス

### コーディング規約
- **Google スタイルの docstring**
- **厳格な型アノテーション**
- **関数型アプローチの推奨**
- **単一責任の原則**

### バージョン管理
- **Git**: ソースコード管理
- **GitHub**: リポジトリホスティング

### パッケージ管理
- **uv**: 高速な依存関係管理
