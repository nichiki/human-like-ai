# Pythonコーディング規約

## ドキュメント

- すべてのファイル、クラス、関数、メソッドにdocstringを記述する
- Googleスタイルのdocstring形式を使用する
- ファイルの目的、関数の引数、戻り値、例外を明記する

## コード品質

- すべての関数とメソッドに型アノテーションを使用する
- インポートは標準ライブラリ、サードパーティ、ローカルの順に整理する
- 命名規則を厳守する（クラス名はUpperCamelCase、関数・変数名はsnake_case）
- コードフォーマットはRuffとmypyの設定に従う
- 行の最大長は88文字、インデントは4スペース

## プログラミングパラダイム

- 可能な限り関数型アプローチを採用する（純粋関数、不変データ構造）
- クラスを使用する場合は単一責任の原則を守る
- 継承よりもコンポジションを優先する

詳細についは@clinerules-bank/languages/python 以下を参照すること