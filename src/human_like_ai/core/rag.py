"""
RAG(Retrieval-Augmented Generation)モジュール。

このモジュールは、キャラクター設定などのドキュメントから関連情報を検索するための
RAGシステムを提供します。
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.human_like_ai.config.settings import Settings, get_settings


class RAGService(ABC):
    """RAGサービスの抽象基底クラス。

    RAGサービスのインターフェースを定義します。
    """

    @abstractmethod
    def initialize(self, documents: list[str]) -> None:
        """RAGシステムを初期化します。

        Args:
            documents: 初期化に使用するドキュメントのリスト
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """クエリに関連するドキュメントを取得します。

        Args:
            query: 検索クエリ
            k: 取得するドキュメントの数

        Returns:
            List[Dict[str, Any]]: 関連ドキュメントのリスト
        """
        pass


class FAISSRAGService(RAGService):
    """FAISS を使用した RAG サービス実装。

    FAISS ベクトルデータベースを使用して、効率的な検索を提供します。

    Attributes:
        settings: アプリケーション設定
        text_splitter: テキスト分割器
        embeddings: 埋め込みモデル
        vector_store: ベクトルストア
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """初期化メソッド。

        Args:
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。
        """
        self.settings = settings or get_settings()
        self.text_splitter = CharacterTextSplitter(
            separator='\n\n',  # 段落ごとに分割
            chunk_size=500,  # おおよそ500文字
            chunk_overlap=50,  # オーバーラップをもたせる
        )
        self.embeddings = OpenAIEmbeddings()
        self.vector_store: FAISS | None = None

    def initialize(self, documents: list[str]) -> None:
        """RAGシステムを初期化します。

        Args:
            documents: 初期化に使用するドキュメントのリスト
        """
        # ドキュメントをチャンクに分割
        chunks = []
        for doc in documents:
            chunks.extend(self.text_splitter.split_text(doc))

        # ベクトルストアを作成
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """クエリに関連するドキュメントを取得します。

        Args:
            query: 検索クエリ
            k: 取得するドキュメントの数

        Returns:
            List[Dict[str, Any]]: 関連ドキュメントのリスト

        Raises:
            ValueError: ベクトルストアが初期化されていない場合
        """
        if not self.vector_store:
            raise ValueError(
                'ベクトルストアが初期化されていません。'
                'initialize() を呼び出してください。'
            )

        # クエリに関連するドキュメントを検索
        docs = self.vector_store.similarity_search(query, k=k)

        # 結果を整形
        results = []
        for doc in docs:
            results.append(
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': getattr(doc, 'score', None),
                }
            )

        return results


class CharacterRAGService(FAISSRAGService):
    """キャラクター設定用の RAG サービス。

    キャラクターシートから情報を検索するための特化したサービスです。

    Attributes:
        character_loader: キャラクター設定ローダー
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """初期化メソッド。

        Args:
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。
        """
        super().__init__(settings)
        from src.human_like_ai.config.character import CharacterLoader

        self.character_loader = CharacterLoader(settings)

    def initialize_from_character_sheet(self) -> None:
        """キャラクターシートから RAG システムを初期化します。"""
        # キャラクターシートを読み込み
        # character_data = self.character_loader.get_character_data()
        character_text = self.character_loader.get_character_text()

        # RAGシステムを初期化
        self.initialize([character_text])

    def retrieve_character_info(self, query: str, k: int = 3) -> str:
        """キャラクター情報を検索し、結果を文字列として返します。

        Args:
            query: 検索クエリ
            k: 取得するドキュメントの数

        Returns:
            str: 検索結果の文字列表現
        """
        results = self.retrieve(query, k)
        return '\n\n'.join([result['content'] for result in results])
