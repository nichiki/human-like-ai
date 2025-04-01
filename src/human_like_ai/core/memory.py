"""
記憶管理モジュール。

このモジュールは、会話履歴、長期記憶、関心事などの記憶を管理するためのクラスを提供します。
"""

from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.human_like_ai.config.settings import Settings, get_settings
from src.human_like_ai.emotion.models import DEFAULT_TIMEZONE


class MemoryManager:
    """記憶管理クラス。

    会話履歴、長期記憶、関心事などの記憶を管理します。

    Attributes:
        settings: アプリケーション設定
        chat_history: 会話履歴
        memories: 長期記憶のリスト
        attentions: 関心事のリスト
        max_history_length: 保持する会話履歴の最大長
    """

    def __init__(
        self,
        settings: Settings | None = None,
        max_history_length: int = 10,
    ) -> None:
        """初期化メソッド。

        Args:
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。
            max_history_length: 保持する会話履歴の最大長
        """
        self.settings = settings or get_settings()
        self.chat_history: list[BaseMessage] = []
        self.memories: list[str] = []
        self.attentions: list[str] = []
        self.max_history_length = max_history_length

    def add_user_message(self, content: str) -> None:
        """ユーザーメッセージを会話履歴に追加します。

        Args:
            content: メッセージ内容
        """
        self.chat_history.append(HumanMessage(content=content))
        self._trim_history()

    def add_ai_message(self, content: str) -> None:
        """AIメッセージを会話履歴に追加します。

        Args:
            content: メッセージ内容
        """
        self.chat_history.append(AIMessage(content=content))
        self._trim_history()

    def add_system_message(self, content: str) -> None:
        """システムメッセージを会話履歴に追加します。

        Args:
            content: メッセージ内容
        """
        self.chat_history.append(SystemMessage(content=content))
        self._trim_history()

    def _trim_history(self) -> None:
        """会話履歴を最大長に制限します。"""
        if len(self.chat_history) > self.max_history_length:
            self.chat_history = self.chat_history[-self.max_history_length :]

    def clear_history(self) -> None:
        """会話履歴をクリアします。"""
        self.chat_history = []

    def add_memory(self, memory: str) -> None:
        """長期記憶を追加します。

        Args:
            memory: 記憶内容
        """
        if memory not in self.memories:
            self.memories.append(memory)

    def remove_memory(self, memory: str) -> None:
        """長期記憶を削除します。

        Args:
            memory: 削除する記憶内容
        """
        if memory in self.memories:
            self.memories.remove(memory)

    def add_attention(self, attention: str) -> None:
        """関心事を追加します。

        Args:
            attention: 関心事内容
        """
        if attention not in self.attentions:
            self.attentions.append(attention)

    def remove_attention(self, attention: str) -> None:
        """関心事を削除します。

        Args:
            attention: 削除する関心事内容
        """
        if attention in self.attentions:
            self.attentions.remove(attention)

    def get_chat_history(self) -> list[BaseMessage]:
        """会話履歴を取得します。

        Returns:
            list[BaseMessage]: 会話履歴のリスト
        """
        return self.chat_history

    def get_memories_text(self) -> str:
        """長期記憶をテキスト形式で取得します。

        Returns:
            str: 長期記憶のテキスト表現
        """
        return '\n'.join(self.memories) if self.memories else 'なし'

    def get_attentions_text(self) -> str:
        """関心事をテキスト形式で取得します。

        Returns:
            str: 関心事のテキスト表現
        """
        return '\n'.join(self.attentions) if self.attentions else 'なし'

    def get_current_datetime(self) -> str:
        """現在の日時を取得します。

        Returns:
            str: 現在の日時の文字列表現
        """
        return datetime.now(DEFAULT_TIMEZONE).strftime('%Y/%m/%d %H:%M:%S')

    def get_prompt_context(self) -> dict[str, Any]:
        """プロンプト用のコンテキストを取得します。

        Returns:
            Dict[str, Any]: プロンプト用のコンテキスト
        """
        return {
            'chat_history': self.chat_history,
            'memories': self.get_memories_text(),
            'attentions': self.get_attentions_text(),
            'datetime': self.get_current_datetime(),
        }
