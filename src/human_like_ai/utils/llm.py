"""
LLM関連ユーティリティモジュール。

このモジュールは、LLMとの対話を行うためのユーティリティクラスを提供します。
"""

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.human_like_ai.config.settings import Settings, get_settings


class LLMService:
    """LLMサービスクラス。

    LLMとの対話を行うためのサービスを提供します。

    Attributes:
        settings: アプリケーション設定
        model: LLMモデル
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """初期化メソッド。

        Args:
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。
        """
        self.settings = settings or get_settings()
        self.model = ChatOpenAI(
            model=self.settings.model_name,
            temperature=self.settings.temperature,
        )

    def generate(self, context: dict[str, Any]) -> str:
        """LLMを使用して応答を生成します。

        Args:
            context: プロンプトコンテキスト

        Returns:
            str: 生成された応答
        """
        # プロンプトテンプレートが提供されている場合は使用
        if 'prompt_template' in context:
            prompt_template = context.pop('prompt_template')
            messages = prompt_template.format_prompt(**context).to_messages()
        # メッセージが直接提供されている場合は使用
        elif 'messages' in context:
            messages = context['messages']
        # それ以外の場合はエラー
        else:
            raise ValueError('プロンプトテンプレートまたはメッセージが必要です。')

        # LLMを呼び出して応答を生成
        response = self.model.invoke(messages)
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)

    def generate_with_messages(self, messages: list[BaseMessage]) -> str:
        """メッセージリストを使用して応答を生成します。

        Args:
            messages: メッセージリスト

        Returns:
            str: 生成された応答
        """
        response = self.model.invoke(messages)
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)

    def generate_with_prompt(
        self, prompt_template: ChatPromptTemplate, **kwargs: dict[str, Any]
    ) -> str:
        """プロンプトテンプレートを使用して応答を生成します。

        Args:
            prompt_template: プロンプトテンプレート
            **kwargs: プロンプト変数

        Returns:
            str: 生成された応答
        """
        messages = prompt_template.format_prompt(**kwargs).to_messages()
        response = self.model.invoke(messages)
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)
