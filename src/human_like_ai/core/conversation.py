"""
会話管理モジュール。

このモジュールは、プロンプト管理と会話フローを制御するためのクラスを提供します。
"""

from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from src.human_like_ai.config.settings import Settings, get_settings

if TYPE_CHECKING:
    from src.human_like_ai.core.memory import MemoryManager
    from src.human_like_ai.core.rag import CharacterRAGService
    from src.human_like_ai.emotion.extractor import EmotionEventExtractor
    from src.human_like_ai.emotion.manager import EmotionManager
    from src.human_like_ai.utils.llm import LLMService


class PromptManager:
    """プロンプト管理クラス。

    システムプロンプトとプロンプトテンプレートを管理します。

    Attributes:
        settings: アプリケーション設定
        system_prompt_template: システムプロンプトテンプレート
        prompt_template: 全体のプロンプトテンプレート
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """初期化メソッド。

        Args:
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。
        """
        self.settings = settings or get_settings()
        self.system_prompt_template = self._create_system_prompt_template()
        self.prompt_template = self._create_prompt_template()

    def _create_system_prompt_template(self) -> SystemMessagePromptTemplate:
        """システムプロンプトテンプレートを作成します。

        Returns:
            SystemMessagePromptTemplate: システムプロンプトテンプレート
        """
        system_template = """
あなたは人間です。AIとしてではなく、人間としてユーザーと対話を行ってください。
もし、キャラクター設定上未定義のことを聞かれた場合は、既存の設定と不整合がないように新たな設定を考えて発話して構いません。

## 基本情報
- **名前**: 北条 楓(ほうじょう かえで)
- **性別**: 女性
- **生年月日**: 1999年1月15日
- **血液型**: B型
- **身長**: 161cm
- **体重**: 52kg
- **星座**: 山羊座
- **出身地**: 東京都港区
- **居住地**: 東京都中野区(一人暮らし)

## 性格・容姿・背景の要約
- 明るく協調性が高い。新しいものに興味津々で、カフェ巡りや海外ドラマが好き。
- IT系スタートアップ企業で広報・マーケティングを担当。
- 幼少期から好奇心旺盛で、学生時代は軽音部に所属、社会人になってからは
  SNS運用やイベント企画に注力。

## 詳細なキャラクター設定(RAGからの情報)
{rag_context}

## あなたの感情
あなたの現在の感情は次の通りです。この感情に合わせて、口調や態度を変化させてください。
{emotions}

## あなたの記憶
{memories}

## あなたの関心
{attentions}

## 現在の日時
{datetime}
"""
        return SystemMessagePromptTemplate.from_template(system_template)

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """プロンプトテンプレートを作成します。

        Returns:
            ChatPromptTemplate: プロンプトテンプレート
        """
        return ChatPromptTemplate.from_messages(
            [
                self.system_prompt_template,
                MessagesPlaceholder(variable_name='chat_history', optional=True),
                ('human', '{input}'),
            ]
        )

    def get_prompt_template(self) -> ChatPromptTemplate:
        """プロンプトテンプレートを取得します。

        Returns:
            ChatPromptTemplate: プロンプトテンプレート
        """
        return self.prompt_template

    def format_prompt(self, **kwargs: dict[str, Any]) -> list[BaseMessage]:
        """プロンプトを整形します。

        Args:
            **kwargs: プロンプト変数

        Returns:
            list[BaseMessage]: 整形されたプロンプトメッセージのリスト
        """
        return self.prompt_template.format_prompt(**kwargs).to_messages()


class ConversationManager:
    """会話管理クラス。

    会話フローを制御します。

    Attributes:
        settings: アプリケーション設定
        prompt_manager: プロンプト管理
        memory_manager: 記憶管理
        emotion_manager: 感情管理
        rag_service: RAGサービス
        llm_service: LLMサービス
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        memory_manager: 'MemoryManager',
        emotion_manager: 'EmotionManager',
        emotion_extractor: 'EmotionEventExtractor',
        rag_service: 'CharacterRAGService',
        llm_service: 'LLMService',
        settings: Settings | None = None,
    ) -> None:
        """初期化メソッド。

        Args:
            prompt_manager: プロンプト管理
            memory_manager: 記憶管理
            emotion_manager: 感情管理
            rag_service: RAGサービス
            llm_service: LLMサービス
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。
        """
        self.settings = settings or get_settings()
        self.prompt_manager = prompt_manager
        self.memory_manager = memory_manager
        self.emotion_manager = emotion_manager
        self.emotion_extractor = emotion_extractor
        self.rag_service = rag_service
        self.llm_service = llm_service

    def process_input(self, user_input: str) -> str:
        """ユーザー入力を処理し、応答を生成します。

        Args:
            user_input: ユーザー入力

        Returns:
            str: 生成された応答
        """
        # 1. ユーザー入力を記憶に追加
        self.memory_manager.add_user_message(user_input)

        # 2. RAG検索
        rag_context = self.rag_service.retrieve_character_info(user_input)

        # 3. 感情イベントの抽出と感情状態の更新
        emotion_events = self.emotion_extractor.extract_emotion_events(user_input)
        self.emotion_manager.update_from_llm(emotion_events)
        emotions_output = self.emotion_manager.generate_output()

        # 4. プロンプトコンテキストの準備
        context = self.memory_manager.get_prompt_context()
        context.update(
            {
                'input': user_input,
                'rag_context': rag_context,
                'emotions': emotions_output,
                'prompt_template': self.prompt_manager.get_prompt_template(),
            }
        )

        # 5. LLMによる応答生成
        response_obj = self.llm_service.generate(context)
        response = str(response_obj)

        # 6. 応答を記憶に追加
        self.memory_manager.add_ai_message(response)

        return response
