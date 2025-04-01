"""
エージェントモジュール。

このモジュールは、人間らしいAIエージェントの基本クラスとファクトリーを提供します。
"""

from typing import Any

from src.human_like_ai.config.settings import Settings, get_settings
from src.human_like_ai.core.conversation import ConversationManager, PromptManager
from src.human_like_ai.core.memory import MemoryManager
from src.human_like_ai.core.rag import CharacterRAGService
from src.human_like_ai.emotion.extractor import EmotionEventExtractor
from src.human_like_ai.emotion.manager import EmotionManager
from src.human_like_ai.utils.llm import LLMService
from src.human_like_ai.utils.logging import get_default_logger


class Agent:
    """人間らしいAIエージェントクラス。

    人間らしい対話を行うためのエージェントを提供します。

    Attributes:
        settings: アプリケーション設定
        llm_service: LLMサービス
        memory_manager: 記憶管理
        emotion_manager: 感情管理
        emotion_extractor: 感情抽出
        rag_service: RAGサービス
        prompt_manager: プロンプト管理
        conversation_manager: 会話管理
        logger: ロガー
    """

    def __init__(
        self,
        llm_service: LLMService,
        memory_manager: MemoryManager,
        emotion_manager: EmotionManager,
        emotion_extractor: EmotionEventExtractor,
        rag_service: CharacterRAGService,
        prompt_manager: PromptManager,
        settings: Settings | None = None,
    ) -> None:
        """初期化メソッド。

        Args:
            llm_service: LLMサービス
            memory_manager: 記憶管理
            emotion_manager: 感情管理
            emotion_extractor: 感情抽出
            rag_service: RAGサービス
            prompt_manager: プロンプト管理
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。
        """
        self.settings = settings or get_settings()
        self.llm_service = llm_service
        self.memory_manager = memory_manager
        self.emotion_manager = emotion_manager
        self.emotion_extractor = emotion_extractor
        self.rag_service = rag_service
        self.prompt_manager = prompt_manager
        self.conversation_manager = ConversationManager(
            prompt_manager=prompt_manager,
            memory_manager=memory_manager,
            emotion_manager=emotion_manager,
            emotion_extractor=emotion_extractor,
            rag_service=rag_service,
            llm_service=llm_service,
            settings=settings,
        )
        self.logger = get_default_logger()

    def process_input(self, user_input: str) -> str:
        """ユーザー入力を処理し、応答を生成します。

        Args:
            user_input: ユーザー入力

        Returns:
            str: 生成された応答
        """
        self.logger.info(f'ユーザー入力: {user_input}')
        response = self.conversation_manager.process_input(user_input)
        self.logger.info(f'エージェント応答: {response}')
        return response

    def initialize_memories(self, memories: list[str]) -> None:
        """長期記憶を初期化します。

        Args:
            memories: 長期記憶のリスト
        """
        for memory in memories:
            self.memory_manager.add_memory(memory)

    def initialize_attentions(self, attentions: list[str]) -> None:
        """関心事を初期化します。

        Args:
            attentions: 関心事のリスト
        """
        for attention in attentions:
            self.memory_manager.add_attention(attention)

    def get_state(self) -> dict[str, Any]:
        """エージェントの状態を取得します。

        Returns:
            Dict[str, any]: エージェントの状態
        """
        return {
            'emotions': self.emotion_manager.get_emotions(),
            'global_mood': self.emotion_manager.get_global_mood(),
            'memories': self.memory_manager.memories,
            'attentions': self.memory_manager.attentions,
            'chat_history': self.memory_manager.chat_history,
        }


class AgentFactory:
    """エージェントファクトリークラス。

    エージェントを作成するためのファクトリーを提供します。
    """

    @staticmethod
    def create_agent(settings: Settings | None = None) -> Agent:
        """エージェントを作成します。

        Args:
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。

        Returns:
            Agent: 作成されたエージェント
        """
        settings = settings or get_settings()
        logger = get_default_logger(settings)
        logger.info('エージェントの作成を開始します。')

        # 各コンポーネントの初期化
        llm_service = LLMService(settings)
        memory_manager = MemoryManager(settings)
        emotion_manager = EmotionManager()
        emotion_extractor = EmotionEventExtractor(settings.model_name)
        rag_service = CharacterRAGService(settings)
        prompt_manager = PromptManager(settings)

        # RAGサービスの初期化
        logger.info('キャラクターRAGサービスを初期化します。')
        rag_service.initialize_from_character_sheet()

        # エージェントの作成
        agent = Agent(
            llm_service=llm_service,
            memory_manager=memory_manager,
            emotion_manager=emotion_manager,
            emotion_extractor=emotion_extractor,
            rag_service=rag_service,
            prompt_manager=prompt_manager,
            settings=settings,
        )

        # 初期記憶と関心事の設定
        agent.initialize_memories(
            ['ユーザーは気さくで優しい。', 'ユーザーは犬が好き。']
        )
        agent.initialize_attentions(['おなかが空いた。', '眠い。'])

        logger.info('エージェントの作成が完了しました。')
        return agent
