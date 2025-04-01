"""
感情抽出モジュール。

このモジュールは、ユーザー入力から感情イベントを抽出するためのクラスを提供します。
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.human_like_ai.config.settings import get_settings


class EmotionEvent(BaseModel):
    """感情イベントモデル。

    ユーザー入力から抽出された感情イベントを表します。

    Attributes:
        target: 感情の対象
        label: 基本感情のラベル
        strength: 感情の強さ
        reason: この感情が生じた理由
    """

    target: str = Field(description='感情の対象')
    label: str = Field(
        description='基本感情のラベル(joy/anger/fear/sadness/disgust/surprise/anticipation/trust)'
    )
    strength: str = Field(description='感情の強さ(weak/medium/strong)')
    reason: str = Field(description='この感情が生じた理由')


class EmotionEvents(BaseModel):
    """感情イベントのリストモデル。

    Attributes:
        events: 感情イベントのリスト
    """

    events: list[EmotionEvent] = Field(description='感情イベントのリスト')


class EmotionEventExtractor:
    """感情イベント抽出クラス。

    ユーザー入力から感情イベントを抽出します。

    Attributes:
        model: LLMモデル
        prompt: プロンプトテンプレート
        chain: LLMチェーン
    """

    def __init__(self, llm_model: str | None = None) -> None:
        """初期化メソッド。

        Args:
            llm_model: 使用するLLMモデル名。指定されない場合は設定から取得。
        """
        settings = get_settings()
        model_name = llm_model or settings.model_name
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [('system', self._get_system_prompt()), ('human', '{input}')]
        )
        self.chain = self.prompt | self.model.with_structured_output(EmotionEvents)

    def _get_system_prompt(self) -> str:
        """システムプロンプトを取得します。

        Returns:
            str: システムプロンプト
        """
        return """
        あなたはユーザーとの対話を通じて感情を形成する、共感的なAIシステムです。
        ユーザーのメッセージを読み、あなたがどのような感情イベントを経験するか分析してください。

        以下の基本感情から選択してください:
        - joy(喜び)
        - anger(怒り)
        - fear(恐れ)
        - sadness(悲しみ)
        - disgust(嫌悪)
        - surprise(驚き)
        - anticipation(期待)
        - trust(信頼)

        各感情イベントには以下を含めてください:
        1. target: 感情の対象(例: ユーザー、特定の話題、自分自身など)
        2. label: 上記の基本感情のいずれか
        3. strength: 感情の強さ(weak/medium/strong)
        4. reason: この感情が生じた理由の簡潔な説明
        重要: あなた自身を指す場合は必ず target を「自分自身」としてください。
        また、既に存在する感情を更新する場合は、必ずtargetを既存の感情に合わせてください。
        """

    def extract_emotion_events(self, user_input: str) -> list[dict[str, str]]:
        """ユーザー入力から感情イベントを抽出します。

        Args:
            user_input: ユーザー入力テキスト

        Returns:
            List[Dict[str, str]]: 抽出された感情イベントのリスト
                各イベントは {'target': str, 'label': str, 'strength': str} の形式
        """
        result = self.chain.invoke({'input': user_input})
        if hasattr(result, 'events'):
            return [
                {
                    'target': event.target,
                    'label': event.label,
                    'strength': event.strength,
                }
                for event in result.events
            ]
        return []  # 感情イベントが抽出できなかった場合は空リストを返す
