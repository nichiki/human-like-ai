"""
感情モデルモジュール。

このモジュールは、感情の基本モデルと関連する定数を定義します。
"""

from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from src.human_like_ai.config.settings import get_settings

# タイムゾーン設定
settings = get_settings()
DEFAULT_TIMEZONE = ZoneInfo(settings.timezone)

# 感情強度の最小閾値
MIN_INTENSITY_THRESHOLD: float = 0.0


class BasicEmotion(Enum):
    """基本感情の列挙型。

    8つの基本感情を定義します。
    """

    JOY = 'joy'
    ANTICIPATION = 'anticipation'
    ANGER = 'anger'
    DISGUST = 'disgust'
    SADNESS = 'sadness'
    SURPRISE = 'surprise'
    FEAR = 'fear'
    TRUST = 'trust'

    @property
    def japanese(self) -> str:
        """感情の日本語名を取得します。

        Returns:
            str: 感情の日本語名
        """
        mapping = {
            BasicEmotion.JOY: '喜び',
            BasicEmotion.ANTICIPATION: '期待',
            BasicEmotion.ANGER: '怒り',
            BasicEmotion.DISGUST: '嫌悪',
            BasicEmotion.SADNESS: '悲しみ',
            BasicEmotion.SURPRISE: '驚き',
            BasicEmotion.FEAR: '恐れ',
            BasicEmotion.TRUST: '信頼',
        }
        return mapping[self]


# LLMからのイベント更新用強さマッピング
EVENT_STRENGTH_MAPPING: dict[str, float] = {
    'weak': 0.03,
    'medium': 0.05,
    'strong': 0.10,
}

# 基本感情の強さに応じた別名
ALTERNATE_NAMES: dict[BasicEmotion, dict[str, dict[str, str]]] = {
    BasicEmotion.JOY: {
        'weak': {'en': 'mild joy', 'jp': 'ほのかな喜び'},
        'strong': {'en': 'ecstasy', 'jp': '恍惚'},
    },
    BasicEmotion.ANTICIPATION: {
        'weak': {'en': 'hope', 'jp': '希望'},
        'strong': {'en': 'eagerness', 'jp': '熱望'},
    },
    BasicEmotion.ANGER: {
        'weak': {'en': 'irritation', 'jp': '苛立ち'},
        'strong': {'en': 'rage', 'jp': '激怒'},
    },
    BasicEmotion.DISGUST: {
        'weak': {'en': 'revulsion', 'jp': '軽い嫌悪'},
        'strong': {'en': 'loathing', 'jp': '激しい嫌悪'},
    },
    BasicEmotion.SADNESS: {
        'weak': {'en': 'melancholy', 'jp': '物悲しさ'},
        'strong': {'en': 'grief', 'jp': '深い悲しみ'},
    },
    BasicEmotion.SURPRISE: {
        'weak': {'en': 'mild surprise', 'jp': '軽い驚き'},
        'strong': {'en': 'astonishment', 'jp': '大いなる驚き'},
    },
    BasicEmotion.FEAR: {
        'weak': {'en': 'apprehension', 'jp': '不安'},
        'strong': {'en': 'terror', 'jp': '恐怖'},
    },
    BasicEmotion.TRUST: {
        'weak': {'en': 'fondness', 'jp': '好意'},
        'strong': {'en': 'admiration', 'jp': '深い信頼'},
    },
}

# 応用感情(複合感情)のマスタ
COMPOUND_EMOTIONS: dict[frozenset[BasicEmotion], dict[str, str]] = {
    frozenset({BasicEmotion.ANTICIPATION, BasicEmotion.JOY}): {
        'en': 'Optimism',
        'jp': '楽観',
    },
    frozenset({BasicEmotion.SURPRISE, BasicEmotion.SADNESS}): {
        'en': 'Disappointment',
        'jp': '失望',
    },
    frozenset({BasicEmotion.ANGER, BasicEmotion.JOY}): {'en': 'Pride', 'jp': '誇り'},
    frozenset({BasicEmotion.FEAR, BasicEmotion.SADNESS}): {
        'en': 'Despair',
        'jp': '絶望',
    },
    frozenset({BasicEmotion.DISGUST, BasicEmotion.JOY}): {
        'en': 'Morbidness',
        'jp': '病的状態',
    },
    frozenset({BasicEmotion.TRUST, BasicEmotion.SADNESS}): {
        'en': 'Sentimentality',
        'jp': '感傷',
    },
    frozenset({BasicEmotion.ANGER, BasicEmotion.ANTICIPATION}): {
        'en': 'Aggressiveness',
        'jp': '積極性',
    },
    frozenset({BasicEmotion.FEAR, BasicEmotion.SURPRISE}): {'en': 'Awe', 'jp': '畏敬'},
    frozenset({BasicEmotion.DISGUST, BasicEmotion.ANTICIPATION}): {
        'en': 'Cynicism',
        'jp': '冷笑',
    },
    frozenset({BasicEmotion.TRUST, BasicEmotion.SURPRISE}): {
        'en': 'Curiosity',
        'jp': '好奇心',
    },
    frozenset({BasicEmotion.SADNESS, BasicEmotion.ANTICIPATION}): {
        'en': 'Pessimism',
        'jp': '悲観',
    },
    frozenset({BasicEmotion.JOY, BasicEmotion.SURPRISE}): {
        'en': 'Delight',
        'jp': '歓喜',
    },
    frozenset({BasicEmotion.DISGUST, BasicEmotion.ANGER}): {
        'en': 'Contempt',
        'jp': '軽蔑',
    },
    frozenset({BasicEmotion.TRUST, BasicEmotion.FEAR}): {
        'en': 'Submission',
        'jp': '服従',
    },
    frozenset({BasicEmotion.SADNESS, BasicEmotion.ANGER}): {'en': 'Envy', 'jp': '羨望'},
    frozenset({BasicEmotion.JOY, BasicEmotion.FEAR}): {'en': 'Guilt', 'jp': '罪悪感'},
    frozenset({BasicEmotion.SURPRISE, BasicEmotion.ANGER}): {
        'en': 'Outrage',
        'jp': '憤慨',
    },
    frozenset({BasicEmotion.ANTICIPATION, BasicEmotion.FEAR}): {
        'en': 'Anxiety',
        'jp': '不安',
    },
    frozenset({BasicEmotion.SADNESS, BasicEmotion.DISGUST}): {
        'en': 'Remorse',
        'jp': '自責',
    },
    frozenset({BasicEmotion.JOY, BasicEmotion.TRUST}): {'en': 'Love', 'jp': '愛'},
    frozenset({BasicEmotion.SURPRISE, BasicEmotion.DISGUST}): {
        'en': 'Unbelief',
        'jp': '不信',
    },
    frozenset({BasicEmotion.ANTICIPATION, BasicEmotion.TRUST}): {
        'en': 'Hope',
        'jp': '希望',
    },
    frozenset({BasicEmotion.FEAR, BasicEmotion.DISGUST}): {'en': 'Shame', 'jp': '恥'},
    frozenset({BasicEmotion.ANGER, BasicEmotion.TRUST}): {
        'en': 'Dominance',
        'jp': '優位',
    },
}

# 基本感情の反対関係のマスタ
OPPOSITE_EMOTIONS: dict[BasicEmotion, BasicEmotion] = {
    BasicEmotion.JOY: BasicEmotion.SADNESS,
    BasicEmotion.SADNESS: BasicEmotion.JOY,
    BasicEmotion.TRUST: BasicEmotion.DISGUST,
    BasicEmotion.DISGUST: BasicEmotion.TRUST,
    BasicEmotion.FEAR: BasicEmotion.ANGER,
    BasicEmotion.ANGER: BasicEmotion.FEAR,
    BasicEmotion.SURPRISE: BasicEmotion.ANTICIPATION,
    BasicEmotion.ANTICIPATION: BasicEmotion.SURPRISE,
}

# 強さ判定の閾値マスタ
INTENSITY_THRESHOLDS: dict[BasicEmotion, tuple[float, float]] = {
    BasicEmotion.JOY: (0.3, 0.7),
    BasicEmotion.ANTICIPATION: (0.3, 0.7),
    BasicEmotion.ANGER: (0.3, 0.7),
    BasicEmotion.DISGUST: (0.3, 0.7),
    BasicEmotion.SADNESS: (0.3, 0.7),
    BasicEmotion.SURPRISE: (0.3, 0.7),
    BasicEmotion.FEAR: (0.3, 0.7),
    BasicEmotion.TRUST: (0.3, 0.7),
}
DEFAULT_INTENSITY_THRESHOLDS: tuple[float, float] = (0.3, 0.7)


def get_intensity_category(emotion: BasicEmotion, intensity: float) -> str:
    """感情の強度カテゴリを取得します。

    Args:
        emotion: 基本感情
        intensity: 感情の強度

    Returns:
        str: 強度カテゴリ('weak', 'basic', 'strong'のいずれか)
    """
    lower, upper = INTENSITY_THRESHOLDS.get(emotion, DEFAULT_INTENSITY_THRESHOLDS)
    if intensity < lower:
        return 'weak'
    elif intensity < upper:
        return 'basic'
    else:
        return 'strong'


class Emotion(BaseModel):
    """感情モデル。

    特定の対象に対する感情の状態を表します。

    Attributes:
        label: 基本感情のラベル
        intensity: 感情の累積状態の強さ(0.0〜1.0)
        target: 感情の対象
        decay_rate: 単位時間あたりの減衰率
        amplification: 増幅係数
        last_updated: 最終更新時刻
    """

    label: BasicEmotion = Field(..., description='基本感情のラベル')
    intensity: float = Field(0, description='感情の累積状態の強さ', ge=0, le=1)
    target: str = Field(..., description='感情の対象')
    decay_rate: float = Field(
        0.01, description='単位時間あたりの減衰率(例: 1分あたり1%減少)', ge=0, le=1
    )
    amplification: float = Field(
        1.0,
        description='増幅係数(同一対象で同じ感情が連続する際に掛け合わせる)',
        ge=0,
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(DEFAULT_TIMEZONE),
        description='最終更新時刻',
    )
