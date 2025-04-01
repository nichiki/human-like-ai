"""
感情モデルモジュールのテスト。

このモジュールは、感情モデルモジュールの機能をテストします。
"""

from datetime import datetime, timedelta

import pytest

from src.human_like_ai.emotion.models import (
    ALTERNATE_NAMES,
    COMPOUND_EMOTIONS,
    DEFAULT_TIMEZONE,
    EVENT_STRENGTH_MAPPING,
    INTENSITY_THRESHOLDS,
    OPPOSITE_EMOTIONS,
    BasicEmotion,
    Emotion,
    get_intensity_category,
)


def test_basic_emotion_enum() -> None:
    """基本感情の列挙型テスト。"""
    # 列挙型の値の確認
    assert BasicEmotion.JOY.value == 'joy'
    assert BasicEmotion.ANGER.value == 'anger'
    assert BasicEmotion.FEAR.value == 'fear'
    assert BasicEmotion.SADNESS.value == 'sadness'
    assert BasicEmotion.DISGUST.value == 'disgust'
    assert BasicEmotion.SURPRISE.value == 'surprise'
    assert BasicEmotion.ANTICIPATION.value == 'anticipation'
    assert BasicEmotion.TRUST.value == 'trust'

    # 日本語名の確認
    assert BasicEmotion.JOY.japanese == '喜び'
    assert BasicEmotion.ANGER.japanese == '怒り'
    assert BasicEmotion.FEAR.japanese == '恐れ'
    assert BasicEmotion.SADNESS.japanese == '悲しみ'
    assert BasicEmotion.DISGUST.japanese == '嫌悪'
    assert BasicEmotion.SURPRISE.japanese == '驚き'
    assert BasicEmotion.ANTICIPATION.japanese == '期待'
    assert BasicEmotion.TRUST.japanese == '信頼'


def test_event_strength_mapping() -> None:
    """イベント強度マッピングのテスト。"""
    assert EVENT_STRENGTH_MAPPING['weak'] == 0.03
    assert EVENT_STRENGTH_MAPPING['medium'] == 0.05
    assert EVENT_STRENGTH_MAPPING['strong'] == 0.10


def test_alternate_names() -> None:
    """感情の別名テスト。"""
    # 喜びの別名
    assert ALTERNATE_NAMES[BasicEmotion.JOY]['weak']['jp'] == 'ほのかな喜び'
    assert ALTERNATE_NAMES[BasicEmotion.JOY]['strong']['jp'] == '恍惚'

    # 怒りの別名
    assert ALTERNATE_NAMES[BasicEmotion.ANGER]['weak']['jp'] == '苛立ち'
    assert ALTERNATE_NAMES[BasicEmotion.ANGER]['strong']['jp'] == '激怒'


def test_compound_emotions() -> None:
    """複合感情テスト。"""
    # 喜び + 期待 = 楽観
    compound_key = frozenset({BasicEmotion.JOY, BasicEmotion.ANTICIPATION})
    assert COMPOUND_EMOTIONS[compound_key]['jp'] == '楽観'

    # 怒り + 嫌悪 = 軽蔑
    compound_key = frozenset({BasicEmotion.ANGER, BasicEmotion.DISGUST})
    assert COMPOUND_EMOTIONS[compound_key]['jp'] == '軽蔑'


def test_opposite_emotions() -> None:
    """反対感情テスト。"""
    assert OPPOSITE_EMOTIONS[BasicEmotion.JOY] == BasicEmotion.SADNESS
    assert OPPOSITE_EMOTIONS[BasicEmotion.SADNESS] == BasicEmotion.JOY
    assert OPPOSITE_EMOTIONS[BasicEmotion.ANGER] == BasicEmotion.FEAR
    assert OPPOSITE_EMOTIONS[BasicEmotion.FEAR] == BasicEmotion.ANGER
    assert OPPOSITE_EMOTIONS[BasicEmotion.TRUST] == BasicEmotion.DISGUST
    assert OPPOSITE_EMOTIONS[BasicEmotion.DISGUST] == BasicEmotion.TRUST
    assert OPPOSITE_EMOTIONS[BasicEmotion.ANTICIPATION] == BasicEmotion.SURPRISE
    assert OPPOSITE_EMOTIONS[BasicEmotion.SURPRISE] == BasicEmotion.ANTICIPATION


def test_intensity_thresholds() -> None:
    """強度閾値テスト。"""
    for emotion in BasicEmotion:
        lower, upper = INTENSITY_THRESHOLDS[emotion]
        assert 0 <= lower < upper <= 1


def test_get_intensity_category() -> None:
    """強度カテゴリ取得テスト。"""
    # 喜びの強度カテゴリ
    assert get_intensity_category(BasicEmotion.JOY, 0.1) == 'weak'
    assert get_intensity_category(BasicEmotion.JOY, 0.5) == 'basic'
    assert get_intensity_category(BasicEmotion.JOY, 0.8) == 'strong'

    # 怒りの強度カテゴリ
    assert get_intensity_category(BasicEmotion.ANGER, 0.1) == 'weak'
    assert get_intensity_category(BasicEmotion.ANGER, 0.5) == 'basic'
    assert get_intensity_category(BasicEmotion.ANGER, 0.8) == 'strong'


def test_emotion_model() -> None:
    """感情モデルテスト。"""
    # 基本的な感情モデルの作成
    emotion = Emotion(
        label=BasicEmotion.JOY,
        intensity=0.5,
        target='ユーザー',
        decay_rate=0.01,
        amplification=1.2,
    )

    # 属性の確認
    assert emotion.label == BasicEmotion.JOY
    assert emotion.intensity == 0.5
    assert emotion.target == 'ユーザー'
    assert emotion.decay_rate == 0.01
    assert emotion.amplification == 1.2
    assert isinstance(emotion.last_updated, datetime)


def test_emotion_model_default_values() -> None:
    """感情モデルのデフォルト値テスト。"""
    # 最小限の引数で感情モデルを作成
    emotion = Emotion(
        label=BasicEmotion.JOY,
        target='ユーザー',
    )

    # デフォルト値の確認
    assert emotion.intensity == 0
    assert emotion.decay_rate == 0.01
    assert emotion.amplification == 1.0
    assert isinstance(emotion.last_updated, datetime)


def test_emotion_model_validation() -> None:
    """感情モデルのバリデーションテスト。"""
    # 強度が範囲外の場合
    with pytest.raises(ValueError):
        Emotion(
            label=BasicEmotion.JOY,
            intensity=1.5,  # 1.0を超える値
            target='ユーザー',
        )

    with pytest.raises(ValueError):
        Emotion(
            label=BasicEmotion.JOY,
            intensity=-0.5,  # 0.0未満の値
            target='ユーザー',
        )

    # 減衰率が範囲外の場合
    with pytest.raises(ValueError):
        Emotion(
            label=BasicEmotion.JOY,
            target='ユーザー',
            decay_rate=1.5,  # 1.0を超える値
        )

    with pytest.raises(ValueError):
        Emotion(
            label=BasicEmotion.JOY,
            target='ユーザー',
            decay_rate=-0.5,  # 0.0未満の値
        )

    # 増幅係数が負の場合
    with pytest.raises(ValueError):
        Emotion(
            label=BasicEmotion.JOY,
            target='ユーザー',
            amplification=-0.5,  # 0.0未満の値
        )


def test_emotion_model_last_updated() -> None:
    """感情モデルの最終更新時刻テスト。"""
    # 現在時刻の取得
    now = datetime.now(DEFAULT_TIMEZONE)

    # 感情モデルの作成
    emotion = Emotion(
        label=BasicEmotion.JOY,
        target='ユーザー',
    )

    # 最終更新時刻が現在時刻に近いことを確認
    time_diff = abs((emotion.last_updated - now).total_seconds())
    assert time_diff < 1.0  # 1秒以内の差

    # 最終更新時刻を明示的に設定
    past_time = now - timedelta(hours=1)
    emotion = Emotion(
        label=BasicEmotion.JOY,
        target='ユーザー',
        last_updated=past_time,
    )

    # 設定した時刻が反映されていることを確認
    assert emotion.last_updated == past_time
