"""
感情マネージャーモジュールのテスト。

このモジュールは、感情マネージャーモジュールの機能をテストします。
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.human_like_ai.emotion.manager import EmotionManager
from src.human_like_ai.emotion.models import (
    DEFAULT_TIMEZONE,
    BasicEmotion,
    Emotion,
)


@pytest.fixture
def emotion_manager() -> EmotionManager:
    """感情マネージャーのフィクスチャ。"""
    return EmotionManager()


def test_emotion_manager_init(emotion_manager: EmotionManager) -> None:
    """感情マネージャーの初期化テスト。"""
    assert emotion_manager.emotions == []
    # global_moodは初期化時に各感情に0.0が設定される
    assert all(value == 0.0 for value in emotion_manager.global_mood.values())
    assert emotion_manager._dirty is False


def test_find_event(emotion_manager: EmotionManager) -> None:
    """イベント検索テスト。"""
    # テスト用の感情イベントを追加
    emotion = Emotion(
        label=BasicEmotion.JOY,
        intensity=0.5,
        target='ユーザー',
        decay_rate=0.01,
        amplification=1.0,
    )
    emotion_manager.emotions.append(emotion)

    # 存在するイベントの検索
    found = emotion_manager._find_event('ユーザー', BasicEmotion.JOY)
    assert found is emotion

    # 存在しないイベントの検索
    not_found = emotion_manager._find_event('ユーザー', BasicEmotion.ANGER)
    assert not_found is None


def test_find_opposite_event(emotion_manager: EmotionManager) -> None:
    """反対感情イベント検索テスト。"""
    # テスト用の感情イベントを追加
    emotion = Emotion(
        label=BasicEmotion.JOY,
        intensity=0.5,
        target='ユーザー',
        decay_rate=0.01,
        amplification=1.0,
    )
    emotion_manager.emotions.append(emotion)

    # 反対感情イベントの検索
    opposite = emotion_manager._find_opposite_event('ユーザー', BasicEmotion.SADNESS)
    assert opposite is emotion

    # 存在しない反対感情イベントの検索
    not_found = emotion_manager._find_opposite_event('ユーザー', BasicEmotion.ANGER)
    assert not_found is None


def test_derive_compound_emotion(emotion_manager: EmotionManager) -> None:
    """複合感情導出テスト。"""
    # テスト用の感情イベントを作成
    joy = Emotion(
        label=BasicEmotion.JOY,
        intensity=0.5,
        target='ユーザー',
        decay_rate=0.01,
        amplification=1.0,
    )
    anticipation = Emotion(
        label=BasicEmotion.ANTICIPATION,
        intensity=0.5,
        target='ユーザー',
        decay_rate=0.01,
        amplification=1.0,
    )

    # 複合感情の導出
    compound = emotion_manager._derive_compound_emotion([joy, anticipation])
    assert compound is not None
    assert compound['jp'] == '楽観'

    # 単一感情の場合は複合感情にならない
    single = emotion_manager._derive_compound_emotion([joy])
    assert single is None

    # 複合感情が定義されていない組み合わせ
    undefined = emotion_manager._derive_compound_emotion(
        [
            Emotion(
                label=BasicEmotion.JOY,
                intensity=0.5,
                target='A',
                decay_rate=0.01,
                amplification=1.0,
            ),
            Emotion(
                label=BasicEmotion.ANGER,
                intensity=0.5,
                target='A',
                decay_rate=0.01,
                amplification=1.0,
            ),
            Emotion(
                label=BasicEmotion.FEAR,
                intensity=0.5,
                target='A',
                decay_rate=0.01,
                amplification=1.0,
            ),
        ]
    )
    assert undefined is None


def test_update_emotion_new(emotion_manager: EmotionManager) -> None:
    """新規感情更新テスト。"""
    # 新しい感情の更新
    emotion_manager.update_emotion(BasicEmotion.JOY, 'ユーザー', 'medium')

    # 感情が追加されたことを確認
    assert len(emotion_manager.emotions) == 1
    emotion = emotion_manager.emotions[0]
    assert emotion.label == BasicEmotion.JOY
    assert emotion.target == 'ユーザー'
    assert emotion.intensity == 0.05  # medium = 0.05
    assert emotion_manager._dirty is True


def test_update_emotion_existing(emotion_manager: EmotionManager) -> None:
    """既存感情更新テスト。"""
    # 既存の感情を追加
    emotion = Emotion(
        label=BasicEmotion.JOY,
        intensity=0.5,
        target='ユーザー',
        decay_rate=0.01,
        amplification=1.0,
    )
    emotion_manager.emotions.append(emotion)

    # 同じ感情の更新
    emotion_manager.update_emotion(BasicEmotion.JOY, 'ユーザー', 'medium')

    # 感情が更新されたことを確認
    assert len(emotion_manager.emotions) == 1
    updated = emotion_manager.emotions[0]
    assert updated.intensity == 0.55  # 0.5 + 0.05
    assert emotion_manager._dirty is True


def test_update_emotion_opposite(emotion_manager: EmotionManager) -> None:
    """反対感情更新テスト。"""
    # 既存の感情を追加
    emotion = Emotion(
        label=BasicEmotion.JOY,
        intensity=0.5,
        target='ユーザー',
        decay_rate=0.01,
        amplification=1.0,
    )
    emotion_manager.emotions.append(emotion)

    # 反対感情の更新
    emotion_manager.update_emotion(BasicEmotion.SADNESS, 'ユーザー', 'medium')

    # 既存の感情が減少したことを確認
    assert len(emotion_manager.emotions) == 1
    updated = emotion_manager.emotions[0]
    assert updated.label == BasicEmotion.JOY
    assert updated.intensity == 0.45  # 0.5 - 0.05
    assert emotion_manager._dirty is True


def test_update_emotion_opposite_removal(emotion_manager: EmotionManager) -> None:
    """反対感情による削除テスト。"""
    # 既存の感情を追加(弱い感情)
    emotion = Emotion(
        label=BasicEmotion.JOY,
        intensity=0.03,
        target='ユーザー',
        decay_rate=0.01,
        amplification=1.0,
    )
    emotion_manager.emotions.append(emotion)

    # 反対感情の更新(強い感情)
    emotion_manager.update_emotion(BasicEmotion.SADNESS, 'ユーザー', 'medium')

    # 既存の感情が削除され、反対感情が追加されたことを確認
    assert len(emotion_manager.emotions) == 1
    new_emotion = emotion_manager.emotions[0]
    assert new_emotion.label == BasicEmotion.SADNESS
    assert new_emotion.intensity == pytest.approx(0.02, abs=1e-10)  # 0.05 - 0.03
    assert emotion_manager._dirty is True


def test_update_from_llm(emotion_manager: EmotionManager) -> None:
    """LLMからの更新テスト。"""
    # LLMからの感情イベント
    events = [
        {'target': 'ユーザー', 'label': 'joy', 'strength': 'medium'},
        {'target': '話題', 'label': 'interest', 'strength': 'strong'},  # 無効な感情
        {'target': '自分自身', 'label': 'fear', 'strength': 'weak'},
    ]

    # 感情の更新
    emotion_manager.update_from_llm(events)

    # 有効な感情のみが追加されたことを確認
    assert len(emotion_manager.emotions) == 2
    labels = [e.label for e in emotion_manager.emotions]
    assert BasicEmotion.JOY in labels
    assert BasicEmotion.FEAR in labels


def test_update_global_mood(emotion_manager: EmotionManager) -> None:
    """全体的な感情状態の更新テスト。"""
    # テスト用の感情イベントを追加
    emotion_manager.emotions = [
        Emotion(
            label=BasicEmotion.JOY,
            intensity=0.5,
            target='A',
            decay_rate=0.01,
            amplification=1.0,
        ),
        Emotion(
            label=BasicEmotion.JOY,
            intensity=0.7,
            target='B',
            decay_rate=0.01,
            amplification=1.0,
        ),
        Emotion(
            label=BasicEmotion.ANGER,
            intensity=0.3,
            target='C',
            decay_rate=0.01,
            amplification=1.0,
        ),
    ]

    # 全体的な感情状態の更新
    emotion_manager.update_global_mood()

    # 全体的な感情状態の確認
    assert emotion_manager.global_mood[BasicEmotion.JOY] == 0.6  # (0.5 + 0.7) / 2
    assert emotion_manager.global_mood[BasicEmotion.ANGER] == 0.3


def test_apply_decay(emotion_manager: EmotionManager) -> None:
    """感情の減衰テスト。"""
    # 現在時刻の取得
    now = datetime.now(DEFAULT_TIMEZONE)
    past = now - timedelta(minutes=10)

    # テスト用の感情イベントを追加
    emotion_manager.emotions = [
        Emotion(
            label=BasicEmotion.JOY,
            intensity=0.5,
            target='A',
            last_updated=past,
            decay_rate=0.01,
            amplification=1.0,
        ),
        Emotion(
            label=BasicEmotion.ANGER,
            intensity=0.05,
            target='B',
            last_updated=past,
            decay_rate=0.01,
            amplification=1.0,
        ),
    ]

    # 感情の減衰を適用
    with patch('src.human_like_ai.emotion.manager.datetime') as mock_datetime:
        mock_datetime.now.return_value = now
        emotion_manager.apply_decay(unit_seconds=60)

    # 減衰後の感情を確認
    assert len(emotion_manager.emotions) == 1  # 弱い怒りは削除される
    assert emotion_manager.emotions[0].label == BasicEmotion.JOY
    assert emotion_manager.emotions[0].intensity < 0.5  # 減衰している
    # _commit_updates()が呼ばれるため、_dirtyはFalseになる
    assert emotion_manager._dirty is False


def test_generate_output(emotion_manager: EmotionManager) -> None:
    """出力生成テスト。"""
    # テスト用の感情イベントを追加
    emotion_manager.emotions = [
        Emotion(
            label=BasicEmotion.JOY,
            intensity=0.5,
            target='ユーザー',
            decay_rate=0.01,
            amplification=1.0,
        ),
        Emotion(
            label=BasicEmotion.ANTICIPATION,
            intensity=0.5,
            target='ユーザー',
            decay_rate=0.01,
            amplification=1.0,
        ),
        Emotion(
            label=BasicEmotion.ANGER,
            intensity=0.8,
            target='話題',
            decay_rate=0.01,
            amplification=1.0,
        ),
    ]

    # 全体的な感情状態を設定
    emotion_manager.global_mood = {
        BasicEmotion.JOY: 0.5,
        BasicEmotion.ANTICIPATION: 0.5,
        BasicEmotion.ANGER: 0.8,
    }

    # 出力の生成
    output = emotion_manager.generate_output()

    # 出力の確認
    assert '基本感情' in output
    assert '喜び' in output
    assert '期待' in output
    assert '激怒' in output  # 強い怒り
    assert '[ユーザー]' in output
    assert '[話題]' in output
    assert '楽観' in output  # 複合感情


def test_get_emotions(emotion_manager: EmotionManager) -> None:
    """感情イベントリスト取得テスト。"""
    # テスト用の感情イベントを追加
    emotions = [
        Emotion(
            label=BasicEmotion.JOY,
            intensity=0.5,
            target='A',
            decay_rate=0.01,
            amplification=1.0,
        ),
        Emotion(
            label=BasicEmotion.ANGER,
            intensity=0.3,
            target='B',
            decay_rate=0.01,
            amplification=1.0,
        ),
    ]
    emotion_manager.emotions = emotions

    # 感情イベントリストの取得
    result = emotion_manager.get_emotions()
    assert result == emotions


def test_get_global_mood(emotion_manager: EmotionManager) -> None:
    """全体的な感情状態取得テスト。"""
    # テスト用の全体的な感情状態を設定
    mood = {
        BasicEmotion.JOY: 0.5,
        BasicEmotion.ANGER: 0.3,
    }
    emotion_manager.global_mood = mood

    # 全体的な感情状態の取得
    result = emotion_manager.get_global_mood()
    assert result == mood
