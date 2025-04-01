"""
感情管理モジュール。

このモジュールは、感情の状態を管理し、更新するためのクラスを提供します。
"""

import logging
from datetime import datetime

from src.human_like_ai.emotion.models import (
    ALTERNATE_NAMES,
    COMPOUND_EMOTIONS,
    DEFAULT_TIMEZONE,
    EVENT_STRENGTH_MAPPING,
    MIN_INTENSITY_THRESHOLD,
    OPPOSITE_EMOTIONS,
    BasicEmotion,
    Emotion,
    get_intensity_category,
)

# モジュール専用のロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class EmotionManager:
    """感情管理クラス。

    LLMからの感情イベント(対象、基本感情、イベント強さカテゴリ)を受け取り、
    内部状態(emotions と global_mood)を更新・管理します。

    Attributes:
        emotions: 感情イベントのリスト
        global_mood: 全体的な感情状態
        _dirty: 更新フラグ
    """

    def __init__(self) -> None:
        """初期化メソッド。"""
        self.emotions: list[Emotion] = []
        self.global_mood: dict[BasicEmotion, float] = dict.fromkeys(BasicEmotion, 0.0)
        self._dirty: bool = False

    # ── ヘルパーメソッド ──

    def _find_event(self, target: str, label: BasicEmotion) -> Emotion | None:
        """指定の対象と感情ラベルに合致するイベントを返します。

        Args:
            target: 感情の対象
            label: 感情ラベル

        Returns:
            Optional[Emotion]: 合致するイベント、見つからない場合はNone
        """
        for e in self.emotions:
            if e.target == target and e.label == label:
                return e
        return None

    def _find_opposite_event(self, target: str, label: BasicEmotion) -> Emotion | None:
        """指定対象における反対感情のイベントを返します。

        Args:
            target: 感情の対象
            label: 感情ラベル

        Returns:
            Optional[Emotion]: 反対感情のイベント、見つからない場合はNone
        """
        opposite_label = OPPOSITE_EMOTIONS.get(label)
        if opposite_label:
            return self._find_event(target, opposite_label)
        return None

    def _derive_compound_emotion(
        self, target_events: list[Emotion]
    ) -> dict[str, str] | None:
        """対象ごとの基本感情集合から複合感情を導出します。

        Args:
            target_events: 対象に対する感情イベントのリスト

        Returns:
            Optional[Dict[str, str]]: 複合感情の辞書、存在しない場合はNone
        """
        be_set = {e.label for e in target_events}
        if len(be_set) >= 2:  # 2つ以上の基本感情がある場合のみ
            for compound_set, compound_info in COMPOUND_EMOTIONS.items():
                if be_set == set(compound_set):
                    return compound_info
        return None

    def _commit_updates(self) -> None:
        """更新があった場合、global_mood を再計算し、dirty フラグをリセットします。"""
        if self._dirty:
            logger.debug('Global mood の再計算を開始します。')
            self.update_global_mood()
            self._dirty = False

    # ── イベント更新 ──

    def update_emotion(self, label: BasicEmotion, target: str, strength: str) -> None:
        """感情を更新します。

        Args:
            label: 基本感情のラベル
            target: 感情の対象
            strength: 感情の強さ('weak', 'medium', 'strong')
        """
        event_intensity = EVENT_STRENGTH_MAPPING.get(strength.lower(), 0.05)
        logger.debug(
            f'update_emotion: target={target}, label={label.value}, '
            f'strength={strength}, event_intensity={event_intensity}'
        )

        same_event = self._find_event(target, label)
        opposite_event = self._find_opposite_event(target, label)

        if opposite_event:
            new_opposite_intensity = (
                opposite_event.intensity
                - event_intensity * opposite_event.amplification
            )
            logger.debug(
                f'反対感情の相殺処理: {opposite_event.label.value} の'
                f'新しい強度={new_opposite_intensity:.2f}'
            )
            if new_opposite_intensity <= MIN_INTENSITY_THRESHOLD:
                surplus = -new_opposite_intensity
                self.emotions.remove(opposite_event)
                logger.debug(
                    f'反対感情イベント {opposite_event.label.value} を削除します。'
                    f'余剰分={surplus:.2f}'
                )
                if surplus > 0:
                    if same_event:
                        updated_intensity = (
                            same_event.intensity + surplus * same_event.amplification
                        )
                        same_event.intensity = min(updated_intensity, 1.0)
                        same_event.last_updated = datetime.now(DEFAULT_TIMEZONE)
                        logger.debug(
                            f'同一感情イベントを更新: 新強度={same_event.intensity:.2f}'
                        )
                    else:
                        new_emotion = Emotion(
                            label=label,
                            intensity=surplus,
                            target=target,
                            decay_rate=0.01,
                            amplification=1.0,
                            last_updated=datetime.now(DEFAULT_TIMEZONE),
                        )
                        self.emotions.append(new_emotion)
                        logger.debug(
                            f'新しい感情イベントを追加: {label.value}、'
                            f'強度={surplus:.2f}'
                        )
            else:
                opposite_event.intensity = new_opposite_intensity
                opposite_event.last_updated = datetime.now(DEFAULT_TIMEZONE)
                logger.debug(
                    f'反対感情イベントの強度を更新: {opposite_event.label.value}、'
                    f'新強度={new_opposite_intensity:.2f}'
                )
        else:
            if same_event:
                updated_intensity = (
                    same_event.intensity + event_intensity * same_event.amplification
                )
                same_event.intensity = min(updated_intensity, 1.0)
                same_event.last_updated = datetime.now(DEFAULT_TIMEZONE)
                logger.debug(
                    f'既存イベントを更新: {label.value}、'
                    f'新強度={same_event.intensity:.2f}'
                )
            else:
                new_emotion = Emotion(
                    label=label,
                    intensity=event_intensity,
                    target=target,
                    decay_rate=0.01,
                    amplification=1.0,
                    last_updated=datetime.now(DEFAULT_TIMEZONE),
                )
                self.emotions.append(new_emotion)
                logger.debug(
                    f'新しいイベントを追加: {label.value}、強度={event_intensity:.2f}'
                )
        self._dirty = True

    def update_from_llm(self, events: list[dict[str, str]]) -> None:
        """LLMからの感情イベントリストで更新します。

        Args:
            events: 感情イベントのリスト
                各イベントは {'target': str, 'label': str, 'strength': str} の形式
        """
        for event in events:
            label_str = event.get('label', '').lower()
            try:
                label = BasicEmotion(label_str)
            except ValueError:
                continue
            target = event.get('target', '').strip()
            strength = event.get('strength', 'medium')
            self.update_emotion(label, target, strength)
        self._commit_updates()

    def update_global_mood(self) -> None:
        """全体的な感情状態を更新します。"""
        sums: dict[BasicEmotion, float] = dict.fromkeys(BasicEmotion, 0.0)
        counts: dict[BasicEmotion, int] = dict.fromkeys(BasicEmotion, 0)
        for e in self.emotions:
            sums[e.label] += e.intensity
            counts[e.label] += 1
        self.global_mood = {}
        for be in BasicEmotion:
            if counts[be] > 0:
                self.global_mood[be] = sums[be] / counts[be]
            else:
                self.global_mood[be] = 0.0
        logger.debug(f'Global mood 更新: {self.global_mood}')

    # ── イベントの減衰および削除 ──

    def apply_decay(self, unit_seconds: int = 60) -> None:
        """感情の減衰を適用します。

        Args:
            unit_seconds: 減衰の単位時間(秒)
        """
        now = datetime.now(DEFAULT_TIMEZONE)
        updated_events = []
        for e in self.emotions:
            elapsed = (now - e.last_updated).total_seconds()
            units = elapsed / unit_seconds
            decay_amount = e.decay_rate * units
            new_intensity = max(0, e.intensity - decay_amount)
            logger.debug(
                f'apply_decay: {e.label.value} の経過秒数={elapsed:.2f}, '
                f'減衰量={decay_amount:.2f}, 新強度={new_intensity:.2f}'
            )
            e.intensity = new_intensity
            e.last_updated = now
            if e.intensity > MIN_INTENSITY_THRESHOLD:
                updated_events.append(e)
        self.emotions = updated_events
        self._dirty = True
        self._commit_updates()

    # ── 出力生成 ──

    def generate_output(self) -> str:
        """感情状態の出力を生成します。

        Returns:
            str: 感情状態の文字列表現
        """
        self._commit_updates()

        lines = []
        lines.append('# 基本感情')
        global_parts = []
        nonzero = False
        for be in BasicEmotion:
            total = self.global_mood.get(be, 0.0)
            if total > 0:
                nonzero = True
                cat = get_intensity_category(be, total)
                if cat == 'basic':
                    part = f'{be.japanese}: {total:.2f}'
                else:
                    alt = ALTERNATE_NAMES[be][cat]['jp']
                    part = f'{be.japanese}({alt}): {total:.2f}'
                global_parts.append(part)
        lines.append('、'.join(global_parts) if nonzero else 'ニュートラル')
        lines.append('')
        lines.append('# 対象毎の感情')
        targets = {e.target for e in self.emotions}
        for target in targets:
            target_events = [
                e for e in self.emotions if e.target == target and e.intensity > 0
            ]
            event_parts = []
            for e in target_events:
                cat = get_intensity_category(e.label, e.intensity)
                if cat == 'basic':
                    part = f'{e.label.japanese}: {e.intensity:.2f}'
                else:
                    alt = ALTERNATE_NAMES[e.label][cat]['jp']
                    part = f'{e.label.japanese}({alt}): {e.intensity:.2f}'
                event_parts.append(part)
            compound = self._derive_compound_emotion(target_events)
            compound_str = f'、複合感情: {compound["jp"]}' if compound else ''
            lines.append(f'[{target}] ' + '、'.join(event_parts) + compound_str)
        return '\n'.join(lines)

    def get_emotions(self) -> list[Emotion]:
        """感情イベントのリストを取得します。

        Returns:
            List[Emotion]: 感情イベントのリスト
        """
        return self.emotions

    def get_global_mood(self) -> dict[BasicEmotion, float]:
        """全体的な感情状態を取得します。

        Returns:
            Dict[BasicEmotion, float]: 全体的な感情状態
        """
        return self.global_mood
