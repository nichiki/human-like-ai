"""
キャラクター設定管理モジュール。

このモジュールは、キャラクターシート(YAML)の読み込みと管理を行います。
"""

from pathlib import Path
from typing import Any

import yaml

from src.human_like_ai.config.settings import Settings


class CharacterLoader:
    """キャラクター設定ローダークラス。

    YAMLファイルからキャラクター設定を読み込み、提供します。

    Attributes:
        settings: アプリケーション設定
        _character_data: 読み込まれたキャラクターデータ
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """初期化メソッド。

        Args:
            settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。
        """
        from src.human_like_ai.config.settings import get_settings

        self.settings = settings or get_settings()
        self._character_data: dict[str, Any] = {}

    def load(self, file_path: Path | None = None) -> dict[str, Any]:
        """キャラクター設定を読み込みます。

        Args:
            file_path: キャラクターシートのファイルパス。
                       指定されない場合は設定のパスを使用。

        Returns:
            Dict[str, Any]: 読み込まれたキャラクターデータ

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            yaml.YAMLError: YAMLの解析エラーが発生した場合
        """
        path = file_path or self.settings.character_sheet_path

        try:
            with open(path, encoding='utf-8') as f:
                self._character_data = yaml.safe_load(f)
            return self._character_data
        except FileNotFoundError:
            raise FileNotFoundError(
                f'キャラクターシートが見つかりません: {path}'
            ) from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f'キャラクターシートの解析エラー: {e}') from e

    def get_character_text(self) -> str:
        """キャラクターデータをテキスト形式で取得します。

        Returns:
            str: キャラクターデータのYAMLテキスト表現
        """
        if not self._character_data:
            self.load()
        return yaml.dump(self._character_data, allow_unicode=True)

    def get_character_data(self) -> dict[str, Any]:
        """キャラクターデータを取得します。

        Returns:
            Dict[str, Any]: キャラクターデータ
        """
        if not self._character_data:
            self.load()
        return self._character_data
