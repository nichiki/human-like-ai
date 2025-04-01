"""
テスト用の共通フィクスチャとヘルパー関数。

このモジュールは、テスト全体で使用するフィクスチャやヘルパー関数を提供します。
"""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from src.human_like_ai.config.settings import Settings


class MockSettings(Settings):
    """テスト用のモック設定クラス。

    実際のAPIキーを使用せず、テスト用の設定を提供します。
    """

    openai_api_key: str = 'test-api-key'
    model_name: str = 'test-model'
    temperature: float = 0.0
    character_sheet_path: Path = (
        Path(__file__).parent / 'data' / 'test_character_sheet.yaml'
    )
    timezone: str = 'UTC'


@pytest.fixture
def mock_settings() -> MockSettings:
    """テスト用のモック設定を提供します。

    Returns:
        MockSettings: テスト用のモック設定
    """
    return MockSettings()


@pytest.fixture
def test_character_data() -> dict[str, Any]:
    """テスト用のキャラクターデータを提供します。

    Returns:
        Dict[str, Any]: テスト用のキャラクターデータ
    """
    return {
        'basic_info': {
            'name': '北条 楓',
            'gender': '女性',
            'birth_date': '1999-01-15',
            'blood_type': 'B型',
            'height': 161,
            'weight': 52,
            'zodiac': '山羊座',
            'birthplace': '東京都港区',
            'residence': '東京都中野区',
        },
        'personality': {
            'traits': ['明るい', '協調性がある', '好奇心旺盛'],
            'likes': ['カフェ巡り', '海外ドラマ', '新しいもの'],
            'dislikes': ['退屈な時間', '閉鎖的な環境', '計画性のなさ'],
        },
        'background': {
            'occupation': 'IT系スタートアップ企業の広報・マーケティング担当',
            'education': '私立大学の経営学部卒業',
            'family': '両親と弟が実家に住んでいる',
            'hobbies': ['カフェ巡り', '写真撮影', 'SNS更新'],
        },
        'appearance': {
            'hair': '肩くらいの長さのストレートヘア、茶色に染めている',
            'eyes': '丸くて大きい、明るい茶色',
            'style': 'カジュアルだがトレンドを意識したファッション',
            'features': '笑顔が特徴的、よく笑う',
        },
    }


@pytest.fixture
def test_character_yaml(test_character_data: dict[str, Any]) -> str:
    """テスト用のキャラクターYAMLを提供します。

    Args:
        test_character_data: テスト用のキャラクターデータ

    Returns:
        str: テスト用のキャラクターYAML
    """
    import yaml

    return yaml.dump(test_character_data, allow_unicode=True)


@pytest.fixture
def test_character_file(tmp_path: Path, test_character_yaml: str) -> Path:
    """テスト用のキャラクターファイルを作成します。

    Args:
        tmp_path: 一時ディレクトリのパス
        test_character_yaml: テスト用のキャラクターYAML

    Returns:
        Path: テスト用のキャラクターファイルのパス
    """
    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / 'test_character_sheet.yaml'
    file_path.write_text(test_character_yaml, encoding='utf-8')
    return file_path


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """テスト用の環境変数を設定します。

    テスト終了後に元の環境変数に戻します。

    Yields:
        None
    """
    # 元の環境変数を保存
    original_env = os.environ.copy()

    # テスト用の環境変数を設定
    os.environ['OPENAI_API_KEY'] = 'test-api-key'
    os.environ['MODEL_NAME'] = 'test-model'
    os.environ['TEMPERATURE'] = '0.0'
    os.environ['TIMEZONE'] = 'UTC'

    yield

    # 元の環境変数に戻す
    os.environ.clear()
    os.environ.update(original_env)
