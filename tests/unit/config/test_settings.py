"""
設定モジュールのテスト。

このモジュールは、設定モジュールの機能をテストします。
"""

from pathlib import Path
from unittest.mock import patch

from _pytest.monkeypatch import MonkeyPatch

from src.human_like_ai.config.settings import Settings, get_settings


def test_settings_default_values() -> None:
    """デフォルト設定値のテスト。"""
    settings = Settings(openai_api_key='test-key')
    assert settings.model_name == 'gpt-4o'
    assert settings.temperature == 0.0
    assert settings.timezone == 'Asia/Tokyo'
    assert isinstance(settings.character_sheet_path, Path)


def test_settings_custom_values() -> None:
    """カスタム設定値のテスト。"""
    settings = Settings(
        openai_api_key='test-key',
        model_name='custom-model',
        temperature=0.5,
        character_sheet_path=Path('/path/to/character.yaml'),
        timezone='UTC',
    )
    assert settings.openai_api_key == 'test-key'
    assert settings.model_name == 'custom-model'
    assert settings.temperature == 0.5
    assert settings.character_sheet_path == Path('/path/to/character.yaml')
    assert settings.timezone == 'UTC'


def test_get_settings_with_env_vars(mock_env_vars: None) -> None:
    """環境変数からの設定読み込みテスト。"""
    settings = get_settings()
    assert settings.openai_api_key == 'test-api-key'
    assert settings.model_name == 'test-model'
    assert settings.temperature == 0.0
    assert settings.timezone == 'UTC'


def test_get_settings_with_missing_env_vars(monkeypatch: MonkeyPatch) -> None:
    """環境変数が不足している場合のテスト。"""
    # 環境変数をクリア
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    # dotenvのload_dotenvをモック化して.envファイルを読み込まないようにする
    with patch('dotenv.load_dotenv'):
        # os.getenvをモック化してOPENAI_API_KEYが存在しないようにする
        with patch('os.getenv') as mock_getenv:

            def mock_getenv_side_effect(
                key: str, default: str | None = None
            ) -> str | None:
                if key == 'OPENAI_API_KEY':
                    return 'dummy-api-key'
                return default

            mock_getenv.side_effect = mock_getenv_side_effect
            settings = get_settings()
            assert settings.openai_api_key == 'dummy-api-key'


def test_mock_settings_fixture(mock_settings: Settings) -> None:
    """モック設定フィクスチャのテスト。"""
    assert mock_settings.openai_api_key == 'test-api-key'
    assert mock_settings.model_name == 'test-model'
    assert mock_settings.temperature == 0.0
    assert mock_settings.timezone == 'UTC'
