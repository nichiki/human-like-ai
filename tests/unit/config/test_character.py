"""
キャラクター設定モジュールのテスト。

このモジュールは、キャラクター設定モジュールの機能をテストします。
"""

from pathlib import Path

import pytest
import yaml

from src.human_like_ai.config.character import CharacterLoader


def test_character_loader_init(mock_settings) -> None:
    """CharacterLoaderの初期化テスト。"""
    loader = CharacterLoader(mock_settings)
    assert loader.settings == mock_settings
    assert loader._character_data == {}


def test_character_loader_load(test_character_file, mock_settings) -> None:
    """キャラクター設定の読み込みテスト。"""
    # 設定のパスを一時ファイルに変更
    mock_settings.character_sheet_path = test_character_file

    # ローダーの作成と読み込み
    loader = CharacterLoader(mock_settings)
    data = loader.load()

    # データの検証
    assert 'basic_info' in data
    assert data['basic_info']['name'] == '北条 楓'
    assert 'personality' in data
    assert 'traits' in data['personality']
    assert '明るい' in data['personality']['traits']


def test_character_loader_load_file_not_found() -> None:
    """存在しないファイルの読み込みテスト。"""
    loader = CharacterLoader()
    loader.settings.character_sheet_path = Path('/non/existent/path.yaml')

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_character_loader_get_character_data(
    test_character_file, mock_settings
) -> None:
    """キャラクターデータの取得テスト。"""
    # 設定のパスを一時ファイルに変更
    mock_settings.character_sheet_path = test_character_file

    # ローダーの作成
    loader = CharacterLoader(mock_settings)

    # データの取得（初回はロードが実行される）
    data = loader.get_character_data()
    assert 'basic_info' in data
    assert data['basic_info']['name'] == '北条 楓'

    # データの変更
    loader._character_data = {'test': 'data'}

    # 再度取得（既にロード済みなのでロードは実行されない）
    data = loader.get_character_data()
    assert data == {'test': 'data'}


def test_character_loader_get_character_text(
    test_character_file, mock_settings
) -> None:
    """キャラクターテキストの取得テスト。"""
    # 設定のパスを一時ファイルに変更
    mock_settings.character_sheet_path = test_character_file

    # ローダーの作成
    loader = CharacterLoader(mock_settings)

    # テキストの取得
    text = loader.get_character_text()
    assert isinstance(text, str)

    # テキストからデータを復元して検証
    data = yaml.safe_load(text)
    assert 'basic_info' in data
    assert data['basic_info']['name'] == '北条 楓'
