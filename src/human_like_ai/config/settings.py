"""
設定管理モジュール。

このモジュールは、アプリケーション全体の設定を管理します。
環境変数からの読み込みや、デフォルト値の設定を行います。
"""

from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    """アプリケーション設定クラス。

    環境変数から設定を読み込み、デフォルト値を提供します。

    Attributes:
        openai_api_key: OpenAI APIキー
        model_name: 使用するLLMモデル名
        temperature: LLM生成時の温度パラメータ
        character_sheet_path: キャラクターシートのファイルパス
        timezone: タイムゾーン設定
    """

    openai_api_key: str
    model_name: str = 'gpt-4o'
    temperature: float = 0.0
    character_sheet_path: Path = (
        Path(__file__).parent.parent / 'data' / 'character_sheet.yaml'
    )
    timezone: str = 'Asia/Tokyo'


def get_settings() -> Settings:
    """設定インスタンスを取得します。

    環境変数から設定を読み込みます。
    OPENAI_API_KEYが設定されていない場合はダミー値を使用します。

    Returns:
        Settings: 設定インスタンス
    """
    try:
        # 環境変数から読み込み
        import os

        from dotenv import load_dotenv

        load_dotenv()

        # 必須の環境変数
        openai_api_key = os.getenv('OPENAI_API_KEY', 'dummy-api-key')

        # オプションの環境変数
        model_name = os.getenv('MODEL_NAME', 'gpt-4o')
        temperature = float(os.getenv('TEMPERATURE', '0.0'))
        character_sheet_path = os.getenv(
            'CHARACTER_SHEET_PATH',
            str(Path(__file__).parent.parent / 'data' / 'character_sheet.yaml'),
        )
        timezone = os.getenv('TIMEZONE', 'Asia/Tokyo')

        return Settings(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature,
            character_sheet_path=Path(character_sheet_path),
            timezone=timezone,
        )
    except Exception as e:
        # 環境変数が設定されていない場合などのエラーハンドリング
        raise ValueError(f'設定の読み込みに失敗しました: {e}') from e
