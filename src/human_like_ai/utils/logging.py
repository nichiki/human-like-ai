"""
ロギングユーティリティモジュール。

このモジュールは、アプリケーション全体で使用するロギング機能を提供します。
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from src.human_like_ai.config.settings import Settings, get_settings


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
    log_to_console: bool = True,
    settings: Settings | None = None,
) -> logging.Logger:
    """ロガーをセットアップします。

    Args:
        name: ロガー名
        level: ログレベル
        log_file: ログファイルパス。指定されない場合はログファイルに出力しない。
        log_to_console: コンソールにログを出力するかどうか
        settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。

    Returns:
        logging.Logger: セットアップされたロガー
    """
    settings = settings or get_settings()

    # ロガーの取得
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 既存のハンドラをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # フォーマッタの作成
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # コンソールへの出力
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ファイルへの出力
    if log_file:
        # ログディレクトリの作成
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_default_logger(settings: Settings | None = None) -> logging.Logger:
    """デフォルトのロガーを取得します。

    Args:
        settings: アプリケーション設定。指定されない場合はデフォルト設定を使用。

    Returns:
        logging.Logger: デフォルトのロガー
    """
    settings = settings or get_settings()

    # ログディレクトリの設定
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # 日付ベースのログファイル名
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = log_dir / f'human_like_ai_{today}.log'

    return setup_logger(
        name='human_like_ai',
        level=logging.INFO,
        log_file=str(log_file),
        log_to_console=True,
        settings=settings,
    )


# モジュールレベルのロガー
logger = get_default_logger()
