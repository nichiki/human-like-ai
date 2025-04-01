#!/usr/bin/env python
"""
人間らしいAIエージェントのエントリーポイント。

このスクリプトは、human-like-aiパッケージを使用して
チャットボットを起動し、ユーザーとの対話を処理します。
"""

import argparse
import sys

from src.human_like_ai.config.settings import Settings, get_settings
from src.human_like_ai.core.agent import AgentFactory
from src.human_like_ai.utils.logging import get_default_logger


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します。

    Returns:
        argparse.Namespace: 解析された引数
    """
    parser = argparse.ArgumentParser(description='人間らしいAIエージェント')
    parser.add_argument(
        '--model',
        type=str,
        help='使用するLLMモデル名(例: gpt-4o)',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='LLM生成時の温度パラメータ(0.0〜1.0)',
    )
    parser.add_argument(
        '--character-sheet',
        type=str,
        help='キャラクターシートのファイルパス',
    )
    return parser.parse_args()


def create_settings_from_args(args: argparse.Namespace) -> Settings | None:
    """コマンドライン引数から設定を作成します。

    Args:
        args: コマンドライン引数

    Returns:
        Optional[Settings]: 作成された設定。引数が指定されていない場合はNone。
    """
    # 引数が指定されていない場合はNoneを返す
    if not any(vars(args).values()):
        return None

    # デフォルト設定を取得
    settings = get_settings()

    # 引数で指定された値で設定を上書き
    if args.model:
        settings.model_name = args.model
    if args.temperature is not None:
        settings.temperature = args.temperature
    if args.character_sheet:
        settings.character_sheet_path = args.character_sheet

    return settings


def main() -> None:
    """メイン関数。"""
    # コマンドライン引数の解析
    args = parse_args()
    settings = create_settings_from_args(args)
    logger = get_default_logger(settings)

    try:
        # エージェントの作成
        logger.info('エージェントを作成します。')
        agent = AgentFactory.create_agent(settings)
        logger.info('エージェントの作成が完了しました。')

        # チャットループ
        print('チャットボットが起動しました。終了するには "exit" と入力してください。')
        while True:
            try:
                user_input = input('> ')
                if user_input.lower() in ('exit', 'quit', 'q'):
                    break

                # エージェントからの応答を取得
                response = agent.process_input(user_input)

                # 応答を表示
                print(response)

            except KeyboardInterrupt:
                print('\nプログラムを終了します。')
                break
            except Exception as e:
                logger.error(f'エラーが発生しました: {e}', exc_info=True)
                print(f'エラーが発生しました: {e}')

    except Exception as e:
        logger.error(f'初期化中にエラーが発生しました: {e}', exc_info=True)
        print(f'初期化中にエラーが発生しました: {e}')
        sys.exit(1)

    logger.info('プログラムを終了します。')
    print('プログラムを終了します。')


if __name__ == '__main__':
    main()
