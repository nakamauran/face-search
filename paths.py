"""exe化（PyInstaller）でも壊れないパス解決ヘルパー.

PyInstaller で固めると `__file__` は exe 内部の一時展開先を指し、
カレントディレクトリは「exe を起動した場所」になるため、
「スクリプトのある場所」としては使えない。このモジュールはその違いを吸収する。

- exe_dir(): exe なら exe のあるフォルダ、通常実行ならこのファイルのあるフォルダ
- default_index_path(): index.npz の既定の保存先（写真フォルダ直下）
- fallback_index_path(): 写真フォルダに書けない場合の退避先（exe のあるフォルダ）
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def is_frozen() -> bool:
    """PyInstaller で固めた exe として動いているかどうか."""
    return bool(getattr(sys, "frozen", False))


def exe_dir() -> Path:
    """実行ファイルのあるフォルダを返す.

    exe の場合: `sys.executable` のあるフォルダ。
    通常実行の場合: このファイルのあるフォルダ（プロジェクトフォルダ）。
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(relative: str) -> Path:
    """PyInstaller で同梱したデータファイルの実行時パスを返す.

    onefile の場合、同梱ファイルは一時フォルダ（`sys._MEIPASS`）に
    展開されるため、そこを経由する。現状このプロトタイプに同梱する
    データファイルはないが、将来アイコン等を追加するとき用に用意する。
    """
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative
    return Path(__file__).resolve().parent / relative


def is_writable(directory: str | os.PathLike) -> bool:
    """フォルダに書き込めるかどうか（実書き込みはしない）."""
    return os.access(str(directory), os.W_OK)


def _safe_folder_name(folder: str | os.PathLike) -> str:
    """フォルダ名をファイル名に使える形にする（空白・記号を _ にする）."""
    name = Path(folder).name or "photos"
    name = re.sub(r"[^0-9A-Za-zぁ-んァ-ン一-鿿_.-]+", "_", name)
    return name[:60] or "photos"


def default_index_path(photo_folder: str | os.PathLike) -> Path:
    """index.npz の既定の保存先: 写真フォルダ直下."""
    return Path(photo_folder) / "index.npz"


def fallback_index_path(photo_folder: str | os.PathLike) -> Path:
    """写真フォルダに書けない場合の index.npz の退避先: exe のあるフォルダ."""
    return exe_dir() / f"index_{_safe_folder_name(photo_folder)}.npz"


def choose_index_path(photo_folder: str | os.PathLike) -> tuple[Path, str]:
    """index.npz の保存先を決める.

    戻り値: (保存先パス, 理由メモ)。写真フォルダが書き込み可能ならそこ、
    書けないなら exe のあるフォルダへ退避する（退避先も書けない場合は
    後で保存時にエラーになる）。
    """
    folder = Path(photo_folder)
    if is_writable(folder):
        return default_index_path(folder), ""
    fallback = fallback_index_path(folder)
    note = (
        f"写真フォルダに書き込めないため、index.npz は exe のあるフォルダに保存します: {fallback}"
    )
    return fallback, note
