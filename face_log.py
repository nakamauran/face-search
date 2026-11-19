"""ログ出力（face-search.log）と標準ストリームの安全化.

exe 版はコンソールが出ないため、起動・処理・エラー時の例外全文を
日時付きでログファイルに残す。既定は exe と同じフォルダの
face-search.log（書けなければ一時フォルダへフォールバックする。
log_path() は実際に使っている場所を返す）。

コンソールなし exe では sys.stdout / sys.stderr が None になる。
InsightFace は初回モデルDL時に tqdm で stderr へ書くため、そのままだと
'NoneType' object has no attribute 'write' で落ちる。main() の最初に
ensure_std_streams() を呼び、None ならログファイルへ向ける差し替え
ストリームに置き換える。tqdm の進捗は '\r' で上書きされるため、
差し替えストリームは途中経過を破棄し、最終行だけログに残す。
"""

from __future__ import annotations

import io
import logging
import sys
import tempfile
import traceback
from pathlib import Path

from paths import exe_dir

LOG_FILENAME = "face-search.log"

# 解決済みのログパス（setup_logging() と ensure_std_streams() で共有）。
_resolved_log_path: Path | None = None
_std_streams_installed = False


def _resolve_log_path() -> Path:
    """実際にログを書く場所を決める（結果をキャッシュする）.

    既定は exe と同じフォルダ。書けなければ一時フォルダへフォールバック
    する。どちらにも書けない場合は一時フォルダのパスを返す
    （呼び出し側が NullHandler に落とす）。
    """
    global _resolved_log_path
    if _resolved_log_path is not None:
        return _resolved_log_path
    preferred = exe_dir() / LOG_FILENAME
    try:
        # 書き込み可能か実際に試す（無ければ作る）。
        with open(preferred, "a", encoding="utf-8"):
            pass
        _resolved_log_path = preferred
    except OSError:
        _resolved_log_path = Path(tempfile.gettempdir()) / LOG_FILENAME
    return _resolved_log_path


def log_path() -> Path:
    """ログファイルのパスを返す（実際に使っている場所）."""
    return _resolve_log_path()


def setup_logging() -> logging.Logger:
    """face-search.log へのログ出力を設定してロガーを返す.

    二重に呼んでもハンドラは増やさない。どこにも書けない場合だけ
    NullHandler に落とす。
    """
    logger = logging.getLogger("face_search")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        handler = logging.FileHandler(log_path(), encoding="utf-8")
    except OSError:
        logger.addHandler(logging.NullHandler())
        return logger
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    """設定済みのロガーを返す（未設定なら設定する）."""
    return setup_logging()


def log_traceback(logger: logging.Logger, message: str) -> str:
    """現在の例外の traceback 全文をログに書き込み、その全文を返す."""
    tb = traceback.format_exc()
    logger.error("%s\n%s", message, tb)
    return tb


class _LogStreamWriter:
    """sys.stdout / sys.stderr の差し替え先。書かれた内容をログファイルへ残す.

    io.TextIOBase は継承しない（encoding 属性が読み取り専用で差し替えら
    れないため）。tqdm が使う write / flush / isatty / encoding だけを
    持つ、最小のファイル風オブジェクト。

    - print() の出力は行単位で face-search.log に追記する。
    - tqdm の進捗は '\r' で上書きされるため、途中経過は破棄し '\n' で
      終わる最終行だけ出す（ログが進捗行で埋まらないようにする）。
    - 書けない場合は黙って捨てる。このライター自体は絶対に例外を出さない。
    """

    # tqdm などが参照する属性。
    encoding = "utf-8"
    errors = "replace"

    def __init__(self, path: Path, prefix: str = ""):
        self._path = path
        self._prefix = prefix
        self._buf = ""
        self.closed = False

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        self.flush()
        self.closed = True

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        # 非TTYであることを示す。進捗行の抑制は _drain() の '\r' 破棄で行う。
        return False

    def write(self, s) -> int:
        if s is None:
            return 0
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        self._drain()
        return len(s)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def _emit(self, line: str) -> None:
        line = line.strip("\r\n")
        if not line:
            return
        try:
            with open(self._path, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"{self._prefix}{line}\n")
        except OSError:
            pass  # 書けなければ捨てる（例外は出さない）

    def _drain(self) -> None:
        """バッファ内の完成した行を出す.

        '\\r' は進捗表示の上書きなので、それまでの内容は破棄する。
        ただし "\\r\\n" は改行として扱う。バッファ末尾の '\\r' は
        次に '\\n' が来るか分からないので、次回まで保持する。
        """
        buf = self._buf
        while True:
            nl = buf.find("\n")
            cr = buf.find("\r")
            if nl == -1 and cr == -1:
                break
            if cr != -1 and (nl == -1 or cr < nl):
                if cr + 1 < len(buf) and buf[cr + 1] == "\n":
                    # "\r\n" 改行：行として出す
                    self._emit(buf[:cr])
                    buf = buf[cr + 2:]
                elif cr + 1 == len(buf):
                    # バッファ末尾の '\r'：次回まで待つ
                    break
                else:
                    # tqdm 式の '\r' 上書き：それまでの内容は破棄
                    buf = buf[cr + 1:]
                continue
            # '\n' が先：そこまでを1行として出す
            self._emit(buf[:nl])
            buf = buf[nl + 1:]
        self._buf = buf

    def flush(self) -> None:
        # 完成した行だけ出す。未完成の行（tqdm の進捗など）は出さない。
        # tqdm は write のたびに flush() を呼ぶため、ここで残りを出すと
        # ログが進捗行で埋まる。
        self._drain()

    def fileno(self):
        raise io.UnsupportedOperation("fileno")


def ensure_std_streams() -> None:
    """sys.stdout / sys.stderr が None なら差し替える。

    PyInstaller console=False の exe では sys.stdout / sys.stderr が
    None になる。InsightFace の初回モデルDL時に tqdm が stderr へ書こう
    として AttributeError になるため、起動時（main() の最初、
    InsightFace に触れる前）に呼ぶこと。既に有効なストリームは触らない。
    """
    global _std_streams_installed
    if _std_streams_installed:
        return
    path = log_path()
    if sys.stdout is None:
        sys.stdout = _LogStreamWriter(path)
    if sys.stderr is None:
        sys.stderr = _LogStreamWriter(path, prefix="[stderr] ")
    _std_streams_installed = True
