"""診断ツール — 顔類似検索プロトタイプの環境・原因切り分け用.

使い方:
    python diagnose.py <写真フォルダ> [クエリ画像] [--detect-first]

調べること: Python バージョン・実行形態（exe か Python か）、
insightface / onnxruntime / opencv / numpy / Pillow のバージョン、
~/.insightface/models/buffalo_l の有無・サイズ、dist フォルダの内容、
写真フォルダと exe のあるフォルダの書き込み権限、写真フォルダの走査
（画像枚数・深さ・非 ASCII パス・対応外の拡張子）、画像の読み込み
テスト（新ローダーと cv2.imread の読み比べ）。--detect-first を付けると
最初の1枚で実際に顔検出を走らせる（集合写真の確認用。モデル読み込み＋
検出で数十秒かかる。初回はモデルDLも走る）。

結果は画面と face-search.log の両方に出す。終了コード: 大きな問題が
無ければ 0、読み込み失敗などがあれば 1。exe 版のユーザーは Python が
無いのでこのツールは使えない。その場合は face-search.log を送ってもらう。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_log import get_logger, log_path  # noqa: E402
from indexer import IMAGE_EXTENSIONS, create_face_app, iter_image_files  # noqa: E402
from paths import exe_dir, is_frozen  # noqa: E402
from search_core import ImageReadError, read_image_bgr  # noqa: E402

MODEL_FILES = ["det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx", "2d106det.onnx", "genderage.onnx"]


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未インストール"


def _fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


class Reporter:
    """行を画面とログの両方に出す."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.logger = get_logger()
        self.problems = 0

    def line(self, text: str = "", problem: bool = False) -> None:
        print(text)
        self.lines.append(text)
        self.logger.info("[diagnose] %s", text)
        if problem:
            self.problems += 1

    def header(self, text: str) -> None:
        self.line()
        self.line(f"=== {text} ===")


def _check_model(rep: Reporter) -> None:
    rep.header("InsightFace モデル (buffalo_l)")
    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    rep.line(f"場所: {model_dir}")
    if not model_dir.is_dir():
        rep.line(
            "モデルフォルダがありません。初回起動時に約281MBをダウンロードします。"
            "ネットワーク接続と空き容量（500MB以上）を確認してください。",
            problem=True,
        )
        return
    total = 0
    for name in MODEL_FILES:
        p = model_dir / name
        if p.is_file():
            size = p.stat().st_size
            total += size
            rep.line(f"  {name}: {_fmt_size(size)}")
        else:
            rep.line(f"  {name}: ありません", problem=True)
    rep.line(f"  合計: {_fmt_size(total)}（目安: 約326MB）")


def _check_dist(rep: Reporter) -> None:
    rep.header("exe / dist フォルダ")
    rep.line(f"exe として実行中: {'はい' if is_frozen() else 'いいえ（Python 版）'}")
    rep.line(f"exe のあるフォルダ: {exe_dir()}")
    exe_path = exe_dir() / "face-search.exe"
    rep.line(f"face-search.exe: {'あります' if exe_path.is_file() else 'ありません'}")
    dlls = sorted(exe_dir().glob("onnxruntime*.dll"))
    if dlls:
        rep.line(f"onnxruntime DLL: {len(dlls)}件（例: {dlls[0].name}）")
    else:
        rep.line("onnxruntime DLL: exe のあるフォルダにありません（Python 版では正常）")
    dist = Path(__file__).resolve().parent / "dist" / "face-search"
    if dist.is_dir():
        files = [p for p in dist.rglob("*") if p.is_file()]
        rep.line(f"dist/face-search: {len(files)}ファイルあります")
        if not (dist / "face-search.exe").is_file():
            rep.line("  face-search.exe が dist にありません", problem=True)
    # 書き込み権限
    import os

    for label, folder in [("写真フォルダの代わりに exe のあるフォルダ", exe_dir())]:
        rep.line(
            f"{label}の書き込み権限: "
            f"{'書き込めます' if os.access(str(folder), os.W_OK) else '書き込めません'}"
        )


def _check_folder(rep: Reporter, folder: Path) -> list[Path]:
    rep.header(f"写真フォルダ: {folder}")
    if not folder.is_dir():
        rep.line("フォルダが見つかりません。", problem=True)
        return []
    import os

    rep.line(
        "書き込み権限: "
        f"{'書き込めます' if os.access(str(folder), os.W_OK) else '書き込めません（index.npz は exe のあるフォルダへ退避します）'}"
    )
    files = iter_image_files(folder)
    rep.line(f"画像ファイル: {len(files)}枚（サブフォルダも再帰的に走査）")
    if files:
        depths = [len(p.relative_to(folder).parts) for p in files]
        rep.line(f"  最も深い階層: {max(depths)}階層（1=トップ直下）")
        non_ascii = [p for p in files if any(ord(c) > 127 for c in str(p))]
        rep.line(f"  日本語など非 ASCII を含むパス: {len(non_ascii)}枚")
        if non_ascii:
            rep.line(f"    例: {non_ascii[0]}")
    from collections import Counter

    suffixes: Counter[str] = Counter()
    for p in folder.rglob("*"):
        if p.is_file():
            suffixes[p.suffix.lower()] += 1
    unsupported = {
        s: n for s, n in suffixes.items() if s and s not in IMAGE_EXTENSIONS and s != ".npz"
    }
    if unsupported:
        rep.line(f"  対応外の拡張子: {unsupported}（.heic/.heif は pillow-heif が無いと読めません）")
    return files


def _check_load(rep: Reporter, path: Path, label: str) -> bool:
    rep.header(f"読み込みテスト: {label} ({path.name})")
    ok_new = False
    try:
        img = read_image_bgr(path)
        rep.line(f"  新ローダー（read_image_bgr）: 読めました（{img.shape[1]}x{img.shape[0]}）")
        ok_new = True
    except ImageReadError as e:
        rep.line(f"  新ローダー（read_image_bgr）: 読めませんでした: {e}", problem=True)
    try:
        import cv2

        old = cv2.imread(str(path))
        rep.line(f"  cv2.imread: {'読めました' if old is not None else '読めませんでした（None）'}")
        if old is None and ok_new:
            rep.line(
                "  → 日本語パスなどで cv2.imread が読めず、新ローダーでは読める状態です。"
                "この画像は修正版ではインデックスに入ります。"
            )
    except ImportError:
        rep.line("  cv2.imread: OpenCV が無いため比較できません")
    return ok_new


def _check_detect_first(rep: Reporter, files: list[Path]) -> None:
    rep.header("顔検出テスト（最初の1枚、集合写真の確認用）")
    rep.line("モデルを読み込んでいます。初回はダウンロードが入ります...")
    try:
        app = create_face_app()
    except Exception as e:
        rep.line(f"モデル読み込みに失敗しました: {e}", problem=True)
        return
    path = files[0]
    try:
        img = read_image_bgr(path)
    except ImageReadError as e:
        rep.line(f"画像を読めませんでした: {e}", problem=True)
        return
    faces = list(app.get(img))
    rep.line(f"画像: {path.name}（{img.shape[1]}x{img.shape[0]}）")
    rep.line(f"検出した顔: {len(faces)}人")
    sizes = []
    for f in faces:
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        sizes.append((x2 - x1, y2 - y1, float(f.det_score)))
    sizes.sort(key=lambda s: -s[0])
    for w, h, s in sizes[:10]:
        rep.line(f"  顔 {w:.0f}x{h:.0f}px 検出スコア {s:.2f}")
    for min_size in (50, 20, 16):
        kept = [s for s in sizes if s[0] >= min_size and s[1] >= min_size and s[2] >= 0.5]
        rep.line(f"  min_face_size={min_size}: 残る顔 {len(kept)}/{len(faces)}人")
    rep.line(
        "  集合写真では顔が 20px 前後と小さいことが多く、50px 除外だと"
        "ほとんど残りません。既定は 20px です。"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="顔類似検索プロトタイプの診断ツール")
    parser.add_argument("folder", help="写真フォルダ")
    parser.add_argument("query", nargs="?", default=None, help="クエリ画像（省略可）")
    parser.add_argument(
        "--detect-first",
        action="store_true",
        help="最初の1枚で実際に顔検出を走らせる（集合写真の確認用。数十秒かかる）",
    )
    args = parser.parse_args(argv)

    rep = Reporter()
    rep.line(f"診断を開始します。ログ: {log_path()}")
    rep.header("環境")
    rep.line(f"Python: {platform.python_version()} ({sys.executable})")
    rep.line(f"OS: {platform.system()} {platform.release()}")
    for name in ("insightface", "onnxruntime", "opencv-python-headless", "numpy", "Pillow", "PyInstaller"):
        rep.line(f"{name}: {_pkg_version(name)}")

    _check_model(rep)
    _check_dist(rep)
    files = _check_folder(rep, Path(args.folder))
    ok = True
    if files:
        ok = _check_load(rep, files[0], "写真フォルダの最初の1枚")
    if args.query:
        q = Path(args.query)
        if q.is_file():
            ok = _check_load(rep, q, "クエリ画像") and ok
        else:
            rep.line(f"クエリ画像が見つかりません: {q}", problem=True)
            ok = False
    if args.detect_first and files:
        _check_detect_first(rep, files)

    rep.header("まとめ")
    if rep.problems == 0:
        rep.line("問題は見つかりませんでした。")
    else:
        rep.line(f"問題が {rep.problems}件あります。上の内容を確認してください。")
        rep.line("exe 版で「インデックス作成に失敗しました」が出る場合は、")
        rep.line("face-search.log もあわせて確認してください。")
    rep.line(f"診断結果は face-search.log にも追記しました。ログ: {log_path()}")
    return 1 if (rep.problems > 0 or not ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
