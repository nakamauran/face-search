"""顔類似検索プロトタイプ — メインGUI（tkinterのみ）.

使い方:
  1. 左で写真フォルダを選び「インデックス作成」を押す
     （初回は InsightFace buffalo_l のモデルをダウンロードします）
  2. 中央でクエリ画像を選ぶ。複数人写っていたら、探す顔を選ぶ
  3. しきい値と表示件数を調整して「検索」を押す
  4. 結果をクリックすると右に拡大表示される
     （ヒット顔はオレンジ枠、他の顔はグレー枠）
  5. コピーしたい結果にチェックを入れて保存先を選び、コピーする
     元ファイルは残ります。同名ファイルは連番になります。

ローカル専用。写真は外部にアップロードしません。
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from face_log import ensure_std_streams, get_logger, log_path, log_traceback, setup_logging
from indexer import PROVIDER_CPU, PROVIDER_LABELS, _preload_cuda_dlls, build_index, create_face_app
from paths import choose_index_path
from search_core import (
    FaceIndex,
    ImageReadError,
    SearchResult,
    copy_files,
    face_box,
    face_det_score,
    face_embedding,
    get_faces_from_image,
    load_index,
    read_image_bgr,
    search,
)

# 顔枠の色（モックアップと合わせる）
HIT_COLOR = (255, 140, 0)      # オレンジ
OTHER_COLOR = (150, 150, 150)  # グレー
QUERY_COLOR = (255, 140, 0)

# バージョン。タイトルバーとステータスバーに出す。修正時はここだけ上げる。
VERSION = "v0.7.0 (2026-11-19)"


# 画像ヘルパー

def load_pil_image(path: str) -> Image.Image | None:
    """画像ファイルを PIL 画像（RGB）で読む. 読めなければ None."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception:
        return None


def draw_boxes(
    pil_img: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    hit_index: int | None = None,
) -> Image.Image:
    """画像に顔枠を描く. ヒット顔だけオレンジ、それ以外はグレー."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = HIT_COLOR if i == hit_index else OTHER_COLOR
        width = 4 if i == hit_index else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def resize_to_fit(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """縦横比を保って指定サイズに収める."""
    img = pil_img.copy()
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


def scale_boxes(
    boxes: list[tuple[float, float, float, float]], scale_x: float, scale_y: float
) -> list[tuple[float, float, float, float]]:
    return [
        (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        for (x1, y1, x2, y2) in boxes
    ]


# メインアプリ

class FaceSearchApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"顔類似検索プロトタイプ {VERSION}")
        self.geometry("1180x760")
        self.logger = get_logger()

        # 状態
        self.photo_folder: str = ""
        self.index_path: str = ""
        self.index: FaceIndex | None = None
        self.face_app = None  # InsightFace は初回使用時に遅延初期化
        self._face_app_det_size: tuple[int, int] | None = None
        self.face_app_loading = False

        self.query_path: str = ""
        self.query_pil: Image.Image | None = None
        self.query_faces: list = []          # InsightFace の Face オブジェクト
        self.query_embeddings: list[np.ndarray] = []
        self.selected_query_face: int = 0

        # 並列数・作り直し（ウィジェットは _build_left で作る）
        self.workers_var: tk.StringVar | None = None
        self.rebuild_var: tk.BooleanVar | None = None

        self.results: list[SearchResult] = []
        self.checked: dict[int, bool] = {}   # face_index -> コピー対象か
        self.check_vars: dict[int, tk.BooleanVar] = {}
        self.selected_result: int = -1       # results 内の位置
        self.dest_folder: str = ""

        self._thumb_refs: dict[int, ImageTk.PhotoImage] = {}
        self._query_thumb: ImageTk.PhotoImage | None = None
        self._detail_photo: ImageTk.PhotoImage | None = None
        self._busy = False

        self._build_ui()

    # UI 構築
    def _build_ui(self) -> None:
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(main, width=260)
        center = ttk.Frame(main)
        right = ttk.Frame(main, width=330)
        main.add(left, weight=0)
        main.add(center, weight=1)
        main.add(right, weight=0)

        self._build_left(left)
        self._build_center(center)
        self._build_right(right)

        # ステータスバー（左：メッセージ、右：バージョン）
        status_bar = ttk.Frame(self)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="準備完了。写真フォルダを選んでインデックスを作成してください。")
        ttk.Label(status_bar, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self.version_var = tk.StringVar(value=VERSION)
        ttk.Label(status_bar, textvariable=self.version_var, relief=tk.SUNKEN, anchor=tk.E).pack(
            side=tk.RIGHT
        )

    # 左パネル
    def _build_left(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="写真フォルダ", font=("", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        self.folder_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.folder_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="選ぶ", command=self.on_choose_folder, width=6).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(parent, text="統計", font=("", 10, "bold")).pack(anchor=tk.W, pady=(12, 2))
        self.stats_var = tk.StringVar(value="写真: -\n顔: -")
        ttk.Label(parent, textvariable=self.stats_var, justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Label(parent, text="インデックス", font=("", 10, "bold")).pack(anchor=tk.W, pady=(12, 2))
        self.progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X)
        self.progress_var = tk.StringVar(value="未作成")
        ttk.Label(parent, textvariable=self.progress_var).pack(anchor=tk.W)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        self.btn_build = ttk.Button(btn_row, text="インデックス作成", command=self.on_build_index)
        self.btn_build.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_row, text="開く", command=self.on_open_index, width=6).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(parent, text="モデル: InsightFace buffalo_l", wraplength=240).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(parent, text="検出 SCRFD / 特徴量 ArcFace 512次元", wraplength=240).pack(anchor=tk.W)
        ttk.Label(parent, text="除外: 20px未満の顔 / 検出スコア 0.5未満", wraplength=240).pack(anchor=tk.W)

        ttk.Label(parent, text="検出サイズ", font=("", 10, "bold")).pack(anchor=tk.W, pady=(12, 2))
        self.det_size_var = tk.StringVar(value="640")
        det_combo = ttk.Combobox(
            parent, textvariable=self.det_size_var, state="readonly", width=34
        )
        det_combo["values"] = ["640", "1024", "1280"]
        det_combo.current(0)
        det_combo.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(
            parent,
            text="640=速い、1280=小さい顔も拾うが遅い。集合写真は1280がおすすめ。",
            wraplength=240,
        ).pack(anchor=tk.W)

        ttk.Label(parent, text="並列数", font=("", 10, "bold")).pack(anchor=tk.W, pady=(12, 2))
        self.workers_var = tk.StringVar(value="4")
        workers_combo = ttk.Combobox(
            parent, textvariable=self.workers_var, state="readonly", width=34
        )
        workers_combo["values"] = [str(n) for n in range(1, 9)]
        workers_combo.current(3)
        workers_combo.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(
            parent,
            text="同時に処理する数。4が既定。ワーカー1つあたりモデル読み込みで"
            "メモリを約1GB使うため、メモリが少ないPCでは2以下にしてください。",
            wraplength=240,
        ).pack(anchor=tk.W)

        self.rebuild_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            parent, text="インデックスを作り直す", variable=self.rebuild_var
        ).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(
            parent,
            text="オフ（既定）は差分更新：変わっていない写真は再利用します。"
            "オンにすると全ての写真を処理し直します。",
            wraplength=240,
        ).pack(anchor=tk.W)

        # コピー保存
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=14)
        ttk.Label(parent, text="コピー保存", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(parent, text="保存先フォルダ", wraplength=240).pack(anchor=tk.W)
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X)
        self.dest_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.dest_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row2, text="選ぶ", command=self.on_choose_dest, width=6).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(parent, text="チェックしたものをコピー", command=self.on_copy_checked).pack(fill=tk.X, pady=(6, 2))
        ttk.Button(parent, text="ヒット全件コピー", command=self.on_copy_all).pack(fill=tk.X)
        ttk.Label(
            parent,
            text="元ファイルは残ります。同名は連番（_1, _2…）になります。",
            wraplength=240,
        ).pack(anchor=tk.W, pady=(6, 0))

        # ログ（exe 版はコンソールが出ない）
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=14)
        ttk.Label(parent, text="ログ", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(parent, text=f"ログの場所:\n{log_path()}", wraplength=240).pack(anchor=tk.W)
        ttk.Button(parent, text="ログを開く", command=self.on_open_log).pack(fill=tk.X, pady=(6, 0))
        ttk.Label(
            parent,
            text="エラーが出たときは、このログファイルを送ってもらうと原因が分かります。",
            wraplength=240,
        ).pack(anchor=tk.W, pady=(4, 0))

    # 中央パネル
    def _build_center(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="探したい顔（クエリ）", font=("", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)

        self.query_label = ttk.Label(top, text="クエリ画像が未選択です", relief=tk.SUNKEN, width=28)
        self.query_label.pack(side=tk.LEFT, padx=(0, 8))

        qright = ttk.Frame(top)
        qright.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(qright, text="クエリ画像を選ぶ", command=self.on_choose_query).pack(anchor=tk.W)
        ttk.Label(qright, text="クエリの顔（複数人いたら選ぶ）").pack(anchor=tk.W, pady=(8, 0))
        self.query_face_var = tk.StringVar(value="顔が未検出です")
        self.query_face_combo = ttk.Combobox(
            qright, textvariable=self.query_face_var, state="readonly", width=34
        )
        self.query_face_combo.pack(anchor=tk.W, fill=tk.X)
        self.query_face_combo.bind("<<ComboboxSelected>>", self.on_query_face_selected)

        # しきい値・表示件数・検索
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, pady=(10, 4))
        ttk.Label(ctrl, text="しきい値").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.50)
        self.threshold_scale = ttk.Scale(
            ctrl, from_=0.30, to=0.90, variable=self.threshold_var,
            orient=tk.HORIZONTAL, length=180,
            command=lambda _v: self.threshold_label_var.set(f"{self.threshold_var.get():.2f}"),
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=(4, 2))
        self.threshold_label_var = tk.StringVar(value="0.50")
        ttk.Label(ctrl, textvariable=self.threshold_label_var, width=5).pack(side=tk.LEFT)
        ttk.Label(ctrl, text="表示件数").pack(side=tk.LEFT, padx=(10, 0))
        self.topn_var = tk.IntVar(value=20)
        ttk.Spinbox(ctrl, from_=5, to=100, textvariable=self.topn_var, width=5).pack(side=tk.LEFT, padx=(2, 0))
        self.btn_search = ttk.Button(ctrl, text="検索", command=self.on_search, width=10)
        self.btn_search.pack(side=tk.LEFT, padx=(12, 0))

        self.result_count_var = tk.StringVar(value="結果: -")
        ttk.Label(parent, textvariable=self.result_count_var, font=("", 10, "bold")).pack(anchor=tk.W, pady=(6, 2))

        # 結果一覧（スクロール可）
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        self.results_frame = ttk.Frame(canvas)
        self.results_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.results_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # 右パネル
    def _build_right(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="選択中の結果", font=("", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.detail_label = ttk.Label(parent, text="結果をクリックすると拡大表示されます", relief=tk.SUNKEN)
        self.detail_label.pack(fill=tk.X)
        self.detail_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.detail_var, justify=tk.LEFT, wraplength=320).pack(
            anchor=tk.W, pady=(8, 0)
        )
        ttk.Label(
            parent,
            text="ヒット顔はオレンジ枠、他の顔はグレー枠で表示します。",
            wraplength=320,
        ).pack(anchor=tk.W, pady=(6, 0))

    # 状態表示
    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.update_idletasks()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for w in (self.btn_build, self.btn_search):
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def det_size(self) -> tuple[int, int]:
        """GUI で選ばれた検出サイズを (n, n) で返す."""
        try:
            v = int(self.det_size_var.get())
        except (ValueError, tk.TclError):
            v = 640
        if v not in (640, 1024, 1280):
            v = 640
        return (v, v)

    def workers(self) -> int:
        """GUI で選ばれた並列数（1〜8、既定 4）を返す."""
        try:
            v = int(self.workers_var.get()) if self.workers_var is not None else 4
        except (ValueError, tk.TclError):
            v = 4
        return max(1, min(8, v))

    def ensure_face_app(self, det_size: tuple[int, int] = (640, 640)):
        """InsightFace を遅延初期化する。失敗したら例外を投げる。

        検出サイズを変えたら作り直す（det_size は初期化時に決まる）。
        GPU が使えず CPU に落ちた場合はステータスバーとログに出す。
        """
        if self.face_app is None or self._face_app_det_size != det_size:
            self.set_status("InsightFace を読み込んでいます。初回はモデルをダウンロードします...")
            self.face_app = create_face_app(det_size=det_size)
            self._face_app_det_size = det_size
            # フォールバック時の注意書きだけステータスバーに出す。
            note = getattr(self.face_app, "provider_fallback_note", None)
            if note:
                self.logger.warning(note)
                self.set_status(note + "。")
        return self.face_app

    # 左：フォルダ / インデックス
    def on_choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="写真フォルダを選んでください")
        if folder:
            self.photo_folder = folder
            self.folder_var.set(folder)
            self.set_status(f"写真フォルダ: {folder}")

    def on_open_index(self) -> None:
        path = filedialog.askopenfilename(
            title="index.npz を選んでください",
            filetypes=[("インデックス", "*.npz"), ("すべて", "*.*")],
        )
        if not path:
            return
        try:
            self.index = load_index(path)
            self.index_path = path
        except Exception as e:
            messagebox.showerror("読み込み失敗", f"index.npz を読めませんでした:\n{e}")
            return
        self._update_stats()
        self.set_status(f"インデックスを開きました: {path}（{len(self.index)}件の顔）")

    def _update_stats(self) -> None:
        if self.index is None:
            self.stats_var.set("写真: -\n顔: -")
            return
        self.stats_var.set(f"写真: {self.index.num_images}枚\n顔: {self.index.num_faces}件")

    def on_build_index(self) -> None:
        if not self.photo_folder:
            messagebox.showwarning("フォルダ未選択", "まず写真フォルダを選んでください。")
            return
        if self._busy:
            return
        # 保存先は写真フォルダ直下。書けなければ exe のあるフォルダへ退避
        # （paths.choose_index_path）。カレントには依存しない
        # （exe 化すると起動方法で変わるため）。
        output_path, fallback_note = choose_index_path(self.photo_folder)
        output = str(output_path)
        if fallback_note:
            self.set_status(fallback_note)
        workers = self.workers()
        rebuild = bool(self.rebuild_var.get()) if self.rebuild_var is not None else False
        self.set_busy(True)
        self.progress_var.set("モデルを読み込んでいます...")
        self.set_status("インデックスを作成しています...")
        self.logger.info(
            "インデックス作成開始: フォルダ=%s 保存先=%s 検出サイズ=%s 並列数=%d 作り直し=%s",
            self.photo_folder, output, self.det_size(), workers, rebuild,
        )

        def work() -> None:
            try:
                # 並列時はワーカー側でモデルを作るので、メイン側は作らない。
                # 逐次時のみ先に作ってクエリで使い回す。
                app = None
                if workers <= 1:
                    app = self.ensure_face_app(self.det_size())
                    self.face_app = app

                def progress(done: int, total: int, name: str) -> None:
                    pct = int(done / max(1, total) * 100)
                    self.after(0, lambda: self._on_index_progress(pct, done, total, name))

                stats = build_index(
                    self.photo_folder, output, progress_cb=progress,
                    det_size=self.det_size(), app=app,
                    workers=workers, rebuild=rebuild,
                )
                self.after(0, lambda: self._on_index_done(output, stats))
            except Exception as e:
                # except を抜けると e が消えて lambda 内の str(e) が「None」に
                # なるため、ここで取り出してデフォルト引数で渡す。
                # 発生時点で traceback 全文もログへ書き込む。
                msg = str(e)
                tb = traceback.format_exc()
                log_traceback(self.logger, f"インデックス作成に失敗しました: {msg}")
                self.after(0, lambda msg=msg, tb=tb: self._on_index_error(msg, tb))

        threading.Thread(target=work, daemon=True).start()

    def _on_index_progress(self, pct: int, done: int, total: int, name: str) -> None:
        self.progress["value"] = pct
        self.progress_var.set(f"{done}/{total} {name}")

    def _on_index_done(self, output: str, stats: dict) -> None:
        self.set_busy(False)
        try:
            self.index = load_index(output)
            self.index_path = output
        except Exception as e:
            msg = str(e)
            tb = traceback.format_exc()
            log_traceback(self.logger, f"作成したインデックスの読み込みに失敗しました: {msg}")
            messagebox.showerror(
                "読み込み失敗",
                f"作成したインデックスを読めませんでした:\n{msg}\n\n"
                f"詳細は face-search.log を見てください。\nログの場所: {log_path()}",
            )
            return
        self._update_stats()
        self.progress["value"] = 100
        self.progress_var.set(f"完了: {stats['faces']}件の顔")
        errors = int(stats.get("errors", 0))
        timing = stats.get("timing", {})
        total_seconds = float(timing.get("total_seconds", 0.0))
        if total_seconds >= 60:
            time_text = f"{int(total_seconds // 60)}分{int(total_seconds % 60)}秒"
        else:
            time_text = f"{total_seconds:.1f}秒"
        diff_text = (
            f"再利用 {stats.get('reused_files', 0)}件 / 新規 {stats.get('new_files', 0)}件 / "
            f"変更 {stats.get('changed_files', 0)}件 / 削除 {stats.get('deleted_files', 0)}件"
        )
        self.logger.info(
            "インデックス作成完了: 画像 %s枚、顔 %s件、顔なし %s枚、読めなかった %s枚、"
            "所要時間 %s（%.2f枚/秒、最も時間のかかる段階: %s %.1f秒）、%s、保存先=%s",
            stats.get("images"), stats.get("faces"), stats.get("skipped_no_face"),
            errors, time_text, timing.get("images_per_sec", 0.0),
            timing.get("slowest_stage", "-"), timing.get("slowest_stage_seconds", 0.0),
            diff_text, output,
        )
        for d in stats.get("error_details", []):
            self.logger.info("読めなかった画像: %s: %s", d.get("file"), d.get("reason"))
        provider = stats.get("provider")
        if provider is None:
            # 処理なし（全て再利用）のときは「デバイス CPU」と出さず、
            # 未使用であることと検出されたデバイスを分けて出す。
            detected = stats.get("detected_provider", PROVIDER_CPU)
            device_text = (
                "処理なし（全て再利用）のため未使用"
                f"（検出されたデバイス: {PROVIDER_LABELS.get(detected, detected)}）"
            )
        else:
            device_text = PROVIDER_LABELS.get(provider, provider)
        message = (
            f"インデックスを作成しました。\n"
            f"画像 {stats['images']}枚、登録した顔 {stats['faces']}件\n"
            f"顔が無かった画像: {stats['skipped_no_face']}枚\n"
            f"読めなかった画像: {errors}枚\n"
            f"所要時間: {time_text}（平均 {total_seconds / max(1, int(stats.get('images', 1))):.2f}秒/枚）\n"
            f"{diff_text}\n"
            f"デバイス: {device_text}\n"
            f"保存先: {output}"
        )
        if errors > 0:
            # 読めなかったファイルの内訳をダイアログにも出す（黙って消さない）。
            lines = "\n".join(
                f"- {d['file']}: {d['reason']}" for d in stats.get("error_details", [])
            )
            message += f"\n\n読めなかった画像（最大10件）:\n{lines}"
            message += (
                "\n\n原因が分からない場合は face-search.log を見てください。"
                f"\nログの場所: {log_path()}"
            )
        self.set_status(f"インデックス作成完了: {stats['faces']}件の顔（所要時間 {time_text}）")
        messagebox.showinfo("インデックス作成完了", message)

    def _on_index_error(self, msg: str, detail: str = "") -> None:
        self.set_busy(False)
        self.progress_var.set("失敗しました")
        # traceback 全文は発生時点でログ済み。ダイアログには短い
        # メッセージとログの場所だけ出す。
        messagebox.showerror(
            "インデックス作成に失敗しました",
            f"{msg}\n\n詳細は face-search.log を見てください。\nログの場所: {log_path()}",
        )
        self.set_status("インデックス作成に失敗しました。")

    def on_open_log(self) -> None:
        """ログファイルを開く. 無ければその旨を知らせる."""
        path = log_path()
        if not path.is_file():
            messagebox.showinfo(
                "ログがありません",
                f"ログファイルはまだありません。\n場所: {path}",
            )
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # noqa: S606 - ユーザー自身のログファイルを開くだけ
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as e:
            messagebox.showerror("ログを開けませんでした", f"{path}\n{e}")

    # 中央：クエリ
    def on_choose_query(self) -> None:
        path = filedialog.askopenfilename(
            title="クエリ画像を選んでください",
            filetypes=[("画像", "*.jpg *.jpeg *.png *.bmp *.webp"), ("すべて", "*.*")],
        )
        if not path:
            return
        pil = load_pil_image(path)
        if pil is None:
            messagebox.showerror("読み込み失敗", "画像を読めませんでした。")
            return
        self.query_path = path
        self.query_pil = pil
        self.set_status("クエリ画像から顔を検出しています...")
        self.set_busy(True)

        def work() -> None:
            try:
                app = self.ensure_face_app(self.det_size())
                self.face_app = app
                # cv2.imread は Windows で日本語パスを読めないことがあるため、
                # 日本語パス対応ローダーで読む。
                img_bgr = read_image_bgr(path)
                faces = get_faces_from_image(app, img_bgr)
                embs = [face_embedding(f) for f in faces]
                self.after(0, lambda: self._on_query_detected(faces, embs))
            except Exception as e:
                # except を抜けると e が消えて lambda 内の str(e) が
                # 「None」になるため、ここで取り出す。
                msg = str(e)
                tb = traceback.format_exc()
                log_traceback(self.logger, f"クエリの読み込みに失敗しました: {msg}")
                self.after(0, lambda msg=msg, tb=tb: self._on_query_error(msg, tb))

        threading.Thread(target=work, daemon=True).start()

    def _on_query_detected(self, faces: list, embs: list[np.ndarray]) -> None:
        self.set_busy(False)
        self.query_faces = faces
        self.query_embeddings = embs
        self.selected_query_face = 0
        self.logger.info("クエリ画像から %d 人の顔を検出しました: %s", len(faces), self.query_path)
        if not faces:
            self.query_face_combo["values"] = []
            self.query_face_var.set("顔が検出されませんでした")
            self.query_label.configure(text="顔が検出されませんでした", image="")
            self.set_status("クエリ画像から顔が検出されませんでした。")
            return
        values = [
            f"{i + 1}番目の顔（検出スコア {face_det_score(f):.2f}）"
            for i, f in enumerate(faces)
        ]
        self.query_face_combo["values"] = values
        self.query_face_combo.current(0)
        self.query_face_var.set(values[0])
        self._update_query_preview()
        self.set_status(f"クエリ画像から {len(faces)}人の顔を検出しました。探す顔を選んで検索してください。")

    def _on_query_error(self, msg: str, detail: str = "") -> None:
        self.set_busy(False)
        messagebox.showerror(
            "クエリの読み込みに失敗しました",
            f"{msg}\n\n詳細は face-search.log を見てください。\nログの場所: {log_path()}",
        )
        self.set_status("クエリの読み込みに失敗しました。")

    def on_query_face_selected(self, _event=None) -> None:
        idx = self.query_face_combo.current()
        if idx >= 0:
            self.selected_query_face = idx
            self._update_query_preview()

    def _update_query_preview(self) -> None:
        if self.query_pil is None or not self.query_faces:
            return
        boxes = [face_box(f) for f in self.query_faces]
        img = draw_boxes(self.query_pil, boxes, hit_index=self.selected_query_face)
        thumb = resize_to_fit(img, 220, 220)
        self._query_thumb = ImageTk.PhotoImage(thumb)
        self.query_label.configure(image=self._query_thumb, text="")

    # 中央：検索
    def on_search(self) -> None:
        if self.index is None:
            messagebox.showwarning("インデックス未作成", "まずインデックスを作成するか開いてください。")
            return
        if not self.query_embeddings:
            messagebox.showwarning("クエリ未選択", "まずクエリ画像を選んでください。")
            return
        if self._busy:
            return
        query_emb = self.query_embeddings[self.selected_query_face]
        threshold = float(self.threshold_var.get())
        top_n = int(self.topn_var.get())
        self.set_busy(True)
        self.set_status("検索しています...")

        def work() -> None:
            try:
                results = search(query_emb, self.index, threshold=threshold, top_n=top_n)
                self.after(0, lambda: self._on_search_done(results, threshold))
            except Exception as e:
                # except を抜けると e が消えて lambda 内の str(e) が
                # 「None」になるため、ここで取り出す。
                msg = str(e)
                tb = traceback.format_exc()
                log_traceback(self.logger, f"検索に失敗しました: {msg}")
                self.after(0, lambda msg=msg, tb=tb: self._on_search_error(msg, tb))

        threading.Thread(target=work, daemon=True).start()

    def _on_search_done(self, results: list[SearchResult], threshold: float) -> None:
        self.set_busy(False)
        self.results = results
        self.checked = {r.face_index: False for r in results}
        self.check_vars = {}
        self.selected_result = -1
        self._render_results()
        self.logger.info("検索完了: %d件（しきい値 %.2f）", len(results), threshold)
        self.result_count_var.set(f"結果: {len(results)}件（しきい値 {threshold:.2f}）")
        if not results:
            self.set_status("該当する顔がありませんでした。しきい値を下げて試してください。")
            self.detail_var.set("")
            self.detail_label.configure(text="該当なし", image="")
        else:
            self.set_status(f"{len(results)}件見つかりました。クリックすると右に拡大表示されます。")
            self.on_select_result(0)

    def _on_search_error(self, msg: str, detail: str = "") -> None:
        self.set_busy(False)
        messagebox.showerror(
            "検索に失敗しました",
            f"{msg}\n\n詳細は face-search.log を見てください。\nログの場所: {log_path()}",
        )
        self.set_status("検索に失敗しました。")

    def _render_results(self) -> None:
        for child in self.results_frame.winfo_children():
            child.destroy()
        self._thumb_refs.clear()
        for pos, r in enumerate(self.results):
            card = ttk.Frame(self.results_frame, relief=tk.RIDGE, borderwidth=1)
            card.pack(fill=tk.X, padx=4, pady=4)

            var = tk.BooleanVar(value=False)
            self.check_vars[r.face_index] = var
            chk = ttk.Checkbutton(
                card, variable=var,
                command=lambda fi=r.face_index, v=var: self.checked.__setitem__(fi, bool(v.get())),
            )
            chk.pack(side=tk.LEFT, padx=(4, 0))

            pil = load_pil_image(r.image_path)
            if pil is not None:
                # 同画像の全顔枠を描く（ヒット顔だけオレンジ）
                boxes = self._boxes_for_image(r.image_path)
                hit_pos = self._hit_pos_in_image(r)
                img = draw_boxes(pil, boxes, hit_index=hit_pos)
                thumb = resize_to_fit(img, 180, 120)
                photo = ImageTk.PhotoImage(thumb)
                self._thumb_refs[pos] = photo
                lbl = ttk.Label(card, image=photo, cursor="hand2")
                lbl.pack(side=tk.LEFT, padx=4, pady=4)
                lbl.bind("<Button-1>", lambda _e, p=pos: self.on_select_result(p))
            else:
                ttk.Label(card, text="画像を読めません").pack(side=tk.LEFT, padx=4)

            info = ttk.Frame(card)
            info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ttk.Label(info, text=f"類似度 {r.similarity_percent:.1f}%", font=("", 10, "bold")).pack(anchor=tk.W)
            ttk.Label(info, text=Path(r.image_path).name, wraplength=260).pack(anchor=tk.W)
            ttk.Label(info, text=r.face_label).pack(anchor=tk.W)
            ttk.Label(info, text=f"検出スコア {r.det_score:.2f}").pack(anchor=tk.W)

    def _boxes_for_image(self, image_path: str) -> list[tuple[float, float, float, float]]:
        if self.index is None:
            return []
        out = []
        for p, b in zip(self.index.image_paths, self.index.boxes):
            if p == image_path:
                out.append(tuple(float(v) for v in b))
        return out

    def _hit_pos_in_image(self, result: SearchResult) -> int:
        return int(result.face_no_in_image)

    # 右：詳細
    def on_select_result(self, pos: int) -> None:
        if pos < 0 or pos >= len(self.results):
            return
        self.selected_result = pos
        r = self.results[pos]
        pil = load_pil_image(r.image_path)
        if pil is None:
            self.detail_var.set("画像を読めませんでした。")
            return
        boxes = self._boxes_for_image(r.image_path)
        # 表示用にリサイズし、枠も同じ比率で縮小する
        orig_w, orig_h = pil.size
        disp = resize_to_fit(pil, 320, 320)
        scale_x = disp.size[0] / orig_w
        scale_y = disp.size[1] / orig_h
        scaled = scale_boxes(boxes, scale_x, scale_y)
        img = draw_boxes(disp, scaled, hit_index=self._hit_pos_in_image(r))
        self._detail_photo = ImageTk.PhotoImage(img)
        self.detail_label.configure(image=self._detail_photo, text="")
        self.detail_var.set(
            f"類似度: {r.similarity_percent:.1f}%\n"
            f"ファイル名: {Path(r.image_path).name}\n"
            f"パス: {r.image_path}\n"
            f"写真内の位置: {r.face_label}\n"
            f"検出スコア: {r.det_score:.2f}"
        )

    # 左：コピー保存
    def on_choose_dest(self) -> None:
        folder = filedialog.askdirectory(title="保存先フォルダを選んでください")
        if folder:
            self.dest_folder = folder
            self.dest_var.set(folder)

    def _require_dest(self) -> str | None:
        dest = self.dest_var.get().strip() or self.dest_folder
        if not dest:
            messagebox.showwarning("保存先未選択", "まず保存先フォルダを選んでください。")
            return None
        return dest

    def on_copy_checked(self) -> None:
        dest = self._require_dest()
        if dest is None:
            return
        srcs = [r.image_path for r in self.results if self.checked.get(r.face_index)]
        if not srcs:
            messagebox.showinfo("コピー対象なし", "コピーする結果にチェックを入れてください。")
            return
        self._do_copy(srcs, dest)

    def on_copy_all(self) -> None:
        dest = self._require_dest()
        if dest is None:
            return
        srcs = [r.image_path for r in self.results]
        if not srcs:
            messagebox.showinfo("コピー対象なし", "検索結果がありません。")
            return
        # 同じ画像が複数ヒットしていても1回だけコピーする
        seen: set[str] = set()
        uniq: list[str] = []
        for s in srcs:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        self._do_copy(uniq, dest)

    def _do_copy(self, srcs: list[str], dest: str) -> None:
        try:
            copied = copy_files(srcs, dest)
        except Exception as e:
            messagebox.showerror("コピーに失敗しました", str(e))
            return
        if not copied:
            messagebox.showinfo("コピー完了", "コピーできたファイルがありませんでした。")
            return
        names = "\n".join(Path(t).name for _s, t in copied[:10])
        extra = f"\n...他 {len(copied) - 10}件" if len(copied) > 10 else ""
        messagebox.showinfo(
            "コピー完了",
            f"{len(copied)}件コピーしました。\n保存先: {dest}\n{names}{extra}",
        )
        self.set_status(f"{len(copied)}件コピーしました: {dest}")


def main() -> None:
    import sys as _sys

    # マルチプロセス並列化（spawn 方式・exe 化）のため最初に呼ぶ。
    # Windows の exe ではワーカー子プロセスの起動に必要。
    multiprocessing.freeze_support()
    # PyInstaller console=False の exe では stdout/stderr が None で、
    # InsightFace の初回ダウンロード時に tqdm が stderr へ書いて
    # AttributeError: 'NoneType' object has no attribute 'write' になる。
    # InsightFace・ログ設定の前に安全化する。
    ensure_std_streams()
    logger = setup_logging()
    logger.info(
        "起動: Python %s, exe=%s, ログ=%s",
        _sys.version.split()[0],
        bool(getattr(_sys, "frozen", False)),
        log_path(),
    )
    # CUDA 実行時 DLL を先にプリロードする。しないと
    # onnxruntime_providers_cuda.dll の読み込みがエラー126で失敗し、
    # CUDAExecutionProvider が一覧に出ない。CPU ビルドでは何もしない。
    _preload_cuda_dlls()
    app = FaceSearchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
