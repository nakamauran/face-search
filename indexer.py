"""写真フォルダのインデックス作成.

InsightFace buffalo_l（SCRFD検出 + ArcFace 512次元）で全顔を検出・
特徴量化し、index.npz に保存する。CLI でも使える。

    python indexer.py <写真フォルダ> [-o index.npz] [--workers 4] [--rebuild]

小さい顔・低スコアの顔は除外する（既定: 20px・0.5。集合写真の顔は
20px前後と小さいため 50px だとほぼ全て捨てられる）。初回は
~/.insightface に約281MBのモデルをダウンロードする。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from face_log import ensure_std_streams, setup_logging
from search_core import ImageReadError, normalize_embedding, read_image_bgr, save_index

logger = logging.getLogger("face_search")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
    ".gif",  # PIL 経由で読む（cv2.imdecode では読めない場合がある）
    ".heic", ".heif",  # PIL 経由で読む。pillow-heif が無いと読めない（理由付きでエラーになる）
}

# デバイス選択（CPU / CUDA / DirectML）

PROVIDER_CPU = "cpu"
PROVIDER_CUDA = "cuda"
PROVIDER_DIRECTML = "directml"
PROVIDER_CHOICES = (PROVIDER_CPU, PROVIDER_CUDA, PROVIDER_DIRECTML)

PROVIDER_LABELS = {
    PROVIDER_CPU: "CPU",
    PROVIDER_CUDA: "CUDA (NVIDIA)",
    PROVIDER_DIRECTML: "DirectML (AMD・Intel・NVIDIA共通)",
}

# onnxruntime のプロバイダ名。GPU 系は最後に CPU を足しておく
# （GPU で実行できないモデルがあっても CPU に落ちるようにする）。
_PROVIDER_NAMES = {
    PROVIDER_CPU: ["CPUExecutionProvider"],
    PROVIDER_CUDA: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    PROVIDER_DIRECTML: ["DmlExecutionProvider", "CPUExecutionProvider"],
}

# デバイス名 -> そのデバイスの onnxruntime プロバイダ名
_PROVIDER_EP = {
    PROVIDER_CPU: "CPUExecutionProvider",
    PROVIDER_CUDA: "CUDAExecutionProvider",
    PROVIDER_DIRECTML: "DmlExecutionProvider",
}


def available_providers() -> set[str]:
    """onnxruntime が使えるプロバイダ名の一覧を返す。無ければ空集合。"""
    try:
        import onnxruntime

        return set(onnxruntime.get_available_providers())
    except Exception:
        return set()


def detect_runtime_provider(avail: Optional[set[str]] = None) -> str:
    """使えるデバイスを自動検出する（CUDA > DirectML > CPU）.

    ビルド時に同梱された onnxruntime の種類で決まる。実行時の切り替えは
    ない（デバイスを変えたければ対応するパッケージでビルドし直す）。
    CUDA の実行時 DLL が同梱されていないと CUDAExecutionProvider は
    一覧に出ず、警告もなく CPU が選ばれる点に注意。
    """
    if avail is None:
        avail = available_providers()
    if _PROVIDER_EP[PROVIDER_CUDA] in avail:
        return PROVIDER_CUDA
    if _PROVIDER_EP[PROVIDER_DIRECTML] in avail:
        return PROVIDER_DIRECTML
    return PROVIDER_CPU


# CUDA の実行時 DLL のプリロード

# プリロードする DLL（(同梱先のサブフォルダ, ファイル名)）。依存する順。
# onnxruntime_providers_cuda.dll は onnxruntime_providers_shared.dll 等を
# 静的importするが、共有DLLは _internal\onnxruntime\capi\ にあり、
# Windowsローダーの標準検索パスに含まれないため、プリロードしないと
# プロバイダDLLの LoadLibrary 全体がエラー126で失敗する。
# フルパスで先に読んでおけば、後から名前だけで要求されても読み込み済み
# モジュールが返る。
_CUDA_PRELOAD_DLLS: tuple[tuple[str, str], ...] = (
    ("onnxruntime/capi", "onnxruntime_providers_shared.dll"),
    ("", "cudart64_13.dll"),
    ("", "curand64_10.dll"),  # cufft の依存。無ければスキップされる
    ("", "cublasLt64_13.dll"),
    ("", "cublas64_13.dll"),
    ("", "cufft64_12.dll"),
    # cuDNN 9 のサブDLL名。_infer/_train 付きは cuDNN 8 の名前で 9 には無い。
    # CUDA 13 ビルド（onnxruntime-gpu 1.32.0）のファイル名。
    # なお 1.32.0 のプロバイダDLLは cudart/cudnn/cufft/curand を静的
    # importしないが、セッション作成時に動的ロードされるためプリロードは
    # 引き続き有効。
    ("", "cudnn_engines_runtime_compiled64_9.dll"),
    ("", "cudnn_engines_precompiled64_9.dll"),
    ("", "cudnn_engines_tensor_ir64_9.dll"),
    ("", "cudnn_ext64_9.dll"),
    ("", "cudnn_heuristic64_9.dll"),
    ("", "cudnn_ops64_9.dll"),
    ("", "cudnn_adv64_9.dll"),
    ("", "cudnn_cnn64_9.dll"),
    ("", "cudnn_graph64_9.dll"),
    ("", "cudnn64_9.dll"),
    ("onnxruntime/capi", "onnxruntime_providers_cuda.dll"),
)

# プロセス内で1回だけプリロードするためのフラグ（ワーカーは別プロセス
# なので、それぞれで1回ずつ走る）。
_CUDA_DLLS_PRELOADED = False

# ctypes.CDLL の失敗で出る Windows エラーコードの意味。
# 126=依存DLL欠落、127=関数欠落、1114=DllMain失敗、193=形式不正。
_WINERROR_MEANINGS = {
    126: "モジュールが見つからない（依存DLL欠落）",
    127: "プロシージャが見つからない（関数欠落）",
    1114: "DllMainの失敗",
    193: "実行可能ファイルの形式が正しくない",
}


def _describe_load_error(e: OSError) -> str:
    """ctypes.CDLL の失敗を、原因の winerror 付きで表す。

    PyInstaller の pyimod03_ctypes は元の OSError を包んでしまうため、
    真の Windows エラーコードは __cause__ に隠れる。__cause__ があれば
    原因も出し、winerror が分かれば意味も付ける。
    """
    extra = []
    cause = getattr(e, "__cause__", None)
    if cause is not None:
        extra.append(f"原因: {cause}")
    winerror = getattr(e, "winerror", None)
    if cause is not None:
        winerror = getattr(cause, "winerror", winerror)
    if winerror is not None:
        meaning = _WINERROR_MEANINGS.get(winerror)
        extra.append(
            f"winerror={winerror}" + (f"（{meaning}）" if meaning else "")
        )
    if not extra:
        return str(e)
    return f"{e} ({' / '.join(extra)})"


def _preload_cuda_dlls() -> int:
    """CUDA の実行時 DLL をフルパスでプリロードする。

    凍結 exe（sys.frozen）のときのみ。無いファイルは無言スキップ、
    失敗は警告ログを出して続行する（CPU フォールバックを壊さない）。
    戻り値: プリロードできた件数。onnxruntime の import の前に呼ぶこと。
    """
    global _CUDA_DLLS_PRELOADED
    if _CUDA_DLLS_PRELOADED:
        return 0
    _CUDA_DLLS_PRELOADED = True
    if not getattr(sys, "frozen", False):
        return 0
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return 0
    base = Path(meipass)
    loaded = 0
    for sub, name in _CUDA_PRELOAD_DLLS:
        path = base / sub / name if sub else base / name
        if not path.is_file():
            continue
        try:
            ctypes.CDLL(str(path))
            loaded += 1
        except OSError as e:
            logger.warning(
                "CUDA DLL のプリロードに失敗しました: %s: %s",
                name,
                _describe_load_error(e),
            )
    if loaded:
        logger.info("CUDA DLL を %d 件プリロードしました", loaded)
    return loaded


def _make_face_app(providers: list[str], det_size: tuple[int, int], model_root: str):
    """FaceAnalysis を作って prepare まで行う（内部用）。"""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", root=model_root, providers=providers)
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def create_face_app(det_size: tuple[int, int] = (640, 640)):
    """InsightFace の FaceAnalysis（buffalo_l）を初期化して返す.

    使うデバイスは実行環境から自動検出する。GPU が使えなければ警告を
    出して CPU にフォールバックする（CPU まで失敗したときだけ
    RuntimeError を投げる）。返した app には used_provider 属性と
    provider_fallback_note 属性（フォールバック時の文言 or None）が付く。

    モデルは exe に同梱しない（buffalo_l は展開後 326MB）。初回は
    ユーザーのホームフォルダ（~/.insightface）にダウンロードされ、
    2回目以降はそこから読み込む。root を明示しているのは、exe 化した
    ときに `~` の解決に依存しないため。
    """
    # 先に CUDA の実行時 DLL をプリロードする（しないとエラー126で
    # CUDAExecutionProvider が一覧に出ない）。
    _preload_cuda_dlls()
    model_root = str(Path.home() / ".insightface")
    # 一覧と選んだデバイスの両方をログに出す。一覧が無いと「GPU が
    # 使われていない」原因がログから切り分けられない。
    avail = available_providers()
    logger.info("利用可能なプロバイダ: %s", sorted(avail))
    provider = detect_runtime_provider(avail)
    providers = list(_PROVIDER_NAMES[provider])
    fallback_note: Optional[str] = None
    try:
        app = _make_face_app(providers, det_size, model_root)
        app.used_provider = provider
        app.provider_fallback_note = None
        logger.info("推論デバイス: %s", PROVIDER_LABELS[provider])
        return app
    except Exception as e:
        # GPU でセッション作成に失敗したら CPU でもう一度試す。
        if provider != PROVIDER_CPU:
            label = PROVIDER_LABELS[provider]
            fallback_note = f"{label}が使えないためCPUで実行します"
            logger.warning(
                "%s。CPU にフォールバックします: %s: %s",
                fallback_note, type(e).__name__, e,
            )
            try:
                app = _make_face_app(
                    list(_PROVIDER_NAMES[PROVIDER_CPU]), det_size, model_root
                )
                app.used_provider = PROVIDER_CPU
                app.provider_fallback_note = fallback_note
                logger.info("推論デバイス: %s", PROVIDER_LABELS[PROVIDER_CPU])
                return app
            except Exception as e2:
                e = e2  # CPU 側の失敗として下の文言で投げ直す
        # exe 版はコンソールが出ないため、原因が分かる文面にして投げ直す。
        raise RuntimeError(
            "InsightFace モデル（buffalo_l）の読み込みに失敗しました。"
            "初回は約281MBのモデルを "
            f"{model_root} へダウンロードします。"
            "ネットワーク接続と空き容量（500MB以上）を確認してください。"
            "exe 版の場合は dist\\face-search フォルダごとコピーしたか、"
            "ウイルス対策ソフトに隔離されていないかも確認してください。"
            f"元のエラー: {type(e).__name__}: {e}"
        ) from e


# 段階別の時間計測

# 計測する段階。load は read_image_bgr、残りは app.get() の内訳。
_STAGE_KEYS = ("load", "detect", "landmark", "recognition", "genderage")
_STAGE_LABELS = {
    "load": "読み込み",
    "detect": "検出",
    "landmark": "ランドマーク",
    "recognition": "特徴量化",
    "genderage": "性別年齢推定",
}


def _new_stage_timings() -> dict[str, float]:
    return {k: 0.0 for k in _STAGE_KEYS}


def _wrap_app_for_timing(app) -> dict[str, float]:
    """app.get() の内訳（検出 / ランドマーク / 特徴量化 / 性別年齢）を測る.

    各モデルの呼び出しに計測を差し込むだけなので、処理内容・結果は
    変わらない。差し込み済みなら計測値をリセットして同じ dict を返す。
    InsightFace でない app（FakeApp 等）には差し込まず、空の dict を返す。
    """
    timings = getattr(app, "_stage_timings", None)
    if timings is not None:
        for k in timings:
            timings[k] = 0.0
        return timings
    timings = _new_stage_timings()

    det = getattr(app, "det_model", None)
    if det is not None and hasattr(det, "detect"):
        orig_detect = det.detect

        def timed_detect(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig_detect(*args, **kwargs)
            finally:
                timings["detect"] += time.perf_counter() - t0

        det.detect = timed_detect

    models = getattr(app, "models", None) or {}
    for taskname, model in models.items():
        if taskname == "detection" or not hasattr(model, "get"):
            continue
        # タスク名は landmark_3d_68 / landmark_2d_106 のように細かいため、
        # landmark 系はまとめて集計する。
        if taskname.startswith("landmark"):
            key = "landmark"
        elif taskname in timings:
            key = taskname
        else:
            continue
        orig_get = model.get

        def timed_get(*args, _orig=orig_get, _key=key, **kwargs):
            t0 = time.perf_counter()
            try:
                return _orig(*args, **kwargs)
            finally:
                timings[_key] += time.perf_counter() - t0

        model.get = timed_get

    app._stage_timings = timings
    return timings


def _record_error(error_details: list[dict], filename: str, reason: str) -> None:
    """読めなかったファイルの内訳を記録する（最大10件まで）。"""
    if len(error_details) < 10:
        error_details.append({"file": filename, "reason": reason})


def iter_image_files(folder: str | os.PathLike) -> list[Path]:
    """フォルダを再帰走査し、画像ファイルをソートして返す.

    サブフォルダもすべて対象。隠しフォルダ・隠しファイル（. で始まる
    もの）は対象外。権限が無くて入れないフォルダがあっても止まらず続ける。
    """
    root = Path(folder)
    files: list[Path] = []

    def _on_error(_err: OSError) -> None:
        # 入れないフォルダはスキップして続ける
        return None

    for dirpath, dirnames, filenames in os.walk(root, onerror=_on_error, followlinks=False):
        # 隠しフォルダは辿らない（.git や .thumbnails など）
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            p = Path(dirpath) / name
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(p)
    files.sort()
    return files


def _process_one_image(
    app, path: Path, min_face_size: int, min_det_score: float,
    timings: dict[str, float],
) -> tuple[list[tuple], Optional[dict], dict[str, float]]:
    """1枚を処理する（逐次・並列の両経路で共通）.

    戻り値: (records, error, stage)。records は
    (x1, y1, x2, y2, 検出スコア, 正規化済み特徴量) のリスト。
    error は None か {"kind": "unreadable"|"detect", "reason": 理由}。
    stage はこの1枚分の段階別時間。
    """
    t0 = time.perf_counter()
    try:
        img = read_image_bgr(path)
    except ImageReadError as e:
        stage = _new_stage_timings()
        stage["load"] = time.perf_counter() - t0
        return [], {"kind": "unreadable", "reason": str(e)}, stage
    t_load = time.perf_counter() - t0

    before = dict(timings)
    try:
        faces = list(app.get(img))
    except Exception as e:
        stage = _new_stage_timings()
        stage["load"] = t_load
        for k in _STAGE_KEYS[1:]:
            stage[k] = max(0.0, timings.get(k, 0.0) - before.get(k, 0.0))
        return [], {
            "kind": "detect",
            "reason": f"顔検出中にエラー（{type(e).__name__}: {e}）",
        }, stage
    after = dict(timings)
    stage = {"load": t_load}
    for k in _STAGE_KEYS[1:]:
        stage[k] = max(0.0, after.get(k, 0.0) - before.get(k, 0.0))

    # 小さい顔・スコアが低い顔を除外する
    kept = []
    for face in faces:
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        w, h = x2 - x1, y2 - y1
        det_score = float(getattr(face, "det_score", 0.0))
        if w < min_face_size or h < min_face_size:
            continue
        if det_score < min_det_score:
            continue
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = face.embedding
        kept.append((x1, y1, x2, y2, det_score, normalize_embedding(emb)))
    return kept, None, stage


# 並列インデックス作成（マルチプロセス）

# ワーカーごとの状態（ワーカーが自分の FaceAnalysis を1回だけ初期化する）。
_WORKER_STATE: dict = {}


def _index_worker_init(
    det_size: tuple[int, int],
    min_face_size: int,
    min_det_score: float,
) -> None:
    """ワーカーの初期化（spawn でも動くようモジュールレベル関数）.

    スレッド数は1に絞る（複数ワーカーがそれぞれ複数スレッドを使うと
    CPU の取り合いになる）。OMP_NUM_THREADS は onnxruntime の
    セッション作成前に読まれるので、create_face_app() の前に設定する。
    spawn ワーカーはロガーのハンドラを持たないため setup_logging() が
    要る（無いとワーカー側のログが face-search.log に届かない）。
    モデル未DLの状態でワーカーが初めて create_face_app() を呼ぶと
    tqdm が sys.stderr（None）へ書き、'NoneType' write で落ちるため
    ensure_std_streams() で先に逃がす。
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    ensure_std_streams()
    setup_logging()
    app = create_face_app(det_size=det_size)
    timings = _wrap_app_for_timing(app)
    _WORKER_STATE.update(
        {
            "app": app,
            "timings": timings,
            "min_face_size": min_face_size,
            "min_det_score": min_det_score,
            "provider": getattr(app, "used_provider", PROVIDER_CPU),
        }
    )


def _index_worker_process(path_str: str) -> dict:
    """ワーカーで1枚を処理し、結果を返す（picklable な dict）.

    records の特徴量は np.ndarray のまま返す（pickle 可）。
    """
    st = _WORKER_STATE
    records, error, stage = _process_one_image(
        st["app"], Path(path_str),
        st["min_face_size"], st["min_det_score"], st["timings"],
    )
    return {
        "path": path_str,
        "records": records,
        "error": error,
        "stage": stage,
        "provider": st.get("provider", PROVIDER_CPU),
    }


# 差分更新（index.meta.json）

def _meta_path(output: str | os.PathLike) -> Path:
    """index.npz と同じフォルダの index.meta.json のパスを返す."""
    return Path(output).with_suffix(".meta.json")


def _load_previous_index(
    output: str | os.PathLike, params: dict
) -> tuple[dict[str, dict], dict[str, dict]] | tuple[None, None]:
    """既存の index.npz + index.meta.json を読み、再利用できる形で返す.

    戻り値: (path -> その画像の顔行, path -> {"size":, "mtime_ns":})。
    使えない場合（ファイル無し・破損・作成時のパラメータ不一致）は
    (None, None)（呼び出し側は全作り直しにする）。
    """
    from search_core import load_index

    meta_file = _meta_path(output)
    if not Path(output).is_file() or not meta_file.is_file():
        return None, None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("version") != 1:
            return None, None
        old_params = meta.get("params", {})
        for key in ("min_face_size", "min_det_score", "det_size"):
            if old_params.get(key) != params.get(key):
                logger.info(
                    "差分更新をやめ、全作り直しにします（%s が %s から %s に変わりました）。",
                    key, old_params.get(key), params.get(key),
                )
                return None, None
        file_meta = meta.get("files", {})
        if not isinstance(file_meta, dict):
            return None, None
        index = load_index(output)
    except Exception as e:
        logger.info("既存のインデックスを読めなかったため全作り直しにします: %s", e)
        return None, None

    rows: dict[str, dict] = {}
    for i, p in enumerate(index.image_paths):
        entry = rows.setdefault(p, {
            "boxes": [], "embeddings": [], "scores": [],
            "face_no": [], "faces_per": [],
        })
        entry["boxes"].append([float(v) for v in index.boxes[i]])
        entry["embeddings"].append(np.asarray(index.embeddings[i], dtype=np.float32))
        entry["scores"].append(float(index.scores[i]))
        entry["face_no"].append(int(index.face_no_in_image[i]))
        entry["faces_per"].append(int(index.faces_per_image[i]))
    return rows, file_meta


def _save_meta(output: str | os.PathLike, params: dict, file_stats: dict[str, dict]) -> None:
    """index.meta.json を保存する（失敗してもインデックス自体は残す）。"""
    meta = {"version": 1, "params": params, "files": file_stats}
    try:
        _meta_path(output).write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except OSError as e:
        logger.warning("index.meta.json を保存できませんでした: %s", e)


def _log_progress_summary(
    done: int, total: int, faces: int, elapsed: float, stage_totals: dict[str, float]
) -> None:
    """100枚ごとの中間集計をログに出す（1枚ごとの行は出さない）。"""
    stage_text = " / ".join(
        f"{_STAGE_LABELS[k]} {stage_totals.get(k, 0.0):.1f}秒" for k in _STAGE_KEYS
    )
    logger.info(
        "進捗: %d/%d枚（%.0f%%）、%.1f秒経過、%.2f枚/秒、%.1f顔/秒、内訳 %s",
        done, total, done / max(1, total) * 100, elapsed,
        done / max(elapsed, 0.001), faces / max(elapsed, 0.001), stage_text,
    )


def build_index(
    folder: str | os.PathLike,
    output: str | os.PathLike,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    min_face_size: int = 20,
    min_det_score: float = 0.5,
    det_size: tuple[int, int] = (640, 640),
    app=None,
    workers: int = 1,
    rebuild: bool = False,
) -> dict:
    """インデックスを作成して保存し、統計を返す.

    progress_cb(done, total, current_filename) は進捗表示用。
    app を渡せば再利用する（渡さなければここで初期化する）。
    workers は並列数（1〜8、既定 1。GUI の既定は 4）。1 なら逐次、
    2 以上ならマルチプロセスで、プール作成に失敗したら逐次に
    フォールバックする。rebuild が False（既定）なら差分更新する
    （パス＋サイズ＋mtime_ns が一致するファイルは再利用し、
    パラメータが変わっていたら全作り直し）。True なら全処理し直す。
    戻り値は統計の dict（images / faces / errors / reused_files /
    new_files / changed_files / deleted_files / workers / provider /
    detected_provider / timing）。読めなかったファイルの内訳は
    error_details に残す（最大10件）。
    """
    t_start = time.perf_counter()
    workers = max(1, min(8, int(workers) if workers else 1))
    params = {
        "min_face_size": min_face_size,
        "min_det_score": min_det_score,
        "det_size": [det_size[0], det_size[1]],
    }

    files = iter_image_files(folder)
    total = len(files)

    # メイン側でも使うデバイスを検出してログに出す。並列時はワーカーが
    # 自分のモデルを初期化するためメイン側では create_face_app が呼ばれず、
    # 全て再利用の場合にプロバイダ検出自体が一度も走らないため。
    _preload_cuda_dlls()
    avail = available_providers()
    detected_provider = detect_runtime_provider(avail)
    logger.info("利用可能なプロバイダ: %s", sorted(avail))
    logger.info("検出されたデバイス: %s", PROVIDER_LABELS[detected_provider])

    # ファイルごとのサイズ・更新時刻（差分判定用）
    file_stats: dict[str, dict] = {}
    for p in files:
        try:
            st = p.stat()
            file_stats[str(p)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError:
            file_stats[str(p)] = {"size": -1, "mtime_ns": -1}

    # 差分更新：既存インデックスが使えれば、変わっていないファイルは再利用する
    prev_rows: dict[str, dict] = {}
    prev_meta: dict[str, dict] = {}
    reused_keys: set[str] = set()
    reused_files = new_files = changed_files = deleted_files = 0
    to_process: list[Path] = []
    if not rebuild:
        loaded_rows, loaded_meta = _load_previous_index(output, params)
        if loaded_rows is not None:
            prev_rows, prev_meta = loaded_rows, loaded_meta
            for p in files:
                key = str(p)
                old = prev_meta.get(key)
                if old is not None and old.get("size") == file_stats[key]["size"] \
                        and old.get("mtime_ns") == file_stats[key]["mtime_ns"]:
                    # 前回も顔が無かった画像は再利用扱い（処理し直さない）。
                    reused_files += 1
                    reused_keys.add(key)
                else:
                    to_process.append(p)
                    if old is not None:
                        changed_files += 1
                    else:
                        new_files += 1
            deleted_files = len(set(prev_meta) - set(file_stats))
            if deleted_files:
                logger.info("差分更新: 削除されたファイル %d件をインデックスから除きます。", deleted_files)
        else:
            to_process = list(files)
            new_files = len(files)
    else:
        to_process = list(files)
        new_files = len(files)

    image_paths: list[str] = []
    boxes: list[list[float]] = []
    embeddings: list[np.ndarray] = []
    scores: list[float] = []
    face_no: list[int] = []
    faces_per: list[int] = []

    skipped_no_face = 0
    errors = 0
    unreadable = 0
    detect_errors = 0
    error_details: list[dict] = []

    stage_totals = _new_stage_timings()
    # 処理対象が無ければ None のまま（「デバイス CPU」と誤表示しないため）。
    # 処理が走れば逐次・並列のいずれかが実際に使ったデバイスで上書きする。
    used_provider: Optional[str] = None

    def _add_records(path_str: str, records: list[tuple]) -> None:
        """1枚分の顔を登録用リストへ追加する（逐次・並列で共通）。"""
        for j, (x1, y1, x2, y2, det_score, emb) in enumerate(records):
            image_paths.append(path_str)
            boxes.append([x1, y1, x2, y2])
            embeddings.append(np.asarray(emb, dtype=np.float32))
            scores.append(det_score)
            face_no.append(j)
            faces_per.append(len(records))

    def _add_reused(path_str: str) -> None:
        """差分更新で再利用する顔行を登録用リストへ追加する。"""
        entry = prev_rows[path_str]
        for j in range(len(entry["boxes"])):
            image_paths.append(path_str)
            boxes.append(entry["boxes"][j])
            embeddings.append(np.asarray(entry["embeddings"][j], dtype=np.float32))
            scores.append(entry["scores"][j])
            face_no.append(entry["face_no"][j])
            faces_per.append(entry["faces_per"][j])

    def _handle_error(path: Path, error: dict) -> None:
        nonlocal errors, unreadable, detect_errors
        errors += 1
        if error["kind"] == "unreadable":
            unreadable += 1
        else:
            detect_errors += 1
        _record_error(error_details, path.name, error["reason"])

    def _accumulate_stage(stage: dict[str, float]) -> None:
        for k in _STAGE_KEYS:
            stage_totals[k] += stage.get(k, 0.0)

    # 再利用分を除いた処理対象だけを処理し、結果を path -> 結果 で持つ。
    # 最後にファイル順に組み立てる。
    processed: dict[str, tuple[list[tuple], Optional[dict]]] = {}
    done = reused_files  # 再利用分は最初から「済み」として数える

    if to_process:
        if workers > 1:
            try:
                _process_parallel(
                    to_process, det_size, min_face_size, min_det_score,
                    workers, progress_cb, total, done,
                    processed, stage_totals, _handle_error, _accumulate_stage,
                )
                used_provider = _process_parallel.last_provider
            except Exception as e:
                logger.warning(
                    "並列処理に失敗したため逐次処理に切り替えます: %s: %s",
                    type(e).__name__, e,
                )
                # 並列で集めた途中結果は捨て、全て逐次で処理し直す
                # （二重登録・二重計数を防ぐため、カウンタも戻す）。
                processed.clear()
                for k in _STAGE_KEYS:
                    stage_totals[k] = 0.0
                errors = 0
                unreadable = 0
                detect_errors = 0
                error_details.clear()
                used_provider = _process_sequential(
                    to_process, app, det_size, min_face_size, min_det_score,
                    progress_cb, total, reused_files,
                    processed, stage_totals, _handle_error, _accumulate_stage,
                )
        else:
            used_provider = _process_sequential(
                to_process, app, det_size, min_face_size, min_det_score,
                progress_cb, total, reused_files,
                processed, stage_totals, _handle_error, _accumulate_stage,
            )

    # ファイル順に組み立てる（再利用分と処理分を混ぜる）
    for p in files:
        key = str(p)
        if key in reused_keys:
            if key in prev_rows:
                _add_reused(key)
            else:
                skipped_no_face += 1
            continue
        records, error = processed.get(key, ([], None))
        if error is not None:
            continue
        if not records:
            skipped_no_face += 1
            continue
        _add_records(key, records)

    if embeddings:
        emb_arr = np.vstack(embeddings).astype(np.float32)
        box_arr = np.array(boxes, dtype=np.float32)
    else:
        emb_arr = np.empty((0, 512), dtype=np.float32)
        box_arr = np.empty((0, 4), dtype=np.float32)

    save_index(
        output,
        image_paths=image_paths,
        boxes=box_arr,
        embeddings=emb_arr,
        scores=np.array(scores, dtype=np.float32),
        face_no_in_image=np.array(face_no, dtype=np.int64),
        faces_per_image=np.array(faces_per, dtype=np.int64),
    )
    _save_meta(output, params, file_stats)

    total_seconds = time.perf_counter() - t_start
    slowest_key = max(_STAGE_KEYS, key=lambda k: stage_totals.get(k, 0.0))
    timing = {
        "total_seconds": total_seconds,
        "load_seconds": stage_totals["load"],
        "detect_seconds": stage_totals["detect"],
        "landmark_seconds": stage_totals["landmark"],
        "recognition_seconds": stage_totals["recognition"],
        "genderage_seconds": stage_totals["genderage"],
        "images_per_sec": total / max(total_seconds, 0.001),
        "faces_per_sec": len(image_paths) / max(total_seconds, 0.001),
        "slowest_stage": _STAGE_LABELS[slowest_key],
        "slowest_stage_seconds": stage_totals[slowest_key],
    }
    stage_text = " / ".join(
        f"{_STAGE_LABELS[k]} {stage_totals[k]:.1f}秒" for k in _STAGE_KEYS
    )
    if used_provider is None:
        # 処理対象が無く、推論が1枚も走っていない。「デバイス CPU」と出すと
        # GPU が使われていない誤解になるため、未使用であることと検出された
        # デバイスを分けて出す。
        device_text = (
            "処理なし（全て再利用）のため未使用"
            f"（検出されたデバイス: {PROVIDER_LABELS[detected_provider]}）"
        )
    else:
        device_text = PROVIDER_LABELS.get(used_provider, used_provider)
    logger.info(
        "インデックス作成の集計: 総時間 %.1f秒、画像 %d枚（%.2f枚/秒）、顔 %d件（%.1f顔/秒）、"
        "最も時間のかかる段階: %s（%.1f秒）、内訳 %s、再利用 %d件 / 新規 %d件 / 変更 %d件 / 削除 %d件、"
        "並列数 %d、デバイス %s",
        total_seconds, total, timing["images_per_sec"], len(image_paths),
        timing["faces_per_sec"], timing["slowest_stage"],
        timing["slowest_stage_seconds"], stage_text,
        reused_files, new_files, changed_files, deleted_files,
        workers if to_process else 1, device_text,
    )
    return {
        "images": total,
        "faces": len(image_paths),
        "skipped_no_face": skipped_no_face,
        "errors": errors,
        "unreadable": unreadable,
        "detect_errors": detect_errors,
        "error_files": [d["file"] for d in error_details],
        "error_details": error_details,
        "reused_files": reused_files,
        "new_files": new_files,
        "changed_files": changed_files,
        "deleted_files": deleted_files,
        "workers": workers if to_process else 1,
        "provider": used_provider,
        "detected_provider": detected_provider,
        "timing": timing,
    }


def _process_sequential(
    to_process: list[Path],
    app,
    det_size: tuple[int, int],
    min_face_size: int,
    min_det_score: float,
    progress_cb,
    total: int,
    base_done: int,
    processed: dict,
    stage_totals: dict[str, float],
    handle_error,
    accumulate_stage,
) -> str:
    """逐次処理（ワーカー数 1）。app が無ければここで初期化する。

    戻り値: 実際に使ったデバイス名。
    """
    t_start = time.perf_counter()
    if app is None:
        app = create_face_app(det_size=det_size)
    used_provider = getattr(app, "used_provider", PROVIDER_CPU)
    timings = _wrap_app_for_timing(app)
    done = base_done
    for path in to_process:
        records, error, stage = _process_one_image(
            app, path, min_face_size, min_det_score, timings
        )
        accumulate_stage(stage)
        if error is not None:
            handle_error(path, error)
        processed[str(path)] = (records, error)
        done += 1
        if progress_cb is not None:
            progress_cb(done, total, path.name)
        if done % 100 == 0:
            _log_progress_summary(
                done, total, sum(len(v[0]) for v in processed.values()),
                time.perf_counter() - t_start, stage_totals,
            )
    return used_provider


def _process_parallel(
    to_process: list[Path],
    det_size: tuple[int, int],
    min_face_size: int,
    min_det_score: float,
    workers: int,
    progress_cb,
    total: int,
    base_done: int,
    processed: dict,
    stage_totals: dict[str, float],
    handle_error,
    accumulate_stage,
) -> None:
    """マルチプロセス並列処理。結果は processed に path -> (records, error) で入る。

    imap_unordered で進捗を返す。ワーカーは spawn 方式で立ち上げる
    （Windows の既定・exe 化と同じ挙動にするため）。実際に使った
    デバイス名は _process_parallel.last_provider に残す。
    """
    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    last_provider = PROVIDER_CPU
    done = base_done
    faces_done = 0
    with ctx.Pool(
        processes=workers,
        initializer=_index_worker_init,
        initargs=(det_size, min_face_size, min_det_score),
    ) as pool:
        for result in pool.imap_unordered(
            _index_worker_process, [str(p) for p in to_process], chunksize=1
        ):
            path = Path(result["path"])
            records = result["records"]
            error = result["error"]
            accumulate_stage(result["stage"])
            if error is not None:
                handle_error(path, error)
            processed[result["path"]] = (records, error)
            if error is None:
                faces_done += len(records)
            last_provider = result.get("provider", PROVIDER_CPU)
            done += 1
            if progress_cb is not None:
                progress_cb(done, total, path.name)
            if done % 100 == 0:
                _log_progress_summary(
                    done, total, faces_done,
                    time.perf_counter() - t_start, stage_totals,
                )
    _process_parallel.last_provider = last_provider


_process_parallel.last_provider = PROVIDER_CPU


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="顔写真フォルダのインデックス作成 (InsightFace buffalo_l)")
    parser.add_argument("folder", help="写真フォルダ（再帰走査）")
    parser.add_argument("-o", "--output", default=None,
                        help="出力する index.npz（既定: 写真フォルダ直下の index.npz。"
                        "省略するとカレントディレクトリではなく写真フォルダに作る。"
                        "exe 化するとカレントは起動方法で変わるため）")
    parser.add_argument("--min-size", type=int, default=20, help="除外する顔の最小サイズ px（既定: 20。集合写真では顔が小さいため 50 では拾えない）")
    parser.add_argument("--min-score", type=float, default=0.5, help="除外する検出スコアの下限（既定: 0.5）")
    parser.add_argument(
        "--det-size", type=int, default=640, choices=[640, 1024, 1280],
        help="検出時の入力サイズ（既定: 640。1280 にすると小さい顔も拾えるが遅くなる）",
    )
    parser.add_argument(
        "--workers", type=int, default=4, choices=range(1, 9),
        help="並列数（既定: 4。1 なら逐次処理。ワーカー1つあたりモデル読み込みで"
        "メモリを使うため、メモリが少ない PC では 2 以下にすること）",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="既存のインデックスを使わず全ファイルを処理し直す（既定は差分更新）",
    )
    args = parser.parse_args(argv)

    if not Path(args.folder).is_dir():
        print(f"フォルダが見つかりません: {args.folder}", file=sys.stderr)
        return 1

    # -o 省略時は写真フォルダ直下の index.npz にする（カレント依存を避ける）。
    output = args.output or str(Path(args.folder) / "index.npz")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    print("InsightFace (buffalo_l) を読み込んでいます。初回はモデルのダウンロードが入ります...")
    try:
        stats = build_index(
            args.folder,
            output,
            progress_cb=lambda d, t, name: print(f"[{d}/{t}] {name}"),
            min_face_size=args.min_size,
            min_det_score=args.min_score,
            det_size=(args.det_size, args.det_size),
            workers=args.workers,
            rebuild=args.rebuild,
        )
    except RuntimeError as e:
        print(f"失敗しました: {e}", file=sys.stderr)
        return 2
    timing = stats.get("timing", {})
    print(f"完了: 画像 {stats['images']}枚、登録した顔 {stats['faces']}件")
    print(f"  顔が無かった画像: {stats['skipped_no_face']}枚、読めなかった画像: {stats['errors']}枚")
    print(
        f"  差分更新: 再利用 {stats.get('reused_files', 0)}件、新規 {stats.get('new_files', 0)}件、"
        f"変更 {stats.get('changed_files', 0)}件、削除 {stats.get('deleted_files', 0)}件"
    )
    print(
        f"  所要時間: {timing.get('total_seconds', 0):.1f}秒"
        f"（{timing.get('images_per_sec', 0):.2f}枚/秒、"
        f"最も時間のかかる段階: {timing.get('slowest_stage', '-')}"
        f" {timing.get('slowest_stage_seconds', 0):.1f}秒）"
    )
    if stats.get("error_details"):
        print("  読めなかった画像の内訳（最大10件）:")
        for d in stats["error_details"]:
            print(f"    - {d['file']}: {d['reason']}")
    print(f"  保存先: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
