"""顔類似検索のコア（GUI非依存）.

index.npz の読み込み、コサイン類似度での検索、コピー保存の
同名連番解決。InsightFace / OpenCV の細部はここに閉じ込め、
重い依存は関数の引数で受ける。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


# インデックス

@dataclass
class FaceIndex:
    """index.npz を読み込んだ結果を保持する."""

    image_paths: list[str]          # 顔ごとの元画像パス（顔数 N と同じ長さ）
    boxes: np.ndarray               # (N, 4) [x1, y1, x2, y2] float32
    embeddings: np.ndarray          # (N, 512) L2正規化済み float32
    scores: np.ndarray              # (N,) 検出スコア float32
    face_no_in_image: np.ndarray    # (N,) その写真の中で何番目の顔か（0始まり）
    faces_per_image: np.ndarray     # (N,) その写真に写っている顔の総数

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def num_images(self) -> int:
        return len(set(self.image_paths))

    @property
    def num_faces(self) -> int:
        return len(self.image_paths)


def load_index(path: str | os.PathLike) -> FaceIndex:
    """index.npz を読み込んで FaceIndex を返す. ファイルが無ければ FileNotFoundError."""
    data = np.load(path, allow_pickle=False)
    return FaceIndex(
        image_paths=[str(p) for p in data["image_paths"]],
        boxes=np.asarray(data["boxes"], dtype=np.float32),
        embeddings=np.asarray(data["embeddings"], dtype=np.float32),
        scores=np.asarray(data["scores"], dtype=np.float32),
        face_no_in_image=np.asarray(data["face_no_in_image"], dtype=np.int64),
        faces_per_image=np.asarray(data["faces_per_image"], dtype=np.int64),
    )


def save_index(
    path: str | os.PathLike,
    image_paths: Sequence[str],
    boxes: np.ndarray,
    embeddings: np.ndarray,
    scores: np.ndarray,
    face_no_in_image: np.ndarray,
    faces_per_image: np.ndarray,
) -> None:
    """FaceIndex と同じ項目で index.npz を保存する（indexer.py から使う）."""
    # 保存先フォルダが無ければ作る（退避先など親が想定外の場所のときのため）。
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    # dtype=object にすると load_index の allow_pickle=False で読めなくなる
    # ため、画像パスは文字列配列で保存する。
    np.savez_compressed(
        path,
        image_paths=np.array(list(image_paths), dtype=str),
        boxes=np.asarray(boxes, dtype=np.float32),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        face_no_in_image=np.asarray(face_no_in_image, dtype=np.int64),
        faces_per_image=np.asarray(faces_per_image, dtype=np.int64),
    )


# 画像読み込み（日本語パス対応）

class ImageReadError(Exception):
    """画像ファイルが読めなかったときの例外（理由付き）."""


def read_image_bgr(path: str | os.PathLike) -> np.ndarray:
    """画像ファイルを BGR の numpy 配列で読む（日本語パス対応）.

    cv2.imread は Windows で非 ASCII パスを読めないことがあるため、
    バイト列→cv2.imdecode を先に試し、ダメなら PIL（EXIF 補正あり）で
    読む。どちらも失敗したら理由付きの ImageReadError（None は返さない）。
    """
    p = Path(path)
    if not p.is_file():
        raise ImageReadError(f"ファイルが見つかりません: {path}")
    try:
        data = p.read_bytes()
    except OSError as e:
        raise ImageReadError(f"ファイルを開けませんでした（{e}）: {path}") from e
    if not data:
        raise ImageReadError(f"ファイルが空です: {path}")

    cv2_error = ""
    try:
        import cv2

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        cv2_error = "cv2.imdecode が None を返しました（対応外の形式か、破損している可能性）"
    except ImportError as e:
        cv2_error = f"OpenCV が使えません（{e}）"
    except Exception as e:
        cv2_error = f"OpenCV での読み込み中にエラー（{type(e).__name__}: {e}）"

    # PIL フォールバック。チャンネル反転で BGR 化する
    # （cv2 を使わないので、OpenCV が壊れていても読める）。
    try:
        from PIL import Image, ImageOps

        with Image.open(p) as pil:
            pil = ImageOps.exif_transpose(pil)
            rgb = pil.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
        if arr.size == 0:
            raise ImageReadError(f"画像データが空です: {path}")
        return np.ascontiguousarray(arr[:, :, ::-1])
    except ImageReadError:
        raise
    except ImportError as e:
        raise ImageReadError(
            f"画像を読めませんでした: {path}。OpenCV: {cv2_error}。"
            f"Pillow が使えません（{e}）。"
        ) from e
    except Exception as e:
        hint = ""
        if p.suffix.lower() in {".heic", ".heif"}:
            hint = (
                "HEIC/HEIF を読むには pillow-heif が必要です"
                "（pip install pillow-heif）。JPEG/PNG に変換するのも手です。"
            )
        raise ImageReadError(
            f"画像を読めませんでした: {path}。OpenCV: {cv2_error}。"
            f"PIL でも読めませんでした（{type(e).__name__}: {e}）。{hint}"
            "ファイルが破損しているか、対応していない形式の可能性があります。"
        ) from e


# 類似度

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """特徴量を L2 正規化する. ゼロベクトルはそのまま返す."""
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    if norm == 0.0:
        return emb
    return (emb / norm).astype(np.float32)


def cosine_similarities(
    query_embedding: np.ndarray, embeddings: np.ndarray
) -> np.ndarray:
    """クエリと全インデックス顔のコサイン類似度を返す。

    戻り値: (N,) float32、-1.0〜1.0。空なら空配列。
    未正規化の場合はここで正規化する。
    """
    if len(embeddings) == 0:
        return np.empty((0,), dtype=np.float32)
    q = normalize_embedding(query_embedding)
    embs = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embs = embs / norms
    return (embs @ q).astype(np.float32)


@dataclass
class SearchResult:
    face_index: int        # インデックス内での顔番号
    image_path: str        # 元画像パス
    box: tuple[float, float, float, float]
    score: float           # 類似度（コサイン類似度、-1〜1）
    det_score: float       # 検出スコア
    face_no_in_image: int  # 0始まり
    faces_per_image: int

    @property
    def similarity_percent(self) -> float:
        """類似度を 0〜100% に換算する."""
        return float(self.score) * 100.0

    @property
    def face_label(self) -> str:
        """表示用: 「5人中 3番目」."""
        return f"{self.faces_per_image}人中 {self.face_no_in_image + 1}番目"


def search(
    query_embedding: np.ndarray,
    index: FaceIndex,
    threshold: float = 0.5,
    top_n: int = 20,
) -> list[SearchResult]:
    """類似度がしきい値以上の顔を、類似度順（降順）で上位N件返す.

    ArcFace では同一人物で 0.5〜0.7 程度、他人で 0.3 前後になることが
    多い。既定 0.5 は調整の出発点。
    """
    if len(index) == 0:
        return []
    sims = cosine_similarities(query_embedding, index.embeddings)
    order = np.argsort(-sims)
    results: list[SearchResult] = []
    for idx in order:
        if sims[idx] < threshold:
            break  # 降順なので、ここから先はすべてしきい値未満
        i = int(idx)
        x1, y1, x2, y2 = (float(v) for v in index.boxes[i])
        results.append(
            SearchResult(
                face_index=i,
                image_path=index.image_paths[i],
                box=(x1, y1, x2, y2),
                score=float(sims[idx]),
                det_score=float(index.scores[i]),
                face_no_in_image=int(index.face_no_in_image[i]),
                faces_per_image=int(index.faces_per_image[i]),
            )
        )
        if len(results) >= max(1, int(top_n)):
            break
    return results


# クエリ顔の特徴量化（InsightFace は外から渡す）

def get_faces_from_image(app, image_bgr: np.ndarray) -> list:
    """InsightFace FaceAnalysis で画像から全顔を検出して返す.

    app は準備済みの FaceAnalysis。検出なしなら空リスト。
    """
    return list(app.get(image_bgr))


def face_embedding(face) -> np.ndarray:
    """InsightFace の Face オブジェクトから L2 正規化済み特徴量を返す."""
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = face.embedding
    return normalize_embedding(np.asarray(emb, dtype=np.float32))


def face_box(face) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in face.bbox)
    return (x1, y1, x2, y2)


def face_det_score(face) -> float:
    return float(getattr(face, "det_score", 0.0))


# コピー保存（同名は連番で上書きしない）

def resolve_copy_path(dest_dir: str | os.PathLike, filename: str) -> Path:
    """コピー先のパスを決める。同名があれば _1, _2, ... を拡張子の前に付ける."""
    dest = Path(dest_dir)
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = dest / filename
    n = 1
    while candidate.exists():
        candidate = dest / f"{stem}_{n}{suffix}"
        n += 1
    return candidate


def copy_files(
    src_paths: Iterable[str | os.PathLike],
    dest_dir: str | os.PathLike,
    on_copied: Optional[Callable[[str, str], None]] = None,
) -> list[tuple[str, str]]:
    """元ファイルを残したまま（shutil.copy2）保存先へコピーする.

    戻り値: [(元パス, コピー先パス), ...]。保存先フォルダが無ければ作る。
    同名は連番化する。
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, str]] = []
    for src in src_paths:
        src_path = Path(src)
        if not src_path.is_file():
            continue
        target = resolve_copy_path(dest, src_path.name)
        shutil.copy2(src_path, target)
        copied.append((str(src_path), str(target)))
        if on_copied is not None:
            on_copied(str(src_path), str(target))
    return copied
