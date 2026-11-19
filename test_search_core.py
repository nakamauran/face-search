"""search_core の単体テスト.

`pytest test_search_core.py` か `python test_search_core.py` で実行できる。
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from search_core import (  # noqa: E402
    FaceIndex,
    copy_files,
    cosine_similarities,
    normalize_embedding,
    resolve_copy_path,
    search,
)


def _make_index() -> FaceIndex:
    # 3人の顔: A(1,0,0...)、Aに近い顔、Aと直交する顔
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    a_close = normalize_embedding(np.array([0.9, 0.1, 0.0], dtype=np.float32))
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    embs = np.vstack([a, a_close, b]).astype(np.float32)
    return FaceIndex(
        image_paths=["a.jpg", "a2.jpg", "b.jpg"],
        boxes=np.array([[0, 0, 10, 10]] * 3, dtype=np.float32),
        embeddings=embs,
        scores=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        face_no_in_image=np.array([0, 0, 0]),
        faces_per_image=np.array([1, 1, 1]),
    )


def test_cosine_same_is_1():
    idx = _make_index()
    sims = cosine_similarities(idx.embeddings[0], idx.embeddings)
    assert abs(float(sims[0]) - 1.0) < 1e-5, sims
    assert float(sims[2]) < 0.01, sims  # 直交する顔はほぼ 0


def test_search_threshold_and_order():
    idx = _make_index()
    # しきい値 0.5: A本人と Aに近い顔だけ（直交する B は除外）
    results = search(idx.embeddings[0], idx, threshold=0.5, top_n=10)
    assert [r.image_path for r in results] == ["a.jpg", "a2.jpg"], results
    assert results[0].score >= results[1].score
    # しきい値 0.999: A本人だけ（Aに近い顔の類似度は約0.994なので除外される）
    results = search(idx.embeddings[0], idx, threshold=0.999, top_n=10)
    assert [r.image_path for r in results] == ["a.jpg"]
    results = search(idx.embeddings[0], idx, threshold=0.0, top_n=1)
    assert len(results) == 1 and results[0].image_path == "a.jpg"
    empty = FaceIndex([], np.empty((0, 4), np.float32), np.empty((0, 3), np.float32),
                      np.empty((0,), np.float32), np.empty((0,), np.int64), np.empty((0,), np.int64))
    assert search(idx.embeddings[0], empty) == []


def test_resolve_copy_path_no_conflict():
    with tempfile.TemporaryDirectory() as d:
        assert resolve_copy_path(d, "photo.jpg") == Path(d) / "photo.jpg"


def test_resolve_copy_path_numbering():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "photo.jpg").write_text("x")
        assert resolve_copy_path(d, "photo.jpg") == Path(d) / "photo_1.jpg"
        Path(d, "photo_1.jpg").write_text("x")
        assert resolve_copy_path(d, "photo.jpg") == Path(d) / "photo_2.jpg"
        # 拡張子なしも連番になる
        Path(d, "note").write_text("x")
        assert resolve_copy_path(d, "note") == Path(d) / "note_1"


def test_copy_files_keeps_original_and_renumbers():
    with tempfile.TemporaryDirectory() as src_d, tempfile.TemporaryDirectory() as dst_d:
        src = Path(src_d, "photo.jpg")
        src.write_text("data")
        copied = copy_files([str(src)], dst_d)
        assert len(copied) == 1 and Path(copied[0][1]).name == "photo.jpg"
        # 2回目: 同名なので _1 になる。元ファイルは残る
        copied = copy_files([str(src)], dst_d)
        assert Path(copied[0][1]).name == "photo_1.jpg"
        assert src.exists() and src.read_text() == "data"
        assert Path(dst_d, "photo.jpg").exists()
        assert Path(dst_d, "photo_1.jpg").exists()


def test_copy_files_skips_missing():
    with tempfile.TemporaryDirectory() as dst_d:
        assert copy_files(["/nonexistent/photo.jpg"], dst_d) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
