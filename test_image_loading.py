"""画像読み込み・サブフォルダ走査・エラー内訳の単体テスト.

対象は read_image_bgr、iter_image_files、build_index、lambda バグの
回帰、標準ストリームの安全化、ログのフォールバック、spec の datas。

実行: pytest test_image_loading.py
（pytest が無ければ python test_image_loading.py でも簡易実行できる。
ただし lambda バグのテストはディスプレイが必要で、無い環境ではスキップする）
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from search_core import ImageReadError, normalize_embedding, read_image_bgr, search  # noqa: E402
from indexer import build_index, iter_image_files  # noqa: E402


def _make_jpg(path: Path, size: tuple[int, int] = (64, 64), color=(200, 100, 50)) -> Path:
    """PIL でダミーの JPEG を作る（中身は単色。顔は写っていない）。"""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def _fake_face(box, det_score=0.9, emb=None):
    emb = np.zeros(512, dtype=np.float32) if emb is None else np.asarray(emb, dtype=np.float32)
    return SimpleNamespace(
        bbox=np.array(box, dtype=np.float32),
        det_score=det_score,
        normed_embedding=normalize_embedding(emb),
        embedding=emb,
    )


class FakeApp:
    """InsightFace の代わり。get() が返す顔リストを固定する."""

    def __init__(self, faces=None, error: Exception | None = None):
        self.faces = faces if faces is not None else []
        self.error = error

    def get(self, img):
        if self.error is not None:
            raise self.error
        return list(self.faces)


# read_image_bgr

def test_read_image_bgr_japanese_path(tmp_path):
    """日本語のフォルダ名・ファイル名でも読めること."""
    folder = tmp_path / "写真フォルダ"
    p = _make_jpg(folder / "テスト画像.jpg")
    img = read_image_bgr(p)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8


def test_read_image_bgr_pil_fallback(monkeypatch):
    """cv2.imdecode が読めなくても PIL フォールバックで読めること."""
    import cv2

    p = _make_jpg(Path(tempfile.mkdtemp()) / "photo.jpg")
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: None)
    img = read_image_bgr(p)
    assert img.shape == (64, 64, 3)


def test_read_image_bgr_missing_raises(tmp_path):
    with pytest.raises(ImageReadError, match="見つかりません"):
        read_image_bgr(tmp_path / "ない.jpg")


def test_read_image_bgr_empty_raises(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    with pytest.raises(ImageReadError, match="空です"):
        read_image_bgr(p)


def test_read_image_bgr_corrupt_raises(tmp_path):
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"this is not an image at all, just text")
    with pytest.raises(ImageReadError):
        read_image_bgr(p)


def test_read_image_bgr_gif(tmp_path):
    """GIF も PIL 経由で読めること."""
    from PIL import Image

    p = tmp_path / "anim.gif"
    Image.new("RGB", (32, 32), (10, 200, 90)).save(p, "GIF")
    img = read_image_bgr(p)
    assert img.shape == (32, 32, 3)


# iter_image_files（サブフォルダ再帰）

def test_iter_image_files_recursive(tmp_path):
    """トップ＋サブフォルダ＋孫フォルダの jpg を全て拾うこと."""
    _make_jpg(tmp_path / "top.jpg")
    _make_jpg(tmp_path / "sub" / "sub.jpg")
    _make_jpg(tmp_path / "sub" / "deep" / "deep.jpg")
    _make_jpg(tmp_path / "sub" / "deep" / "日本語.jpg")
    (tmp_path / "note.txt").write_text("not an image")
    (tmp_path / "index.npz").write_bytes(b"fake index")
    _make_jpg(tmp_path / ".hidden" / "hidden.jpg")  # 隠しフォルダは対象外
    (tmp_path / ".hiddenfile.jpg").write_bytes(b"x")  # 隠しファイルは対象外

    files = iter_image_files(tmp_path)
    names = sorted(p.name for p in files)
    assert names == ["deep.jpg", "sub.jpg", "top.jpg", "日本語.jpg"], names


# build_index（集合写真・エラー内訳）

def test_build_index_japanese_and_nested_no_errors(tmp_path):
    """日本語パス＋ネストしたフォルダでも「読めなかった画像: 0枚」になること."""
    _make_jpg(tmp_path / "top.jpg")
    _make_jpg(tmp_path / "イベント" / "集合写真.jpg")
    _make_jpg(tmp_path / "イベント" / "2024" / "深い.jpg")
    out = tmp_path / "out" / "index.npz"
    stats = build_index(tmp_path, out, app=FakeApp([]))
    assert stats["images"] == 3
    assert stats["faces"] == 0
    assert stats["errors"] == 0
    assert stats["unreadable"] == 0
    assert stats["skipped_no_face"] == 3
    assert out.is_file()


def test_build_index_group_photo_registers_all_faces(tmp_path):
    """1枚の集合写真から3人全員が登録され、「3人中何番目」が分かること."""
    _make_jpg(tmp_path / "group.jpg", size=(300, 200))
    e0 = np.zeros(512, dtype=np.float32); e0[0] = 1.0
    e1 = np.zeros(512, dtype=np.float32); e1[1] = 1.0
    e2 = np.zeros(512, dtype=np.float32); e2[2] = 1.0
    faces = [
        _fake_face([10, 10, 40, 40], emb=e0),
        _fake_face([60, 10, 90, 40], emb=e1),
        _fake_face([110, 10, 140, 40], emb=e2),
    ]
    out = tmp_path / "index.npz"
    stats = build_index(tmp_path, out, app=FakeApp(faces))
    assert stats["images"] == 1
    assert stats["faces"] == 3
    assert stats["errors"] == 0

    from search_core import load_index

    index = load_index(out)
    assert list(index.face_no_in_image) == [0, 1, 2]
    assert list(index.faces_per_image) == [3, 3, 3]
    # 1番目の顔で検索すると本人が1位で、「3人中1番目」と表示できる
    results = search(e0, index, threshold=0.5, top_n=10)
    assert len(results) == 1
    assert results[0].face_label == "3人中 1番目"


def test_build_index_records_unreadable_files(tmp_path):
    """破損ファイルはエラーとして数え、ファイル名と理由が残ること."""
    _make_jpg(tmp_path / "ok.jpg")
    bad = tmp_path / "壊れた.jpg"
    bad.write_bytes(b"not an image")
    out = tmp_path / "index.npz"
    stats = build_index(tmp_path, out, app=FakeApp([]))
    assert stats["images"] == 2
    assert stats["errors"] == 1
    assert stats["unreadable"] == 1
    assert stats["detect_errors"] == 0
    assert "壊れた.jpg" in stats["error_files"]
    assert stats["error_details"][0]["file"] == "壊れた.jpg"
    assert stats["error_details"][0]["reason"]  # 理由が空でないこと


def test_build_index_counts_detect_errors(tmp_path):
    """顔検出中の例外は「読めなかった」とは別に数えること."""
    _make_jpg(tmp_path / "ok.jpg")
    out = tmp_path / "index.npz"
    stats = build_index(tmp_path, out, app=FakeApp(error=RuntimeError("検出エンジンの故障テスト")))
    assert stats["errors"] == 1
    assert stats["unreadable"] == 0
    assert stats["detect_errors"] == 1
    assert "検出エンジン" in stats["error_details"][0]["reason"]


# lambda バグの回帰テスト（ダイアログに本当のメッセージが出ること）

def test_index_error_dialog_shows_real_message(tmp_path):
    """インデックス作成失敗時に、ダイアログへ「None」ではなく本当の
    エラーメッセージが出ること（except 内の lambda が str(e) を
    遅延評価するバグの回帰テスト）。ディスプレイが無ければスキップ。

    バックグラウンドスレッドからの after() は、メインスレッドが
    mainloop() 実行中でないと動かないため、このテストも mainloop()
    で駆動する（実際のアプリと同じ条件）。
    """
    import tkinter as tk
    from tkinter import messagebox

    try:
        from app import FaceSearchApp
    except Exception as e:
        pytest.skip(f"app を import できませんでした: {e}")

    try:
        app = FaceSearchApp()
    except tk.TclError:
        pytest.skip("ディスプレイが無いため GUI テストをスキップします")
    _make_jpg(tmp_path / "photo.jpg")
    app.photo_folder = str(tmp_path)
    app.folder_var.set(str(tmp_path))
    # 逐次（並列数 1）で実行する。並列だとワーカーが自分のモデルを
    # 初期化するため、差し替えた ensure_face_app が呼ばれず、
    # 失敗を注入できない。
    app.workers_var.set("1")

    captured: dict = {}
    failure: list[str] = []

    def fake_showerror(title, message, **kwargs):
        captured["title"] = title
        captured["message"] = message

    orig_showerror = messagebox.showerror
    orig_ensure = app.ensure_face_app
    messagebox.showerror = fake_showerror
    app.ensure_face_app = lambda det_size=(640, 640): (_ for _ in ()).throw(
        RuntimeError("モデル読み込み失敗テストXYZ")
    )

    def check_and_quit():
        try:
            assert "message" in captured, "エラーダイアログが出ませんでした"
            assert captured["title"] == "インデックス作成に失敗しました"
            assert "モデル読み込み失敗テストXYZ" in captured["message"], captured["message"]
            assert captured["message"].strip() != "None"
            assert "face-search.log" in captured["message"]
        except AssertionError as e:
            failure.append(str(e))
        finally:
            app.destroy()

    try:
        app.after(200, app.on_build_index)
        app.after(8000, check_and_quit)
        app.mainloop()
    finally:
        messagebox.showerror = orig_showerror
        app.ensure_face_app = orig_ensure
        try:
            app.destroy()
        except Exception:
            pass

    assert not failure, failure[0] if failure else ""


# バージョン表示（新旧の見分け）

def test_app_title_and_status_show_version(tmp_path):
    """タイトルバーとステータスバーにバージョンが出ること。
    古い画面のままになる問題の見分け用。tmp_path は簡易実行時の
    スキップ判定のためだけに受ける。
    """
    import tkinter as tk

    try:
        from app import VERSION, FaceSearchApp
    except Exception as e:
        pytest.skip(f"app を import できませんでした: {e}")

    assert VERSION == "v0.7.0 (2026-11-19)"
    try:
        app = FaceSearchApp()
    except tk.TclError:
        pytest.skip("ディスプレイが無いため GUI テストをスキップします")
    try:
        assert VERSION in app.title(), app.title()
        assert VERSION in app.version_var.get(), app.version_var.get()
    finally:
        app.destroy()


def test_build_scripts_default_to_directml():
    """ビルドスクリプトの既定が DirectML であること。

    引数なしでビルドすると DirectML 版になる。GPU の自動検出
    （detect_gpu.ps1）は v0.6.8 で廃止され、NVIDIA→CUDA を選んで
    しまう問題を無くした。bat は Shift_JIS、ps1 は UTF-8+BOM で読む。
    """
    root = Path(__file__).parent
    bat = (root / "build_exe.bat").read_bytes().decode("cp932")
    ps1 = (root / "build_exe.ps1").read_bytes().decode("utf-8-sig")
    assert "detect_gpu" not in bat
    assert "detect_gpu" not in ps1
    assert 'set "PROVIDER=directml"' in bat
    assert '$Provider = "directml"' in ps1
    assert not (root / "detect_gpu.ps1").exists()


def test_build_scripts_force_reinstall_provider_package():
    """ビルドスクリプトがプロバイダ版 onnxruntime を上書き再インストールすること。

    insightface が素の onnxruntime（CPU版）も依存で入れるため、素の
    onnxruntime とプロバイダ版が同じ onnxruntime パッケージを上書き
    し合う。後にインストールされた方が勝つため、素の方が勝つと GPU
    プロバイダが登録されない。プロバイダ版を最後に
    --force-reinstall --no-deps で入れ直すことで確実に勝たせる。
    バージョンはハードコードせず入っているものを使うため、両スクリプト
    に pip show によるバージョン取得があることも確認する。
    bat は Shift_JIS、ps1 は UTF-8+BOM で読む。
    """
    root = Path(__file__).parent
    bat = (root / "build_exe.bat").read_bytes().decode("cp932")
    ps1 = (root / "build_exe.ps1").read_bytes().decode("utf-8-sig")
    for script in (bat, ps1):
        assert "--force-reinstall --no-deps" in script
        assert "onnxruntime-gpu" in script
        assert "onnxruntime-directml" in script
        assert "pip show" in script


# 標準ストリームの安全化（windowed exe で stdout/stderr が None の場合）

def _reset_face_log_state():
    """face_log のキャッシュと face_search ロガーを初期状態に戻す."""
    import logging

    import face_log

    face_log._resolved_log_path = None
    face_log._std_streams_installed = False
    logger = logging.getLogger("face_search")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def test_ensure_std_streams_redirects_none_streams(tmp_path, monkeypatch):
    """sys.stdout / sys.stderr が None でも、print() や tqdm が
    AttributeError にならず、出力が face-search.log に残ること。

    PyInstaller console=False の exe では両方が None になる。
    InsightFace の初回モデルダウンロード時に tqdm が stderr へ書いて
    "'NoneType' object has no attribute 'write'" になった。
    """
    import face_log

    monkeypatch.setattr(face_log, "exe_dir", lambda: tmp_path)
    _reset_face_log_state()
    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)
        assert sys.stdout is None and sys.stderr is None

        face_log.ensure_std_streams()

        assert sys.stdout is not None, "stdout が差し替えられていない"
        assert sys.stderr is not None, "stderr が差し替えられていない"
        assert sys.stdout.write is not None and sys.stderr.write is not None
        assert sys.stdout.flush is not None and sys.stderr.flush is not None
        assert sys.stdout.isatty() is False

        # print() が例外を出さず、ログに残ること
        print("テスト出力123")
        sys.stderr.write("エラー出力abc\n")
        sys.stdout.flush()
        sys.stderr.flush()

        # tqdm が stderr へ書いても例外にならないこと
        # （InsightFace のダウンロード進捗がこの経路）
        try:
            from tqdm import tqdm

            for _ in tqdm(range(3), file=sys.stderr, disable=False):
                pass
        except ImportError:
            pytest.skip("tqdm が無いため tqdm 部分をスキップします")

        log_file = tmp_path / "face-search.log"
        assert log_file.is_file(), "ログファイルが作られていない"
        content = log_file.read_text(encoding="utf-8")
        assert "テスト出力123" in content, content
        assert "エラー出力abc" in content, content
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        _reset_face_log_state()


def test_setup_logging_falls_back_to_temp_when_exe_dir_unwritable(tmp_path, monkeypatch):
    """exe フォルダに書けないとき、ログが一時フォルダへフォールバックし、
    log_path() がその実際の場所を返すこと。

    書けない場所の再現には「ファイル」をフォルダとして渡す
    （root でも権限では再現できないため、NotADirectoryError を使う）。
    """
    import logging
    import tempfile

    import face_log

    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_bytes(b"x")  # フォルダではなくファイル
    monkeypatch.setattr(face_log, "exe_dir", lambda: not_a_dir)
    _reset_face_log_state()
    try:
        path = face_log.log_path()
        assert path.name == "face-search.log"
        assert path.parent == Path(tempfile.gettempdir()), path

        logger = face_log.setup_logging()
        logger.info("フォールバックテスト999")
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
        content = path.read_text(encoding="utf-8")
        assert "フォールバックテスト999" in content, content
    finally:
        _reset_face_log_state()


def test_log_stream_writer_discards_tqdm_progress(tmp_path, monkeypatch):
    """_LogStreamWriter が tqdm の '\\r' 進捗をログに残さないこと。

    進捗行を全部残すと、281MB のダウンロードで数十万行になる。
    '\\n' で終わる行だけ残し、'\\r' の途中経過は破棄すること。
    """
    import face_log

    monkeypatch.setattr(face_log, "exe_dir", lambda: tmp_path)
    _reset_face_log_state()
    try:
        path = face_log.log_path()
        writer = face_log._LogStreamWriter(path)
        writer.write("\r  5%|###       | 10/200 [00:01<00:02]")
        writer.write("\r 50%|#####     | 100/200 [00:02<00:02]")
        writer.write("\r100%|##########| 200/200 [00:03<00:00]\n")
        writer.write("完了しました\n")
        writer.flush()
        content = path.read_text(encoding="utf-8")
        assert "5%" not in content, content
        assert "50%" not in content, content
        assert "100%" in content, content
        assert "完了しました" in content, content
    finally:
        _reset_face_log_state()


def _load_spec_namespace():
    """face_search.spec を PyInstaller 無しで評価し、名前空間を返す。

    PyInstaller 本体と Analysis/PYZ/EXE/COLLECT をスタブに置き換えて
    spec を exec する。
    """
    import types
    from unittest.mock import MagicMock

    spec_path = Path(__file__).with_name("face_search.spec")
    pyinstaller = types.ModuleType("PyInstaller")
    utils = types.ModuleType("PyInstaller.utils")
    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_all = lambda pkg: ([], [], [])
    pyinstaller.utils = utils
    utils.hooks = hooks
    stubbed = {
        "PyInstaller": pyinstaller,
        "PyInstaller.utils": utils,
        "PyInstaller.utils.hooks": hooks,
    }
    saved = {name: sys.modules.get(name) for name in stubbed}
    sys.modules.update(stubbed)
    try:
        ns = {
            "__file__": str(spec_path),
            "__name__": "face_search_spec_test",
            "Analysis": MagicMock(),
            "PYZ": MagicMock(),
            "EXE": MagicMock(),
            "COLLECT": MagicMock(),
        }
        code = compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec")
        exec(code, ns)
        return ns
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _load_spec_datas():
    """face_search.spec を PyInstaller 無しで評価し、datas を返す。"""
    return list(_load_spec_namespace()["datas"])


def test_spec_contains_objects_datas_entry():
    """spec に meanshape_68.pkl を _MEIPASS/objects へ置く記述があること。"""
    text = Path(__file__).with_name("face_search.spec").read_text(encoding="utf-8")
    assert "meanshape_68.pkl" in text, text
    assert '"objects"' in text, text
    assert "find_spec" in text, text


def test_spec_datas_resolves_meanshape_68():
    """spec 評価後の datas に objects 宛てのエントリがあり、実ファイルが存在すること。

    insightface が無い環境ではスキップする。
    """
    import importlib.util

    if importlib.util.find_spec("insightface") is None:
        pytest.skip("insightface がインストールされていません")
    datas = _load_spec_datas()
    entries = [(src, dest) for src, dest in datas if dest == "objects"]
    assert entries, f"objects 宛ての datas がありません: {datas}"
    for src, _dest in entries:
        pkl = Path(src) / "meanshape_68.pkl"
        assert pkl.is_file(), f"meanshape_68.pkl がありません: {pkl}"


def test_insightface_objects_file_exists():
    """venv 内の insightface/data/objects/meanshape_68.pkl が実在すること。"""
    import importlib.util

    spec = importlib.util.find_spec("insightface")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("insightface がインストールされていません")
    pkg_dir = Path(str(spec.submodule_search_locations[0]))
    pkl = pkg_dir / "data" / "objects" / "meanshape_68.pkl"
    assert pkl.is_file(), f"meanshape_68.pkl がありません: {pkl}"


# nvidia 実行時 DLL の同梱

def test_spec_contains_nvidia_dll_collection():
    """spec に nvidia 実行時 DLL を binaries へ回収する記述があること。"""
    text = Path(__file__).with_name("face_search.spec").read_text(encoding="utf-8")
    assert "_collect_nvidia_dlls" in text, text
    assert "nvidia" in text, text
    assert '"bin"' in text or "'bin'" in text, text
    assert "onnxruntime-gpu" in text, text


def test_spec_nvidia_dlls_resolve_from_fake_dir(tmp_path):
    """_collect_nvidia_dlls が偽の nvidia ディレクトリから DLL を返すこと。

    bin 配下の .dll だけを集め、bin 外の .dll や .lib は拾わないこと。
    場所が無い・空のときは空リストになること。
    """
    ns = _load_spec_namespace()
    collect = ns["_collect_nvidia_dlls"]

    nvidia_dir = tmp_path / "nvidia"
    (nvidia_dir / "cublas" / "bin").mkdir(parents=True)
    (nvidia_dir / "cudnn" / "bin").mkdir(parents=True)
    (nvidia_dir / "cublas" / "lib").mkdir(parents=True)
    (nvidia_dir / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"fake")
    (nvidia_dir / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"fake")
    (nvidia_dir / "cudnn" / "bin" / "cudnn64_9.dll").write_bytes(b"fake")
    # bin の外・DLL 以外は拾わないこと
    (nvidia_dir / "cublas" / "bin" / "README.txt").write_text("not a dll")
    (nvidia_dir / "cublas" / "lib" / "cublas.lib").write_bytes(b"fake")
    (nvidia_dir / "stray.dll").write_bytes(b"fake")

    found = collect([str(nvidia_dir)])
    names = sorted(Path(src).name for src, _dest in found)
    assert names == ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"], found
    assert all(dest == "." for _src, dest in found), found

    # CUDA 13 の統合レイアウト（bin/<arch>/ 配下）も拾うこと
    (nvidia_dir / "cu12" / "bin" / "x86_64").mkdir(parents=True)
    (nvidia_dir / "cu12" / "bin" / "x86_64" / "cublas64_12.dll").write_bytes(b"fake")
    found2 = collect([str(nvidia_dir)])
    assert len(found2) == 4, found2

    assert collect([]) == []
    assert collect([str(tmp_path / "no-such-dir")]) == []
    # 未指定時は環境の nvidia を探す。結果は環境依存のため、
    # リストが返ることだけ確認する。
    assert isinstance(collect(None), list)


def test_spec_is_onnxruntime_gpu_build(monkeypatch):
    """onnxruntime-gpu の有無で GPU ビルド判定が変わること。"""
    import importlib.metadata

    ns = _load_spec_namespace()
    is_gpu = ns["_is_onnxruntime_gpu_build"]

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.26.0")
    assert is_gpu() is True

    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert is_gpu() is False


def test_create_face_app_logs_available_providers_and_device(tmp_path, caplog):
    """create_face_app が利用可能プロバイダ一覧と推論デバイスをログに出すこと。

    実モデルで確認する。モデルが無ければスキップ。
    CPU 環境では一覧に CUDA は出ず、推論デバイスは CPU になる。
    """
    import importlib.util
    import logging

    if importlib.util.find_spec("insightface") is None or not _model_available():
        pytest.skip("InsightFace の実モデルがありません")
    from indexer import create_face_app

    with caplog.at_level(logging.INFO, logger="face_search"):
        app = create_face_app(det_size=(640, 640))
    assert getattr(app, "used_provider", None) == "cpu"
    text = caplog.text
    assert "利用可能なプロバイダ" in text, text
    assert "CPUExecutionProvider" in text, text
    assert "推論デバイス" in text, text
    assert "CPU" in text, text


# 並列インデックス作成

def _model_available() -> bool:
    """実モデル（buffalo_l）が ~/.insightface にあるか."""
    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    return (model_dir / "det_10g.onnx").is_file() and (model_dir / "w600k_r50.onnx").is_file()


def _real_photo(name: str) -> Path:
    """ベンチマーク兼テスト用の実写真。AI 生成の顔写真で、InsightFace が
    顔を検出できることを確認済み。
    """
    p = (
        Path(__file__).parent.parent
        / "ts-spaces" / "yan-jian-suo-guiimezi" / "assets" / name
    )
    return p


def test_build_index_parallel_matches_sequential(tmp_path):
    """ワーカー2での並列 build_index が逐次と同じ結果を返すこと。

    実モデルで実写真（集合写真3枚＋顔写真1枚）を使い、顔数・画像パス・
    顔枠・特徴量が一致することを確認する。モデルが無ければスキップ。
    """
    import importlib.util

    if importlib.util.find_spec("insightface") is None or not _model_available():
        pytest.skip("InsightFace の実モデルがありません")
    group = _real_photo("group-photo.webp")
    query = _real_photo("query-face.webp")
    if not group.is_file() or not query.is_file():
        pytest.skip("テスト用の実写真がありません")
    import shutil

    folder = tmp_path / "photos"
    folder.mkdir()
    for i in range(3):
        shutil.copy(group, folder / f"group{i}.webp")
    shutil.copy(query, folder / "portrait.webp")

    out_seq = tmp_path / "seq" / "index.npz"
    out_par = tmp_path / "par" / "index.npz"
    stats_seq = build_index(folder, out_seq, workers=1, rebuild=True)
    stats_par = build_index(folder, out_par, workers=2, rebuild=True)
    assert stats_seq["faces"] == stats_par["faces"] > 0
    assert stats_seq["images"] == stats_par["images"] == 4

    from search_core import load_index

    idx_seq = load_index(out_seq)
    idx_par = load_index(out_par)
    assert idx_seq.image_paths == idx_par.image_paths
    assert np.allclose(idx_seq.boxes, idx_par.boxes, atol=1e-4)
    assert np.allclose(idx_seq.embeddings, idx_par.embeddings, atol=1e-4)
    assert np.allclose(idx_seq.scores, idx_par.scores, atol=1e-4)
    assert list(idx_seq.face_no_in_image) == list(idx_par.face_no_in_image)
    assert list(idx_seq.faces_per_image) == list(idx_par.faces_per_image)
    # 並列側の stats にデバイス・ワーカー数が残っていること
    assert stats_par["workers"] == 2
    assert stats_par["provider"] == "cpu"


def test_build_index_timing_stats(tmp_path):
    """build_index の戻り値に段階別の時間計測が入ること."""
    _make_jpg(tmp_path / "a.jpg")
    _make_jpg(tmp_path / "b.jpg")
    stats = build_index(tmp_path, tmp_path / "index.npz", app=FakeApp([]))
    timing = stats["timing"]
    assert timing["total_seconds"] >= 0
    assert timing["images_per_sec"] > 0
    assert timing["faces_per_sec"] >= 0
    assert timing["slowest_stage"] in ("読み込み", "検出", "ランドマーク", "特徴量化", "性別年齢推定")
    for key in ("load_seconds", "detect_seconds", "landmark_seconds",
                "recognition_seconds", "genderage_seconds"):
        assert timing[key] >= 0


# 差分更新

def test_build_index_differential_update(tmp_path, monkeypatch):
    """差分更新で、変更なしファイルが再利用され、変更ファイルだけ再処理され、
    削除ファイルが除かれること。

    処理したファイルを記録するため read_image_bgr をラッパーで包む。
    FakeApp は全ての画像に同じ3顔を返す。
    """
    import indexer
    from search_core import load_index, read_image_bgr as real_read

    processed: list[str] = []

    def recording_read(path):
        processed.append(str(path))
        return real_read(path)

    monkeypatch.setattr(indexer, "read_image_bgr", recording_read)

    e0 = np.zeros(512, dtype=np.float32); e0[0] = 1.0
    e1 = np.zeros(512, dtype=np.float32); e1[1] = 1.0
    e2 = np.zeros(512, dtype=np.float32); e2[2] = 1.0
    faces = [
        _fake_face([10, 10, 40, 40], emb=e0),
        _fake_face([60, 10, 90, 40], emb=e1),
        _fake_face([110, 10, 140, 40], emb=e2),
    ]
    _make_jpg(tmp_path / "a.jpg")
    _make_jpg(tmp_path / "b.jpg")
    _make_jpg(tmp_path / "c.jpg")
    out = tmp_path / "index.npz"

    # 1回目：全て処理される
    stats1 = build_index(tmp_path, out, app=FakeApp(faces))
    assert stats1["images"] == 3
    assert stats1["faces"] == 9
    assert stats1["new_files"] == 3
    assert stats1["reused_files"] == 0
    assert len(processed) == 3
    assert (tmp_path / "index.meta.json").is_file()
    first = load_index(out)
    emb_a_first = [row.copy() for p, row in zip(first.image_paths, first.embeddings) if p.endswith("a.jpg")]

    # 2回目：何も変わっていないので全て再利用、処理は0件
    processed.clear()
    stats2 = build_index(tmp_path, out, app=FakeApp(faces))
    assert stats2["reused_files"] == 3
    assert stats2["new_files"] == 0
    assert stats2["changed_files"] == 0
    assert stats2["deleted_files"] == 0
    assert stats2["faces"] == 9
    assert processed == []
    second = load_index(out)
    emb_a_second = [row.copy() for p, row in zip(second.image_paths, second.embeddings) if p.endswith("a.jpg")]
    assert len(emb_a_first) == len(emb_a_second) == 3
    for r1, r2 in zip(emb_a_first, emb_a_second):
        assert np.allclose(r1, r2)

    # b を変更（中身を変えてサイズを変える）、c を削除、d を追加
    _make_jpg(tmp_path / "b.jpg", size=(128, 128), color=(10, 20, 30))
    (tmp_path / "c.jpg").unlink()
    _make_jpg(tmp_path / "d.jpg")
    processed.clear()
    stats3 = build_index(tmp_path, out, app=FakeApp(faces))
    assert stats3["reused_files"] == 1, stats3  # a だけ
    assert stats3["changed_files"] == 1, stats3  # b
    assert stats3["new_files"] == 1, stats3  # d
    assert stats3["deleted_files"] == 1, stats3  # c
    assert stats3["images"] == 3
    assert stats3["faces"] == 9
    assert sorted(Path(p).name for p in processed) == ["b.jpg", "d.jpg"]
    third = load_index(out)
    assert not any(p.endswith("c.jpg") for p in third.image_paths)

    # 作り直し：全て処理される
    processed.clear()
    stats4 = build_index(tmp_path, out, app=FakeApp(faces), rebuild=True)
    assert stats4["reused_files"] == 0
    assert stats4["new_files"] == 3
    assert len(processed) == 3

    # 条件（最小顔サイズ）を変えると全作り直しになる
    processed.clear()
    stats5 = build_index(tmp_path, out, app=FakeApp(faces), min_face_size=50)
    assert stats5["reused_files"] == 0
    assert len(processed) == 3


# GPU フォールバック

def test_detect_runtime_provider_prefers_cuda_then_directml(monkeypatch):
    """実行環境のプロバイダ自動検出が CUDA > DirectML > CPU の順になること。

    onnxruntime の一覧を差し替えて判定だけを試す（モデル読み込みはしない）。
    """
    import indexer

    monkeypatch.setattr(
        indexer, "available_providers",
        lambda: {"CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"},
    )
    assert indexer.detect_runtime_provider() == "cuda"
    monkeypatch.setattr(
        indexer, "available_providers",
        lambda: {"CPUExecutionProvider", "DmlExecutionProvider"},
    )
    assert indexer.detect_runtime_provider() == "directml"
    monkeypatch.setattr(indexer, "available_providers", lambda: {"CPUExecutionProvider"})
    assert indexer.detect_runtime_provider() == "cpu"
    monkeypatch.setattr(indexer, "available_providers", lambda: set())
    assert indexer.detect_runtime_provider() == "cpu"


def test_create_face_app_auto_detects_cpu_without_gpu(tmp_path):
    """GPU が無い環境では、create_face_app() が CPU を自動検出して
    例外を出さないこと。実モデルで確認する。モデルが無ければスキップ。"""
    import importlib.util

    if importlib.util.find_spec("insightface") is None or not _model_available():
        pytest.skip("InsightFace の実モデルがありません")
    from indexer import create_face_app

    app = create_face_app(det_size=(640, 640))
    assert getattr(app, "used_provider", None) == "cpu"
    assert getattr(app, "provider_fallback_note", None) is None


# 処理なし表示・ワーカーのログ・メイン側のプロバイダ検出ログ

def test_build_index_all_reused_does_not_report_cpu_as_device(tmp_path, caplog):
    """全件再利用のとき、集計ログと戻り値が「デバイス CPU」と出ないこと。

    処理対象が無いのに used_provider が CPU で初期化されていたため、
    「デバイス CPU」と出て GPU が使われていないと誤解された。
    処理なしの場合は未使用であることと検出されたデバイスを分けて出す。
    """
    import logging

    _make_jpg(tmp_path / "a.jpg")
    _make_jpg(tmp_path / "b.jpg")
    out = tmp_path / "index.npz"
    e0 = np.zeros(512, dtype=np.float32); e0[0] = 1.0
    faces = [_fake_face([10, 10, 40, 40], emb=e0)]
    stats1 = build_index(tmp_path, out, app=FakeApp(faces))
    assert stats1["provider"] == "cpu"  # 処理したので実際に使ったデバイス
    assert stats1["detected_provider"] == "cpu"

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="face_search"):
        stats2 = build_index(tmp_path, out, app=FakeApp(faces))
    assert stats2["reused_files"] == 2
    assert stats2["new_files"] == 0
    assert stats2["provider"] is None
    assert stats2["detected_provider"] == "cpu"
    text = caplog.text
    assert "検出されたデバイス: CPU" in text, text
    assert "処理なし（全て再利用）のため未使用（検出されたデバイス: CPU）" in text, text
    assert "、デバイス CPU" not in text, text


def test_build_index_logs_detected_device_on_main_process(tmp_path, caplog):
    """build_index がメイン側でプロバイダ検出をログに出すこと。

    並列時はワーカーのログだけに頼っていたため、処理対象が無い
    （全て再利用）場合にプロバイダ検出が一度も走らなかった。
    メイン側でも「利用可能なプロバイダ:」「検出されたデバイス:」を出す。
    """
    import logging

    _make_jpg(tmp_path / "a.jpg")
    with caplog.at_level(logging.INFO, logger="face_search"):
        build_index(tmp_path, tmp_path / "index.npz", app=FakeApp([]))
    text = caplog.text
    assert "利用可能なプロバイダ" in text, text
    assert "CPUExecutionProvider" in text, text
    assert "検出されたデバイス: CPU" in text, text


def test_index_worker_init_sets_up_logging(tmp_path, monkeypatch):
    """ワーカー初期化で face-search.log へのハンドラが付くこと。

    spawn ワーカーはロガーのハンドラを持たないため、create_face_app の
    「利用可能なプロバイダ:」「推論デバイス:」やフォールバック警告が
    ログに届かなかった。_index_worker_init が setup_logging() を呼ぶ
    ことを確認する。実モデルは使わず create_face_app を差し替える。
    """
    import logging

    import face_log
    import indexer

    monkeypatch.setattr(face_log, "exe_dir", lambda: tmp_path)
    _reset_face_log_state()
    fake = SimpleNamespace(
        used_provider="cpu", provider_fallback_note=None,
        det_model=None, models={},
    )
    monkeypatch.setattr(indexer, "create_face_app", lambda det_size=(640, 640): fake)
    called = []
    monkeypatch.setattr(indexer, "ensure_std_streams", lambda: called.append(True))
    try:
        indexer._index_worker_init((640, 640), 20, 0.5)
        assert called, "ワーカー初期化で ensure_std_streams が呼ばれていない"
        logger = logging.getLogger("face_search")
        handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert handlers, "ワーカーに FileHandler が付いていない"
        assert Path(handlers[0].baseFilename) == tmp_path / "face-search.log"
        logger.info("ワーカーのログ999")
        for h in handlers:
            h.flush()
        content = (tmp_path / "face-search.log").read_text(encoding="utf-8")
        assert "ワーカーのログ999" in content, content
        assert indexer._WORKER_STATE.get("provider") == "cpu"
    finally:
        _reset_face_log_state()


# CUDA 実行時 DLL のプリロード

def _make_fake_meipass(tmp_path, dlls):
    """偽の _MEIPASS ツリーを作る。dlls は (subdir, filename) の一覧。"""
    for sub, name in dlls:
        d = tmp_path / sub if sub else tmp_path
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"")


def _set_frozen(monkeypatch, tmp_path):
    """_preload_cuda_dlls が凍結 exe として動くよう偽装する。"""
    import sys

    import indexer

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(indexer, "_CUDA_DLLS_PRELOADED", False)


def test_preload_cuda_dlls_loads_in_dependency_order(tmp_path, monkeypatch):
    """偽の _MEIPASS で、正しいフルパス・正しい順序で読まれること。

    onnxruntime_providers_cuda.dll は共有DLLを静的importするため、
    依存順（共有DLL → nvidia DLL → プロバイダDLL）でフルパス指定して
    先に読み込まないと LoadLibrary がエラー126で失敗する。
    ctypes.CDLL を差し替えて呼び出し順を確認する。
    """
    import ctypes

    import indexer

    _make_fake_meipass(tmp_path, indexer._CUDA_PRELOAD_DLLS)
    _set_frozen(monkeypatch, tmp_path)
    calls = []

    def fake_cdll(path):
        calls.append(path)
        return object()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    n = indexer._preload_cuda_dlls()
    assert n == len(indexer._CUDA_PRELOAD_DLLS)
    expected = [
        str(tmp_path / sub / name) if sub else str(tmp_path / name)
        for sub, name in indexer._CUDA_PRELOAD_DLLS
    ]
    assert calls == expected, calls
    # 共有DLLが先、プロバイダDLLが最後であること
    assert calls[0].endswith("onnxruntime_providers_shared.dll")
    assert calls[-1].endswith("onnxruntime_providers_cuda.dll")


def test_preload_cuda_dlls_skips_silently_when_no_files(tmp_path, monkeypatch):
    """ファイルが無いとき無言スキップ（例外なし・0件）すること。

    CPU ビルドでは nvidia DLL が無くて正常。
    """
    import ctypes

    import indexer

    _set_frozen(monkeypatch, tmp_path)  # 空のツリー
    calls = []
    monkeypatch.setattr(ctypes, "CDLL", lambda p: calls.append(p))
    assert indexer._preload_cuda_dlls() == 0
    assert calls == []


def test_preload_cuda_dlls_does_nothing_when_not_frozen(tmp_path, monkeypatch):
    """凍結 exe でない（通常の python 実行）ときは何もしないこと。"""
    import ctypes
    import sys

    import indexer

    monkeypatch.delattr(sys, "frozen", raising=False)
    calls = []
    monkeypatch.setattr(ctypes, "CDLL", lambda p: calls.append(p))
    assert indexer._preload_cuda_dlls() == 0
    assert calls == []


def test_preload_cuda_dlls_continues_on_failure(tmp_path, monkeypatch, caplog):
    """CDLL が失敗しても例外を出さず警告ログを出して続行すること。

    1件の読み込み失敗で CPU フォールバックまで壊してはいけない。
    失敗した DLL 以外は読み込み、件数は成功分だけ返す。
    """
    import ctypes
    import logging

    import indexer

    _make_fake_meipass(tmp_path, indexer._CUDA_PRELOAD_DLLS)
    _set_frozen(monkeypatch, tmp_path)
    calls = []

    def fake_cdll(path):
        calls.append(path)
        if "cudart64_13.dll" in path:
            raise OSError("error 126")
        return object()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    with caplog.at_level(logging.WARNING, logger="face_search"):
        n = indexer._preload_cuda_dlls()
    assert n == len(indexer._CUDA_PRELOAD_DLLS) - 1
    assert len(calls) == len(indexer._CUDA_PRELOAD_DLLS)
    assert "プリロードに失敗" in caplog.text, caplog.text
    assert "cudart64_13.dll" in caplog.text, caplog.text


def test_describe_load_error_formats_cause_and_winerror():
    """_describe_load_error が原因と winerror を出すこと。

    PyInstaller の pyimod03_ctypes は元の OSError を
    `raise PyInstallerImportError(name) from base_error` で包むため、
    そのままでは本当の Windows エラーコードが分からない。
    __cause__ があれば原因も出し、winerror が分かれば意味も付ける。
    """
    import indexer

    plain = OSError("boom")
    assert indexer._describe_load_error(plain) == "boom"

    with_win = OSError("boom")
    with_win.winerror = 127
    text = indexer._describe_load_error(with_win)
    assert "winerror=127" in text, text
    assert "プロシージャが見つからない" in text, text

    cause = OSError("指定されたモジュールが見つかりません。")
    cause.winerror = 126
    wrapped = OSError("Failed to load dynlib/dll 'x.dll'.")
    wrapped.__cause__ = cause
    text = indexer._describe_load_error(wrapped)
    assert "原因:" in text, text
    assert "winerror=126" in text, text
    assert "モジュールが見つからない" in text, text


def test_preload_failure_log_includes_cause_winerror(
    tmp_path, monkeypatch, caplog
):
    """プリロード失敗のログに元の OSError（winerror）が残ること。

    CDLL が PyInstallerImportError 風の例外（__cause__ 付き OSError）
    を投げたとき、ログに原因と winerror=126 等が残る。包まれた後の
    文言だけでは 126/127/1114/193 の区別が付かないため。
    """
    import ctypes
    import logging

    import indexer

    _make_fake_meipass(tmp_path, indexer._CUDA_PRELOAD_DLLS)
    _set_frozen(monkeypatch, tmp_path)

    def fake_cdll(path):
        if "onnxruntime_providers_cuda.dll" in path:
            cause = OSError("指定されたモジュールが見つかりません。")
            cause.winerror = 126
            err = OSError(
                "Failed to load dynlib/dll "
                "'...onnxruntime_providers_cuda.dll'. Most likely this "
                "dynlib/dll was not found when the application was frozen."
            )
            err.__cause__ = cause
            raise err
        return object()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    with caplog.at_level(logging.WARNING, logger="face_search"):
        n = indexer._preload_cuda_dlls()
    assert n == len(indexer._CUDA_PRELOAD_DLLS) - 1
    assert "onnxruntime_providers_cuda.dll" in caplog.text, caplog.text
    assert "winerror=126" in caplog.text, caplog.text
    assert "モジュールが見つからない" in caplog.text, caplog.text


def test_preload_list_uses_real_cudnn_sub_dll_names():
    """プリロード一覧の cudnn サブDLL名が実際のファイル名であること。

    cuDNN 9 の実際の名前は統合名（cudnn_ops64_9.dll 等）で、_infer/_train
    付きは cuDNN 8 の名前。実在しない名前を挙げるとスキップされる。
    実物の wheel 内容で確認。
    """
    import indexer

    names = [name for _, name in indexer._CUDA_PRELOAD_DLLS]
    for expected in (
        "cudnn_ops64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_engines_tensor_ir64_9.dll",
        "cudnn_ext64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn64_9.dll",
    ):
        assert expected in names, expected
    for old in (
        "cudnn_engines64_9.dll",
        "cudnn_ops_infer64_9.dll",
        "cudnn_ops_train64_9.dll",
        "cudnn_adv_infer64_9.dll",
        "cudnn_adv_train64_9.dll",
        "cudnn_cnn_infer64_9.dll",
        "cudnn_cnn_train64_9.dll",
    ):
        assert old not in names, old


def test_preload_list_uses_cuda13_dll_names():
    """プリロード一覧が CUDA 13 ビルドのファイル名であること。

    onnxruntime-gpu 1.32.0（CUDA 13）の実物の wheel 内容で確認。
    cuDNN は CUDA 13 用も 9.x。cu12 系の名前が残っていないこと。
    """
    import indexer

    names = [name for _, name in indexer._CUDA_PRELOAD_DLLS]
    for expected in (
        "cublas64_13.dll",
        "cublasLt64_13.dll",
        "cudart64_13.dll",
        "cufft64_12.dll",
        "curand64_10.dll",
        "cudnn64_9.dll",
    ):
        assert expected in names, expected
    for old in (
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
        "cufft64_11.dll",
    ):
        assert old not in names, old


def test_requirements_gpu_cuda_requires_1_27_or_later():
    """requirements-gpu-cuda.txt が 1.27 以上を指定し <1.27 が無いこと。

    CUDA 12.8 ビルド（1.26.x まで）は CUDA 13 系ドライバでは
    WinError 1114 で動かないため。"""
    import re
    from pathlib import Path

    text = (Path(__file__).parent / "requirements-gpu-cuda.txt").read_text(
        encoding="utf-8"
    )
    assert "<1.27" not in text, text
    assert re.search(r"onnxruntime-gpu\[cuda,cudnn\]>=1\.27,<1\.33", text), text


# GUI の新しい設定

def test_app_build_options_defaults(tmp_path):
    """GUI に並列数・作り直しがあり、既定値と丸めが正しいこと。
    デバイス選択は GUI には無い（ビルド時に決まるため）。"""
    import tkinter as tk

    try:
        from app import FaceSearchApp
    except Exception as e:
        pytest.skip(f"app を import できませんでした: {e}")
    try:
        app = FaceSearchApp()
    except tk.TclError:
        pytest.skip("ディスプレイが無いため GUI テストをスキップします")
    try:
        assert app.workers() == 4
        assert app.rebuild_var.get() is False
        assert not hasattr(app, "device_var")
        # 並列数は 1〜8 に丸める
        app.workers_var.set("8")
        assert app.workers() == 8
        app.workers_var.set("99")
        assert app.workers() == 8
        app.workers_var.set("0")
        assert app.workers() == 1
    finally:
        app.destroy()


def test_ensure_face_app_returns_created_app(tmp_path, monkeypatch):
    """ensure_face_app が作成したアプリを返すこと。

    以前は return 文が無く常に None を返していたため、クエリ
    読み込みで `None.get` となり `'NoneType' object has no attribute
    'get'` で落ちていた（search_core.py:271）。インデックス作成は
    build_index が app=None 時に自分で作るため動いていた。
    実モデルは使わず create_face_app を差し替える。同じ det_size の
    2回目の呼び出しは同じオブジェクトを返す（再作成しない）こと。
    """
    import tkinter as tk
    from types import SimpleNamespace

    try:
        import app as app_module
        from app import FaceSearchApp
    except Exception as e:
        pytest.skip(f"app を import できませんでした: {e}")
    try:
        gui = FaceSearchApp()
    except tk.TclError:
        pytest.skip("ディスプレイが無いため GUI テストをスキップします")
    try:
        fake = SimpleNamespace(provider_fallback_note=None)
        calls = []

        def _fake_create(det_size=(640, 640)):
            calls.append(det_size)
            return fake

        monkeypatch.setattr(app_module, "create_face_app", _fake_create)
        first = gui.ensure_face_app((640, 640))
        assert first is fake, "ensure_face_app が作成したアプリを返していない"
        assert first is not None
        second = gui.ensure_face_app((640, 640))
        assert second is fake
        assert len(calls) == 1, f"同じ det_size で再作成されている: {calls}"
    finally:
        gui.destroy()


if __name__ == "__main__":
    # pytest が無くても簡易実行できる（lambda バグのテストは tkinter が
    # 必要で、ディスプレイが無ければスキップ扱い）。
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        if "monkeypatch" in t.__code__.co_varnames or "tmp_path" in t.__code__.co_varnames:
            # 簡易実行では tmp_path / monkeypatch を用意できないため飛ばす
            print(f"SKIP {t.__name__} (pytest で実行してください)")
            continue
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed (一部スキップあり)")
    sys.exit(1 if failed else 0)
