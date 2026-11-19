# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 用 spec: 顔類似検索プロトタイプを Windows exe 化する。

使い方: build_exe.bat をダブルクリックするか、PowerShell で
    .\\build_exe.ps1 を実行する。手動なら pyinstaller face_search.spec。

出来上がりは onedir 形式: dist\\face-search\\face-search.exe と
同じフォルダ内の DLL・データ一式。exe だけコピーしても動かない。
onedir の理由: 全体が 500MB 前後になるため、onefile にすると起動の
たびに全展開して 10〜30 秒以上かかる。onedir なら展開不要で起動が速い。
onefile にしたい場合は、下部の EXE の exclude_binaries を False にし、
COLLECT を削除すること（欠点は README の「exe化」の章を参照）。

InsightFace のモデル buffalo_l（展開後 326MB）は同梱しない。
初回起動時に %USERPROFILE%\\.insightface へダウンロードされる。
"""

from PyInstaller.utils.hooks import collect_all

import importlib.util
import os
from pathlib import Path

datas = []
binaries = []
hiddenimports = [
    # アプリ本体
    "indexer",
    "search_core",
    "paths",
    # InsightFace 本体と、動的 import されるサブモジュール
    "insightface",
    "insightface.app",
    "insightface.app.face_analysis",
    "insightface.app.common",
    "insightface.model_zoo",
    "insightface.model_zoo.model_zoo",
    "insightface.model_zoo.onnxruntime_utils",
    "insightface.utils",
    "insightface.addons",
    # 推論・画像処理まわり
    "onnxruntime",
    "onnx",
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageTk",
    "tkinter",
    # insightface 2.1 の依存パッケージ
    "scipy",
    "skimage",
    "tqdm",
    "requests",
    "flatbuffers",
    "protobuf",
    "packaging",
]

# 動的ロードされる .pyd/.dll・データファイルをまとめて回収する。
# フックだけでは onnxruntime / cv2 / scipy のバイナリが拾いきれないため。
for _pkg in ("insightface", "onnxruntime", "onnx", "cv2", "scipy", "skimage"):
    _datas, _binaries, _hidden = collect_all(_pkg)
    datas += _datas
    binaries += _binaries
    hiddenimports += _hidden

# meanshape_68.pkl の配置
# get_object() は frozen 時に sys._MEIPASS/objects/meanshape_68.pkl を読むが、
# collect_all("insightface") は _internal/insightface/data/objects/ に置くため
# 見えない。見つからないと landmark 処理で NoneType エラーになり、
# インデックス作成・クエリ読み込みが失敗する。そこで data/objects/ の中身を
# _internal/objects/ へ別途同梱する（datas の宛先 "objects" は _MEIPASS 直下）。
# 同じファイルが2か所に同梱されるが、974B と小さいため問題ない。
try:
    _if_spec = importlib.util.find_spec("insightface")
    _if_locations = getattr(_if_spec, "submodule_search_locations", None) if _if_spec else None
    if _if_locations:
        _if_dir = str(_if_locations[0])
        _objects_dir = os.path.join(_if_dir, "data", "objects")
        if os.path.isdir(_objects_dir):
            datas.append((_objects_dir, "objects"))
        else:
            print(f"[警告] insightface の data/objects が見つかりません: {_objects_dir}")
    else:
        print("[警告] insightface が見つかりません。meanshape_68.pkl を同梱できません。")
except Exception as _exc:  # noqa: BLE001 - ビルドを止めず警告だけ出す
    print(f"[警告] insightface の data/objects の解決に失敗しました: {_exc}")

# CUDA 実行時 DLL（nvidia-*）の同梱
# onnxruntime-gpu の CUDA 実行時 DLL（cublas / cudart / cuDNN 等）は
# onnxruntime パッケージ内になく、site-packages/nvidia/<component>/bin/
# にある。collect_all("onnxruntime") は拾わないため、ここで binaries へ
# 追加する（宛先 "." は _internal 直下で、exe の DLL 探索パスに入る）。
# `pip install onnxruntime-gpu` だけでは nvidia-* は入らない
# （extras 付き onnxruntime-gpu[cuda,cudnn] が必須。
# requirements-gpu-cuda.txt 参照）。nvidia/ が無ければ空リストを返す。
def _collect_nvidia_dlls(locations=None):
    """site-packages/nvidia/ 以下の実行時 DLL を集める。

    「bin」を含むパス配下の *.dll を集める（CUDA 12 の
    nvidia/<component>/bin/ だけでなく、CUDA 13 の
    nvidia/cu<major>/bin/<arch>/ も拾うため）。無ければ空リストを返す。
    """
    if locations is None:
        try:
            _nv_spec = importlib.util.find_spec("nvidia")
            locations = getattr(_nv_spec, "submodule_search_locations", None)
        except Exception:
            locations = None
    dlls = []
    if not locations:
        return dlls
    for _base in locations:
        if not os.path.isdir(_base):
            continue
        for _root, _dirs, _files in os.walk(_base):
            if "bin" not in Path(_root).parts:
                continue
            for _name in _files:
                if _name.lower().endswith(".dll"):
                    dlls.append((os.path.join(_root, _name), "."))
    return dlls


def _is_onnxruntime_gpu_build():
    """onnxruntime-gpu が入っているか（警告を出すかの判定用）。"""
    try:
        import importlib.metadata as _md

        _md.version("onnxruntime-gpu")
        return True
    except Exception:
        return False


try:
    _nvidia_dlls = _collect_nvidia_dlls()
    if _nvidia_dlls:
        binaries += _nvidia_dlls
        print(f"[情報] nvidia 実行時 DLL を {len(_nvidia_dlls)} 件同梱します")
    elif _is_onnxruntime_gpu_build():
        print(
            "[警告] onnxruntime-gpu が入っているのに nvidia 実行時 DLL が"
            "見つかりません。CUDAExecutionProvider が一覧に出ず、GPU を試す"
            "前に CPU が選ばれます。requirements-gpu-cuda.txt の"
            "onnxruntime-gpu[cuda,cudnn] でインストールし直してください。"
        )
except Exception as _exc:  # noqa: BLE001 - ビルドを止めず警告だけ出す
    print(f"[警告] nvidia 実行時 DLL の解決に失敗しました: {_exc}")

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="face-search",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX は無効化: onnxruntime の DLL を壊すことがあるため。
    upx=False,
    # GUI アプリなのでコンソールは出さない。
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="face-search",
)
