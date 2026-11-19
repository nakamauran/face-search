@echo off
rem 顔類似検索プロトタイプを Windows exe 化するビルドスクリプト。
rem ダブルクリックするだけで、仮想環境の作成から PyInstaller ビルドまで行う。
rem 使い方: build_exe.bat [cpu|cuda|directml]（引数なしは DirectML。推奨）
rem 出来上がり: dist\face-search\face-search.exe（フォルダごと使うこと）
cd /d %~dp0
setlocal

echo ================================================
echo  顔類似検索プロトタイプ - exe ビルド
echo ================================================
echo.

rem プロバイダの決定（引数 or 既定 DirectML）
rem onnxruntime 系は同居できないため、exe には1種類しか同梱できない。
set "PROVIDER="
if not "%~1"=="" (
  if /I "%~1"=="cpu" set "PROVIDER=cpu"
  if /I "%~1"=="cuda" set "PROVIDER=cuda"
  if /I "%~1"=="directml" set "PROVIDER=directml"
  if not defined PROVIDER (
    echo [エラー] プロバイダの指定が正しくありません: %~1
    echo 使い方: build_exe.bat [cpu ^| cuda ^| directml]
    echo   引数なしなら DirectML でビルドします。
    pause
    exit /b 1
  )
  echo プロバイダを指定されました: %PROVIDER%
) else (
  echo プロバイダの指定がないため、DirectML でビルドします（推奨）。
  set "PROVIDER=directml"
)
echo ビルドするプロバイダ: %PROVIDER%
echo.

rem Python を探す（3.13.17 を優先。3.10～3.14 に対応）
set "PY="
python --version >nul 2>nul
if not errorlevel 1 set "PY=python"
if "%PY%"=="" (
  py -3.13 --version >nul 2>nul
  if not errorlevel 1 set "PY=py -3.13"
)
if "%PY%"=="" (
  py -3.14 --version >nul 2>nul
  if not errorlevel 1 set "PY=py -3.14"
)
if "%PY%"=="" (
  py -3.12 --version >nul 2>nul
  if not errorlevel 1 set "PY=py -3.12"
)
if "%PY%"=="" (
  py -3.11 --version >nul 2>nul
  if not errorlevel 1 set "PY=py -3.11"
)
if "%PY%"=="" (
  py -3.10 --version >nul 2>nul
  if not errorlevel 1 set "PY=py -3.10"
)
if "%PY%"=="" (
  echo [エラー] Python 3.10～3.14 が見つかりません。
  echo Python 3.13.17 を https://www.python.org/ からインストールしてください。
  echo インストール時に「Add python.exe to PATH」にチェックを入れてください。
  pause
  exit /b 1
)

rem バージョンチェック（3.10～3.14 に対応）
%PY% -c "import sys; v=sys.version_info; raise SystemExit(0 if (v.major, v.minor)==(3,15) else 1)"
if not errorlevel 1 (
  echo [エラー] Python 3.15 はまだ対応していません。
  echo onnxruntime / scikit-image に 3.15 用の wheel がないため、インストールできません。
  echo Python 3.13.17 または 3.14 をインストールしてください。
  pause
  exit /b 1
)
%PY% -c "import sys; v=sys.version_info; raise SystemExit(0 if (3,10) <= (v.major, v.minor) <= (3,14) else 1)"
if errorlevel 1 (
  echo [エラー] Python のバージョンが 3.10～3.14 ではありません。
  %PY% --version
  echo Python 3.13.17 をインストールしてからもう一度実行してください。
  pause
  exit /b 1
)
for /f "tokens=*" %%v in ('%PY% --version') do echo Python: %%v
echo.

rem 仮想環境の作成（.venv-build）
if not exist .venv-build (
  echo 仮想環境 .venv-build を作成しています...
  %PY% -m venv .venv-build
  if errorlevel 1 (
    echo [エラー] 仮想環境の作成に失敗しました。
    pause
    exit /b 1
  )
)
call .venv-build\Scripts\activate.bat
if errorlevel 1 (
  echo [エラー] 仮想環境の有効化に失敗しました。
  pause
  exit /b 1
)

rem 依存パッケージのインストール
echo pip を更新しています...
python -m pip install --upgrade pip
if errorlevel 1 (
  echo [エラー] pip の更新に失敗しました。ネットワークを確認してください。
  pause
  exit /b 1
)
rem onnxruntime 系の入れ替え
rem 3つは同居できないため、まず全部消してから選んだものだけ入れる。
echo onnxruntime 系パッケージを入れ替えています（プロバイダ: %PROVIDER%）...
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>nul
if "%PROVIDER%"=="cuda" (
  set "REQFILE=requirements-gpu-cuda.txt"
) else if "%PROVIDER%"=="directml" (
  set "REQFILE=requirements-gpu-directml.txt"
) else (
  set "REQFILE=requirements.txt"
)
echo 依存パッケージをインストールしています（%REQFILE%、数分かかる場合があります）...
pip install -r %REQFILE%
if errorlevel 1 (
  echo [エラー] %REQFILE% のインストールに失敗しました。
  echo Visual C++ ビルドツールが必要な場合があります。README を参照してください。
  pause
  exit /b 1
)
rem プロバイダ版 onnxruntime の上書き再インストール
rem insightface が素の onnxruntime（CPU版）も依存で入れるため、素のものと
rem プロバイダ版が同じ onnxruntime パッケージを上書きし合う。後に
rem 入れた方が勝つため順序が安定せず、素の方が勝つと GPU プロバイダが
rem 登録されない（v0.6.8 で発生）。プロバイダ版を最後に
rem --force-reinstall --no-deps で入れ直し、確実に勝たせる（--no-deps
rem なので nvidia-* 等の依存は最初のインストールのまま残る）。
rem バージョンはハードコードせず、入っているものをそのまま使う。
set "ORTPKG="
if "%PROVIDER%"=="cuda" set "ORTPKG=onnxruntime-gpu"
if "%PROVIDER%"=="directml" set "ORTPKG=onnxruntime-directml"
if not defined ORTPKG goto :skip_ort_reinstall
set "ORTVER="
for /f "tokens=2" %%V in ('pip show %ORTPKG% 2^>nul ^| findstr /B "Version:"') do set "ORTVER=%%V"
if defined ORTVER goto :do_ort_reinstall
echo [警告] %ORTPKG% がインストールされていません。再インストールをスキップします。
goto :skip_ort_reinstall
:do_ort_reinstall
echo プロバイダ版を上書き再インストールしています（%ORTPKG%==%ORTVER%）...
pip install --force-reinstall --no-deps "%ORTPKG%==%ORTVER%"
if errorlevel 1 (
  echo [エラー] %ORTPKG% の再インストールに失敗しました。
  pause
  exit /b 1
)
:skip_ort_reinstall
echo PyInstaller をインストールしています...
pip install pyinstaller
if errorlevel 1 (
  echo [エラー] PyInstaller のインストールに失敗しました。
  pause
  exit /b 1
)

rem 古いビルド成果物の削除
rem PyInstaller は build/・dist/ を再利用するため、古いコードのままに
rem なることがある。ビルド前に消してから --clean 付きでビルドする。
rem （cd /d %~dp0 済みのため、対象はこのスクリプトと同じフォルダ直下のみ）
if exist build (
  echo 古い build フォルダを削除しています...
  rmdir /s /q build
)
if exist dist (
  echo 古い dist フォルダを削除しています...
  rmdir /s /q dist
)

rem ビルド
echo.
echo ビルドを開始します（数分～10分程度かかります）...
pyinstaller --clean face_search.spec
if errorlevel 1 (
  echo [エラー] ビルドに失敗しました。
  pause
  exit /b 1
)

echo.
echo ================================================
echo  ビルドが完了しました。
echo.
echo  出来上がり: %CD%\dist\face-search\face-search.exe
echo  プロバイダ: %PROVIDER%
echo.
echo  使い方:
echo    dist\face-search フォルダごとコピーしてください。
echo    exe だけをコピーしても動きません。
echo    デスクトップに置く場合は、exe を右クリックして
echo    「ショートカットの作成」からショートカットを作ってください。
echo.
echo  初回起動時は InsightFace のモデル（約281MB）を
echo  %USERPROFILE%\.insightface へダウンロードします。
echo  ネットワークが必要です。数分かかる場合があります。
echo ================================================
pause
