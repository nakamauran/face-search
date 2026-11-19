# 顔類似検索プロトタイプを Windows exe 化するビルドスクリプト（PowerShell 版）。
# 使い方: プロジェクトフォルダで右クリック →「PowerShell で実行」、
# または PowerShell から:  .\build_exe.ps1 [-Provider cpu|cuda|directml]
#   指定なし: DirectML でビルドする（推奨）。
# 実行ポリシーで止まる場合は、管理者で以下を実行してからもう一度試すこと:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# 出来上がり: dist\face-search\face-search.exe（フォルダごと使うこと）

param(
    [string]$Provider = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "================================================"
Write-Host " 顔類似検索プロトタイプ - exe ビルド"
Write-Host "================================================"
Write-Host ""

# プロバイダの決定（引数 or 既定 DirectML）
# onnxruntime 系は同居できないため、exe には1種類しか同梱できない。
$validProviders = @("cpu", "cuda", "directml")
if ($Provider -ne "") {
    $Provider = $Provider.ToLower()
    if ($validProviders -notcontains $Provider) {
        Write-Host "[エラー] プロバイダの指定が正しくありません: $Provider"
        Write-Host "使い方: .\build_exe.ps1 [-Provider cpu|cuda|directml]"
        Write-Host "  指定なしなら DirectML でビルドします。"
        Read-Host "Enter キーで終了します"
        exit 1
    }
    Write-Host "プロバイダを指定されました: $Provider"
} else {
    Write-Host "プロバイダの指定がないため、DirectML でビルドします（推奨）。"
    $Provider = "directml"
}
Write-Host "ビルドするプロバイダ: $Provider"
Write-Host ""

# Python を探す（3.13.17 を優先。3.10〜3.14 に対応）
$candidates = @(
    @{ Cmd = "python"; Args = @() },
    @{ Cmd = "py"; Args = @("-3.13") },
    @{ Cmd = "py"; Args = @("-3.14") },
    @{ Cmd = "py"; Args = @("-3.12") },
    @{ Cmd = "py"; Args = @("-3.11") },
    @{ Cmd = "py"; Args = @("-3.10") },
    @{ Cmd = "py"; Args = @() }
)
$py = $null
$pyArgs = @()
$found15 = $false
foreach ($c in $candidates) {
    try {
        $ver = & $c.Cmd @($c.Args) --version 2>$null
        if ($ver -match "^Python 3\.(\d+)\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -eq 15) { $found15 = $true }
            if ($minor -ge 10 -and $minor -le 14) {
                $py = $c.Cmd
                $pyArgs = $c.Args
                Write-Host "Python: $ver"
                break
            }
        }
    } catch { }
}
if (-not $py) {
    Write-Host "[エラー] Python 3.10〜3.14 が見つかりません。"
    if ($found15) {
        Write-Host "Python 3.15 はまだ対応していません。onnxruntime / scikit-image に"
        Write-Host "3.15 用の wheel がないため、インストールできません。"
    }
    Write-Host "Python 3.13.17 を https://www.python.org/ からインストールしてください。"
    Write-Host "インストール時に「Add python.exe to PATH」にチェックを入れてください。"
    Read-Host "Enter キーで終了します"
    exit 1
}

# 仮想環境の作成（.venv-build）
if (-not (Test-Path ".venv-build")) {
    Write-Host "仮想環境 .venv-build を作成しています..."
    & $py @pyArgs -m venv .venv-build
}
& ".\.venv-build\Scripts\Activate.ps1"

# 依存パッケージのインストール
Write-Host "pip を更新しています..."
python -m pip install --upgrade pip
# 3つは同居できないため、まず全部消してから選んだものだけ入れる
Write-Host "onnxruntime 系パッケージを入れ替えています（プロバイダ: $Provider）..."
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml | Out-Null
$reqFile = switch ($Provider) {
    "cuda" { "requirements-gpu-cuda.txt" }
    "directml" { "requirements-gpu-directml.txt" }
    default { "requirements.txt" }
}
Write-Host "依存パッケージをインストールしています（$reqFile、数分かかる場合があります）..."
pip install -r $reqFile
# プロバイダ版 onnxruntime の上書き再インストール
# insightface が素の onnxruntime（CPU版）も依存で入れるため、素のものと
# プロバイダ版が同じ onnxruntime パッケージを上書きし合う。後に
# 入れた方が勝つため順序が安定せず、素の方が勝つと GPU プロバイダが
# 登録されない（v0.6.8 で発生）。プロバイダ版を最後に
# --force-reinstall --no-deps で入れ直し、確実に勝たせる（--no-deps
# なので nvidia-* 等の依存は最初のインストールのまま残る）。
# バージョンはハードコードせず、入っているものをそのまま使う。
$ortPkg = switch ($Provider) {
    "cuda" { "onnxruntime-gpu" }
    "directml" { "onnxruntime-directml" }
    default { $null }
}
if ($ortPkg) {
    $ortVer = $null
    try {
        $showOut = pip show $ortPkg 2>$null | Out-String
        if ($showOut -match '(?m)^Version:\s*(\S+)\s*$') {
            $ortVer = $Matches[1]
        }
    } catch {
        $ortVer = $null
    }
    if ($ortVer) {
        Write-Host "プロバイダ版を上書き再インストールしています（$ortPkg==$ortVer）..."
        $reinstallOk = $true
        try {
            pip install --force-reinstall --no-deps "$ortPkg==$ortVer"
            if ($LASTEXITCODE -ne 0) { $reinstallOk = $false }
        } catch {
            $reinstallOk = $false
        }
        if (-not $reinstallOk) {
            Write-Host "[エラー] $ortPkg の再インストールに失敗しました。"
            Read-Host "Enter キーで終了します"
            exit 1
        }
    } else {
        Write-Host "[警告] $ortPkg がインストールされていません。再インストールをスキップします。"
    }
}
Write-Host "PyInstaller をインストールしています..."
pip install pyinstaller

# 古いビルド成果物の削除
# PyInstaller は build/・dist/ を再利用するため、古いコードのままに
# なることがある。ビルド前に消してから --clean 付きでビルドする。
# （Set-Location $PSScriptRoot 済みのため、対象はこのスクリプトと同じフォルダ直下のみ）
foreach ($d in @("build", "dist")) {
    $p = Join-Path $PSScriptRoot $d
    if (Test-Path $p) {
        Write-Host "古い $d フォルダを削除しています..."
        Remove-Item -Recurse -Force $p
    }
}

# ビルド
Write-Host ""
Write-Host "ビルドを開始します（数分〜10分程度かかります）..."
pyinstaller --clean face_search.spec

Write-Host ""
Write-Host "================================================"
Write-Host " ビルドが完了しました。"
Write-Host ""
Write-Host " 出来上がり: $PWD\dist\face-search\face-search.exe"
Write-Host " プロバイダ: $Provider"
Write-Host ""
Write-Host " 使い方:"
Write-Host "   dist\face-search フォルダごとコピーしてください。"
Write-Host "   exe だけをコピーしても動きません。"
Write-Host "   デスクトップに置く場合は、exe を右クリックして"
Write-Host "   「ショートカットの作成」からショートカットを作ってください。"
Write-Host ""
Write-Host " 初回起動時は InsightFace のモデル（約281MB）を"
Write-Host " $env:USERPROFILE\.insightface へダウンロードします。"
Write-Host " ネットワークが必要です。数分かかる場合があります。"
Write-Host "================================================"
Read-Host "Enter キーで終了します"
