# voice-memo-kit

ボイスメモ（音声ファイル）を**ローカル WhisperX で文字起こし → Claude で文脈校閲 → Markdown** にし、
その内容を Claude Code への入力（指示・思考・メモ）として使うための共通キット。
複数プロジェクトで同じスキルを共有し、キットを更新すれば全プロジェクトに反映される。

- 文字起こし: WhisperX `large-v3`（INT8 / CUDA）。単語整列は省略して高速化
- 固有名詞補正: プロジェクト別の辞書（JSON）で initial_prompt 予告＋誤認識置換の二段補正
- 校閲: Claude Code CLI（サブスクリプション容量内・追加課金なし）。校閲前データも md 内に保持
- 出力: `<project>/voice_memos/transcripts/<録音日>_<元ファイル名>.md`

## 構成

```
voice-memo-kit/
├── skills/voice-memo/        # Claude Code スキル（この単位をプロジェクトに共有する）
│   ├── SKILL.md              # 手順: 文字起こし → 読込 → 「口述された入力」として解釈
│   ├── transcribe.ps1        # ランチャー（キットの .venv を使う）
│   ├── config.default.json   # 既定設定（パス・モデル・校閲）
│   ├── scripts/              # transcribe.py / engine.py / glossary.py / proofread.py
│   └── prompts/memo_proofread.md
├── requirements.lock.txt     # venv 再現用（pip freeze 固定）
└── .venv/                    # git 管理外。実 venv または既存 venv へのジャンクション
```

## セットアップ（1 回）

### 1. venv

WhisperX + CUDA の venv（約 7GB）。既に同等の venv があるならジャンクションで流用する:

```powershell
# 既存 venv を流用（例）
New-Item -ItemType Junction -Path .\.venv -Target "C:\path\to\existing\.venv"

# 新規に構築する場合（Python 3.10 / CUDA 12.6 wheel 前提）
python -m venv .\.venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.lock.txt
.\.venv\Scripts\python.exe -c "import torch, whisperx; print(torch.cuda.is_available())"  # True
```

環境変数 `VOICE_MEMO_PYTHON` で Python を明示指定することもできる。

### 2. スキルを全プロジェクトに共有（個人スキル）

```powershell
New-Item -ItemType Junction -Path "$env:USERPROFILE\.claude\skills\voice-memo" -Target "C:\Projects\voice-memo-kit\skills\voice-memo"
```

これで Claude Code のどのプロジェクトでも `/voice-memo` が使える。更新は `git pull` のみ。

## 置き場（2 層構造）

```
<project>/voice_memos/          # プロジェクト層（カレントディレクトリ直下）
├── inbox/        # このプロジェクト向けの音声（git 管理外）
├── transcripts/  # 出力先。グローバル層の音声の transcript もここに出る（git 管理外）
├── processed/    # 処理済み音声の退避先（git 管理外）
├── glossary/     # 任意: プロジェクト固有の辞書 *.json（共有してよい）
└── config.json   # 任意: 既定設定の上書き

~/voice_memos/                  # グローバル層（PC 全体で共有、config の global_root）
├── inbox/        # どのプロジェクトにも属さない音声（スマホ同期の着地点など）
├── processed/
└── glossary/     # PC 共通の辞書（人名・常用語）
```

- 音声: プロジェクト inbox → グローバル inbox の順に**両方**処理（プロジェクト優先）。`--no-global` で無視
- 辞書: グローバル＋プロジェクトをマージ。同じ canonical はプロジェクト側が優先し、wrong_variants は和集合
- プロジェクト辞書の探索順は `voice_memos/glossary/` → `tools/glossary/`（無ければプロジェクト辞書なし）
- `config.json` は `config.default.json` と同じキーの部分上書き（例: `{"paths": {"glossary": "tools/glossary"}}`、`{"global_root": "D:/memos"}`）

辞書 JSON の書式:

```json
{
  "title": "プロジェクト用語",
  "entries": [
    {"canonical": "正式表記", "wrong_variants": ["観測された誤認識"], "notes": "読み方・注意"}
  ]
}
```

## 使い方

```powershell
# プロジェクトルートで
& "$env:USERPROFILE\.claude\skills\voice-memo\transcribe.ps1"            # inbox を全部処理
& "$env:USERPROFILE\.claude\skills\voice-memo\transcribe.ps1" "memo.m4a" # 指定ファイル
# オプション: --timestamps / --no-glossary / --no-global / --no-proofread / --keep / --device cpu / --project <dir>
```

Claude Code では `/voice-memo` を打てば、文字起こしから内容の解釈まで一連で行う。

## ロードマップ

- **現在**: 個人スキル（ジャンクション）方式で運用
- **予定**: Claude Code プラグイン化（`.claude-plugin/plugin.json` とマーケットプレイス定義を追加し、
  `/plugin install` で導入・更新できるようにする。venv は `${CLAUDE_PLUGIN_DATA}` へ）
