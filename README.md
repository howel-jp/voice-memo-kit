# voice-memo-kit

ボイスメモ（音声ファイル）を**ローカル WhisperX で文字起こし → Claude で文脈校閲 → Markdown** にし、
その内容を Claude Code への入力（指示・思考・メモ）として使うための共通キット。
複数プロジェクトで同じスキルを共有し、キットを更新すれば全プロジェクトに反映される。

- 文字起こし: WhisperX `large-v3`（INT8 / CUDA）。単語整列は省略して高速化
- 固有名詞補正: プロジェクト別の辞書（JSON）で initial_prompt 予告＋誤認識置換の二段補正
- 校閲: Claude Code CLI（サブスクリプション容量内・追加課金なし）。校閲前データも md 内に保持
- 出力: `<project>/voice_memos/transcripts/<録音日>_<元ファイル名>.md`

## 構成（Claude Code プラグイン・単一スキル構成）

```
voice-memo-kit/                 # = プラグインルート = スキルのベースディレクトリ
├── .claude-plugin/plugin.json  # プラグインマニフェスト（name / version）
├── SKILL.md                    # /voice-memo の手順: 文字起こし → 読込 → 「口述された入力」として解釈
├── transcribe.ps1              # ランチャー（-DataDir <dir> で venv 置き場を指定）
├── setup.ps1                   # venv 構築 or 既存 venv へのリンク
├── config.default.json         # 既定設定（パス・モデル・校閲・グローバル層）
├── scripts/                    # transcribe.py / engine.py / glossary.py / proofread.py
├── prompts/memo_proofread.md   # Claude 校閲プロンプト（レベルB）
└── requirements.lock.txt       # venv 再現用（pip freeze 固定）
```

スキルはプラグインルート直下の `SKILL.md` 1 つ（単一スキルプラグイン）なので、呼び出し名は名前空間なしの `/voice-memo`。
**単一スキルプラグインの呼び出し名はインストール先ディレクトリ名で決まる**（検証済み）ため、プラグイン名を `voice-memo` にしている
（リポジトリ名は `voice-memo-kit`、マーケットプレイス名は `howel-jp`）。

## インストール

```shell
/plugin marketplace add howel-jp/voice-memo-kit
/plugin install voice-memo@howel-jp
```

更新: `/plugin marketplace update howel-jp` → `/plugin update voice-memo@howel-jp`（`plugin.json` の `version` が上がったときだけ配信される）。

インストール後、コマンド一覧には `/voice-memo:voice-memo`（プラグイン名:スキル名）として表示される。`/voice-memo` と打てば前方一致で解決される。
初回実行時は venv 構築の案内（`setup.ps1`）が出る。venv は `~/.claude/plugins/data/voice-memo-howel-jp/.venv` に置かれ、プラグイン更新後も保持される。

開発中・ローカル試験は `claude --plugin-dir C:\Projects\voice-memo-kit`、または
`~/.claude/skills/voice-memo` をキットへのジャンクションにすると `voice-memo@skills-dir` として自動ロードされる
（この方式では `${CLAUDE_PLUGIN_ROOT}` 等の変数は展開されないが、ランチャーのフォールバック探索で動く）。

## セットアップ（venv、1 回）

WhisperX + CUDA の Python 環境（約 7GB）が必要。初めて `/voice-memo` を実行するとランチャーが venv 不在（exit 2）を報告し、
Claude が `setup.ps1` の実行を案内する。手動で行う場合:

```powershell
# a) 既存の WhisperX 入り venv にリンク（ダウンロードなし）
& "<plugin-root>\setup.ps1" -DataDir "<CLAUDE_PLUGIN_DATA>" -LinkTo "C:\path\to\existing\.venv"

# b) 新規構築（Python 3.10 / NVIDIA GPU / ffmpeg 前提）
& "<plugin-root>\setup.ps1" -DataDir "<CLAUDE_PLUGIN_DATA>"
```

`<CLAUDE_PLUGIN_DATA>` は `~/.claude/plugins/data/<plugin-id>/`（プラグイン更新をまたいで保持される）。
`-DataDir` を省略するとプラグインルート直下の `.venv` を使う。環境変数 `VOICE_MEMO_PYTHON` で Python を直接指定することもできる。

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
- 辞書: グローバル → `extra_glossary_dirs` → プロジェクト `voice_memos/glossary/` の順にマージ。同じ canonical は後の層（プロジェクト）が優先し、wrong_variants は和集合
- **プロジェクト共有の辞書を足す**（チームで管理している用語集を、ボイスメモ以外の用途と共用するとき）:
  ```json
  { "paths": { "extra_glossary_dirs": ["glossary"] } }
  ```
  `voice_memos/config.json` に書く。パスはプロジェクトルート基準（絶対パスも可）。既定の `voice_memos/glossary/` は常に有効で、ここに列挙した辞書が**追加**される
- `config.json` は `config.default.json` と同じキーの部分上書き（例: `{"global_root": "D:/memos"}`）

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
# プロジェクトルートで（<plugin-root> はプラグインのインストール先）
& "<plugin-root>\transcribe.ps1" -DataDir "<CLAUDE_PLUGIN_DATA>"            # 両 inbox を全部処理
& "<plugin-root>\transcribe.ps1" -DataDir "<CLAUDE_PLUGIN_DATA>" "memo.m4a" # 指定ファイル
# オプション: --timestamps / --no-glossary / --no-global / --no-proofread / --keep / --device cpu / --project <dir>
```

`-DataDir` を省略しても、ランチャーは `VOICE_MEMO_PYTHON` → `~/.claude/plugins/data/*voice-memo-kit*/.venv` → `<plugin-root>/.venv` の順に venv を探す。
Claude Code では `/voice-memo` を打てば、文字起こしから内容の解釈まで一連で行う。

## 開発

```powershell
# ユニットテスト（GPU・WhisperX 不要。純粋関数と、whisperx / Claude CLI をスタブ化したロジックを検証）
python -m unittest discover -s tests

# ランチャーの引数受け渡しテスト（Python を echo スタブに差し替え。--flags / 位置引数 / -DataDir の分離を検証）
powershell -NoProfile -File tests\test_launcher.ps1
```

性能の目安（RTX 3060 Ti、large-v3 int8、273 秒の音声、2026-08-25 実測）:

| 段階 | 所要 |
|---|---|
| torch + whisperx の import | 約 2.4 秒 |
| モデル読込（プロセス内で 1 回だけ。複数ファイルでは使い回す） | 約 15 秒 |
| 推論（VAD 含む） | 約 7〜8 秒 ＝ 実時間の 35〜40 倍 |
| Claude 校閲（1,000 字） | 約 15 秒 |

推論は GPU で走っており、体感の待ち時間はモデル読込と校閲が主因。`float16` にしても推論は 7% 程度しか縮まらないため既定は `int8` のまま。

CPU のみでも動く（同条件の実測: i7-10700F でモデル読込 16 秒、推論 201 秒 ＝ 実時間の 1.4 倍）。GPU の無い PC では
`voice_memos/config.json` に `{"whisper": {"device": "cpu"}}` を書いておくと毎回 `--device cpu` を付けずに済む。

安全ガード: 校閲結果が入力の 60% 未満の長さになった場合は「要約・欠落の疑い」として棄却し、生の文字起こしを採用する（frontmatter の `proofread: false` で判別できる）。

## ロードマップ

- **現在**: 個人スキル（ジャンクション）方式で運用
- **予定**: Claude Code プラグイン化（`.claude-plugin/plugin.json` とマーケットプレイス定義を追加し、
  `/plugin install` で導入・更新できるようにする。venv は `${CLAUDE_PLUGIN_DATA}` へ）
