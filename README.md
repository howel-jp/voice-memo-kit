# voice-memo-kit

ボイスメモ（音声ファイル）を**ローカル WhisperX で文字起こし → Claude で文脈校閲 → Markdown** にし、
その内容を Claude Code への入力（指示・思考・メモ）として使うための共通キット。
複数プロジェクトで同じスキルを共有し、キットを更新すれば全プロジェクトに反映される。

- 文字起こし: WhisperX `large-v3`（INT8 / CUDA）。単語整列は省略して高速化
- 固有名詞補正: プロジェクト別の辞書（JSON）で initial_prompt 予告＋誤認識置換の二段補正
- 校閲: Claude Code CLI（サブスクリプション容量内・追加課金なし）。校閲前の文と、辞書の置き換え前の文（置き換えがあったとき）も md 内に保持
- 出力: `<project>/voice_memos/transcripts/<録音日>_<元ファイル名>.md`

📝 紹介記事（入れ方・使い方を順に説明）: [ボイスメモをそのままClaude Codeの指示にするスキル｜note](https://note.com/trinity_nexus/n/n7ac7edfae794)

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
├── requirements.lock.txt       # venv 再現用（pip freeze 固定）
├── tests/                      # ユニットテストとランチャーのテスト（下の「開発」）
├── tools/measure/              # 測りの道具（実行時には使わない。docs/maintenance.md）
└── docs/maintenance.md         # 保守の手引き（手動で確かめること・見張っていること・測りの道具）
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
    {"canonical": "正式表記", "wrong_variants": ["観測された誤認識"], "notes": "読み方・注意"},
    {"canonical": "優先して予告したい語", "priority": true}
  ]
}
```

`priority`（任意）: `true` の語は、initial_prompt が Whisper の上限に収まらないとき優先して届ける（次の節）。JSON の `true` のときだけ効く（`"true"` の文字列や `1` は印にならない）。

### initial_prompt の長さと、届ける語の優先

辞書の canonical は、文字起こしの前に initial_prompt（`以下は本音声で言及される固有名詞の一覧です: 語、語、…。`）として Whisper に予告する（A）。
Whisper（faster-whisper 1.2.1）が使うのは initial_prompt の**後ろ 223 トークンだけ**で、それより前は黙って捨てられる。

- **目安: 223 トークン ≒ 前置きの文 24 トークン + 語およそ 25 語。語の長さで 18〜41 語**
  （英数の短い語 41、カタカナ 4 字 37、漢字の複合語 6〜7 字 20、英数を含む長い名前 12 字 18。合成の辞書で測った数。#4）。
- 届かない語も、置き換え（B）と校閲（C）には全部使う。届かないのは Whisper への予告だけ。
- どの語が届くか:
  1. 前置きの文と全部の語が 223 トークンに収まるとき: 全部届く（0.5.0 までと同じ文字列）。
  2. 収まらず、`"priority": true` の語が無いとき: 0.5.0 までと同じ文字列を渡す。届くのは辞書の後ろの方の語で、前置きの文と前の方の語が落ちる。
  3. 収まらず、`"priority": true` の語があるとき: キットが語を選ぶ。前置きの文を入れ、優先の高い順に 1 語ずつ、合わせて 223 トークンに入るなら入れる
     （入らない語は飛ばして次の語へ）。優先の順は、印の語 → 印の無い語。同じ印のうちでは辞書の後ろの方が先。優先の高い語ほど後ろに並ぶ。
     前置きの文に 24 トークンを使うので、届く語の総数は、印の無いときより数語減ることがある。印の語が多すぎて全部は入らないときは、
     印の語の中でも辞書の後ろの方が先に入り、入らなかった印の語は `届かない語:` の一覧で `（印）` が付く
     （`[WARN]` の行の末尾は `印の語のうち N 語は入りません`。N は入らなかった印の語の数）。
     どの語も 1 語で前置きの文と合わせて 223 トークンを超える（語 1 つが 200 トークンほど）ときは 1 語も入らず、prompt は空になる（ログは `届く 0/M 語`。M は辞書の語の数）。
- 「辞書の後ろの方」は、マージの順（グローバル → `extra_glossary_dirs` → プロジェクト。層の中はファイル名の順・書いた順）で後ろの方。
  なので印の無い語は、プロジェクト > extra > グローバルの順に届きやすい。ただし、グローバルの語をプロジェクトの層で書き直した語は、
  マージの順ではグローバルの位置（前の方）のまま。届けたいなら印を付ける。同じ canonical を後の層で書き直すと、印も後の層の項目に従う
  （後の層の項目に `priority` が無ければ、印は付かない）。
- ログ（文字起こしのたびにコンソールに出る。md には書かない）:
  ```
  [A] initial_prompt 113/223 トークン、届く 15/15 語
  [A] initial_prompt 620 トークン（Whisper が使うのは後ろ 223）、届く 28/78 語、届かない 50 語
    [WARN] initial_prompt が 223 トークンを超えるため、前置きの文と 50 語は Whisper に届きません（置き換えと校閲には使います）。…
    届かない語: …
  [A] initial_prompt 223/223 トークン（印の語を優先）、届く 31/78 語（うち印 20/20）、届かない 47 語
  ```
  トークン数は、モデルを読む前に、モデルと同じ `tokenizer.json`（モデルのダウンロードで HF のキャッシュに入るもの）で数える（ネットにはつながない）。
  モデルを一度もダウンロードしていない初回などで数えられないときは `[WARN] initial_prompt のトークン数を数えられません…` が出て、
  印の語を後ろに並べた全部の語を渡す（印が無ければ 0.5.0 と同じ文字列）。
- チャンクの出力: 文字起こしの後に `[A] チャンクの出力（数え直し）: 最大 129 トークン / 7 チャンク（1 チャンクの上限 224）` が出る。
  Whisper は 1 チャンク（音声を 30 秒までに分けた区切り）に 224 トークン（initial_prompt を使い切ると 221）までしか出せず、超える分は黙って切れる。
  200 トークンを超えたチャンクがあると `[WARN]`。数は文字起こしの文を同じトークナイザで数え直した目安で、Whisper が出したトークンの数とは少し違うことがある。

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

測りの道具（`tools/measure/`）の使い方、手動で確かめること、見張っていること（保留）は [`docs/maintenance.md`](docs/maintenance.md) にまとめてある。

性能の目安（RTX 3060 Ti、large-v3 int8、273 秒の音声、2026-08-25 実測。校閲の行だけ 2026-10-24 の実測で、出どころは表の下）:

| 段階 | 所要 |
|---|---|
| torch + whisperx の import | 約 2.4 秒 |
| モデル読込（プロセス内で 1 回だけ。複数ファイルでは使い回す） | 約 15 秒 |
| 推論（VAD 含む） | 約 7〜8 秒 ＝ 実時間の 35〜40 倍 |
| Claude 校閲 | 1,000 字あたり約 25〜45 秒（45 分のメモ ≒ 12,600 字で、測った 1 回は約 7 分） |

校閲の時間は、Claude Code CLI 2.1.309・`claude-sonnet-4-6`・合成のメモの文で、2026-10-24 に 3 通りに測った:
1 回で校閲した 1,011・4,101・8,104 字で 1,000 字あたり 38〜42 秒（#4 の U4）、
12,638 字を 5 まとまりに分けて 1,000 字あたり 25〜45 秒・計 422 秒（#2 の C4）、
入った版のランチャーで約 1,820 字の 2 まとまりを 63・62 秒（#2 の C8）。同じ日の測りなので、日や混み具合で変わりうる。

推論は GPU で走っており、体感の待ち時間はモデル読込と校閲が主因。`float16` にしても推論は 7% 程度しか縮まらないため既定は `int8` のまま。

CPU のみでも動く（同条件の実測: i7-10700F でモデル読込 16 秒、推論 201 秒 ＝ 実時間の 1.4 倍）。GPU の無い PC では
`voice_memos/config.json` に `{"whisper": {"device": "cpu"}}` を書いておくと毎回 `--device cpu` を付けずに済む。

安全ガード: 校閲の結果が次のどれかに当たると使わず、校閲前の文字起こし（辞書の置き換え後）を本文にする。
frontmatter の `proofread: false` と、理由の `proofread_error`（補足は `proofread_error_detail`）で判別できる。

| `proofread_error` | 当たるもの |
|---|---|
| `empty` | 空の結果 |
| `too_short` | 入力の 60% 未満の長さ（要約・欠落の疑い） |
| `incomplete` | 一部が欠けた疑い: Claude CLI が 1 回の応答で返さなかった（JSON の `num_turns` が 1 でない）、段落（空行区切り）の数が入力より減った、または（分けて校閲したとき）入力は文の終わりで終わるのに校閲後がそうでない（切れたか、注記が足された） |
| `timeout` / `cli_error` / `error` | 時間切れ（1 回の呼び出しにつき 900 秒）/ Claude CLI のエラー / その他 |

長い文字起こし（3,000 字を超えるもの）は、3,000 字以下のまとまりに分けて順に校閲し、つなぐ
（45 分のメモ 1 回の校閲で、前の約 4 割が欠けたり、エラーで終わったりしたため。#2・#4）。

- 分け方: まとまりの数は ⌈字数 / 3,000⌉ を目安に、大きさをそろえる。境目は、段落の境目（前が文の終わり）→ 文の終わり →
  段落の境目 → 「、」・空白 の順に選ぶ。3,000 字以下の文字起こしは分けず、0.4.0 までと同じく 1 回で校閲する。
- まとまりごとに上の安全ガードを当てる。通らなかったまとまり（`empty` / `too_short` / `incomplete`、または CLI の `is_error`）は、
  2 つに分けて 1 回だけやり直す（時間切れと、CLI の終了コードのエラーはやり直さない）。
- それでも通らないまとまりがあれば、残りは校閲せず、本文全体を校閲前の文にする（`proofread: false`）。
  `proofread_error_detail` には何番目のまとまりかが前に付く（例: `まとまり 3/5: num_turns 3`）。

出力 md の `<details>`: 「校閲前（辞書の置き換え後）」は本文が校閲後のときだけ、「置き換え前（Whisper の出力）」は辞書の置き換えが 1 箇所以上あったときだけ付く（構成の詳細は `SKILL.md` §3）。

## ロードマップ

- **現在**: 個人スキル（ジャンクション）方式で運用
- **予定**: Claude Code プラグイン化（`.claude-plugin/plugin.json` とマーケットプレイス定義を追加し、
  `/plugin install` で導入・更新できるようにする。venv は `${CLAUDE_PLUGIN_DATA}` へ）
