---
name: voice-memo
description: 音声ファイル（ボイスメモ）をローカル WhisperX で文字起こし＋Claude 校閲し、その内容を「ユーザーからの入力（指示・思考・メモ）」として読み取って会話・作業を始める。引数で音声パスを指定可、省略時はプロジェクトの voice_memos/inbox の音声を処理。「ボイスメモから始めて」「録音を読んで」「音声で指示する」等で起動。
---

# ボイスメモから会話を始める (voice-memo)

音声ファイルをローカルで文字起こしし、その内容を**ユーザーが口頭で渡した入力**として解釈して、
そこから会話・作業を始める手動 Skill。どのプロジェクトでも同じ手順で使える（voice-memo-kit）。

**目的（重要）**: `voice_memos/` の狙いは「AI に指示する際の手打ちコストを減らす」こと。
文字起こし結果は多くの場合 **ユーザーからの指示・依頼・思考の口述**であり、Claude は
それを「タイプされたプロンプト」と同じものとして受け取り、確認してから動く。

**スコープ**: 対象音声の決定 → 文字起こし実行 → transcript 読込 → 内容の理解確認 → そこから会話/作業の開始。
voice_memos の中身はローカル限定（git 管理外が原則）なので、本 Skill は記録・commit を目的としない。

## 0. 前提（2 層構造）

| 層 | 場所 | 役割 |
|---|---|---|
| プロジェクト層 | カレントディレクトリ直下 `voice_memos/` | `inbox/`（このプロジェクト向けの音声）、`transcripts/`（出力先）、`processed/`、`glossary/`（プロジェクト固有の辞書）、`config.json`（任意の設定上書き） |
| グローバル層 | `~/voice_memos/` | `inbox/`（どのプロジェクトにも属さない音声）、`processed/`、`glossary/`（PC 共通の辞書） |

- 音声は **プロジェクト inbox → グローバル inbox の順に両方**処理される（プロジェクト優先）。transcript は常にプロジェクト層の `transcripts/` に出る
- 辞書はグローバル → `extra_glossary_dirs`（`voice_memos/config.json` で明示追加したプロジェクト共有辞書、任意）→ プロジェクト `voice_memos/glossary/` の順に**マージ**（同じ canonical は後の層＝プロジェクトが優先）
- `--no-global` でグローバル層を無視できる

## 1. 対象音声を決める

- **引数あり**（音声ファイル/フォルダのパス）: それを対象にする。
- **引数なし**: プロジェクト `voice_memos/inbox/` とグローバル `~/voice_memos/inbox/` の音声すべてを対象にする。

実行前に両 inbox の中身を確認（処理対象の把握 + 後で新規 transcript を特定するため）:

```powershell
Get-ChildItem voice_memos\inbox, "$env:USERPROFILE\voice_memos\inbox" -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in '.mp3','.m4a','.wav','.aac','.flac','.ogg','.mp4' } | Select-Object Directory, Name
```

対象が無ければ「`voice_memos/inbox/`（プロジェクト）または `~/voice_memos/inbox/`（共通）に音声を置いてください（または音声パスを指定）」と案内して終了。

## 2. 文字起こし＋校閲を実行

ランチャーはプラグインルートの `transcribe.ps1`。プラグインルートは `${CLAUDE_PLUGIN_ROOT}`、
更新をまたいで保持されるデータ置き場（venv）は `${CLAUDE_PLUGIN_DATA}`。
プロジェクトルートをカレントにして実行する:

```powershell
# inbox の音声を全部処理（既定で glossary 補正 + Claude 文脈校閲つき）
& "${CLAUDE_PLUGIN_ROOT}\transcribe.ps1" -DataDir "${CLAUDE_PLUGIN_DATA}" | Out-String

# 引数で特定ファイルを指定する場合
& "${CLAUDE_PLUGIN_ROOT}\transcribe.ps1" -DataDir "${CLAUDE_PLUGIN_DATA}" "<音声パス>" | Out-String
```

- 出力は `voice_memos/transcripts/<録音日>_<元ファイル名>.md`、音声は見つかった層の `processed/` へ自動退避。
- **注意**: `2>&1 | Select-String` 等で stderr を混ぜると、ライブラリ警告が終了エラー扱いされ途中中断する
  ことがある。**素直にそのまま実行**するか、出力を `| Out-String` で受ける。成否は終了コードと transcript 生成有無で判断する。
- 音声が長いとき（目安 30 分以上）は終わるまで 10 分を超えることがある（長い文字起こしは分けて順に校閲するため）。
  前面の timeout の上限を超えそうなら、バックグラウンドで実行し、終わってから §3 へ進む。
- 文字起こし自体が失敗した場合（GPU 不調等）は `--device cpu` を案内、またはユーザーに状況を伝える。
  GPU の無い PC で常用するなら `voice_memos/config.json` に `{"whisper": {"device": "cpu"}}` を書く（実時間の約 1.4 倍かかる）。
- 上のパス変数が展開されず `${CLAUDE_PLUGIN_ROOT}` のまま見える場合（プラグインではなく個人スキルとして読み込まれている）は、
  スキル読込時に示されたベースディレクトリを使い、`-DataDir` は付けない。

### 初回セットアップ（venv が無いとき）

ランチャーが **exit 2**（`Python venv not found`）を返したら、venv が未構築。ユーザーに次の 2 択を確認してから `setup.ps1` を実行する:

```powershell
# a) 既に WhisperX 入りの venv があるなら、それにリンクする（ダウンロードなし）
& "${CLAUDE_PLUGIN_ROOT}\setup.ps1" -DataDir "${CLAUDE_PLUGIN_DATA}" -LinkTo "<既存 venv のパス>"

# b) 新規構築（Python 3.10・NVIDIA GPU・ffmpeg が前提。約 7GB をダウンロード）
& "${CLAUDE_PLUGIN_ROOT}\setup.ps1" -DataDir "${CLAUDE_PLUGIN_DATA}"
```

`setup OK` が出たら §2 の実行に戻る。

## 3. 生成された transcript を読む

実行後、`voice_memos/transcripts/` で**今回生成された md**（更新時刻が最新のもの／§1 で把握した対象に対応するもの）を Read で読む。複数あれば全部。

各 md の構成:
- frontmatter（`date / source / duration / engine / segments / glossary_replacements / proofread`。校閲に失敗したときだけ、続けて `proofread_error / proofread_error_detail`）
  - `proofread`: 本文が Claude で校閲された文ならモデル名。`false` なら本文は校閲されていない（校閲しない設定、文字起こしが空、または校閲に失敗）。
    長い文字起こし（3,000 字を超えるもの）は分けて校閲してつなぐが、モデル名なら本文の全部が校閲後で、一部でも校閲できなければ `false`
  - `proofread_error`: 校閲を試みたが結果を使わなかった理由。`timeout`（時間切れ）/ `cli_error`（Claude CLI のエラー）/ `empty`（空の結果）/ `too_short`（短すぎる＝要約の疑い）/ `incomplete`（一部が欠けた疑い）/ `error`（その他）。`proofread_error_detail` は数などの補足 1 行。
    分けて校閲したときは、何番目のまとまりで止まったかが前に付く（例: `まとまり 3/5: num_turns 3`）
- 本文: `proofread` がモデル名なら校閲後。`false` なら校閲前の文字起こし（辞書の置き換え後）
- `<details>校閲前（辞書の置き換え後）</details>`: 校閲に渡した文。本文が校閲後のときだけ付く（`false` のときは本文と同じなので付かない）
- `<details>置き換え前（Whisper の出力）</details>`: 辞書の置き換え（wrong_variants → canonical）の前の文。置き換えが 1 箇所以上あったときだけ付く。
  `glossary_replacements: 0` なら置き換え前の文は、`proofread: false` のときは本文と、校閲されたときは校閲前の節と同じ
- `<details>` は必要なときだけ参照する。原則は本文を使う

`proofread_error` があるときは、校閲に失敗したことと理由を利用者に 1 行で伝え（例:「校閲が時間切れになったので、校閲前の文字起こしで読みます」）、
本文を校閲されていない文として読む（誤変換・固有名詞・数値をより慎重に扱う）。

## 4. 内容を「ユーザーからの入力」として解釈し、会話を始める（核心）

文字起こし本文を、**ユーザーが口述で渡した指示・思考・メモ**として受け取る。まず取り違えを防ぐため、
**要点を 1〜3 行に圧縮して理解を提示**（＝確認を兼ねる）。そのうえで内容の性質に応じて動く:

| 内容の性質 | Claude の動き |
|---|---|
| **指示・依頼**（「〜を作って」「〜を調べて」「〜を直して」） | タイプされたプロンプトと同様に**着手**する。範囲が大きい/曖昧なら 1 点だけ確認してから進む |
| **思考・アイデアの口述** | 壁打ち相手として整理・構造化・深掘り。論点や抜けを返す |
| **判断が要る話** | 選択肢と論点を並べ、**最終判断はユーザーに委ねる**（判断ごと巻き取らない） |
| **タスクの羅列** | やることリスト化し、優先度・依存・所要を添えて提示 |
| **感情・疲労・近況の吐露** | 受け止める。タスク化を急がない。疲労が見えるなら休息を促す（作業代行は申し出てよいが判断は奪わない） |

複数メモがあるときは 1 件ずつ順に扱う。メモ同士に関連があればまとめて文脈を作る。

## 5. 文字起こし品質への注意

- 音声認識・校閲は完璧ではない。**固有名詞・数値・聞き取りにくい箇所**は誤りうる。意味が通らない箇所や
  重要語が怪しいときは、断定せず「ここはこう言った？」と確認する（誤解で進めない）。判断に効く所ほど慎重に。
- 固有名詞が繰り返し誤変換される場合は、辞書に `wrong_variants → canonical` の追記を**提案**する
  （canonical の確定はユーザーの領分）。置き場所は、そのプロジェクト固有の語なら `voice_memos/glossary/*.json`、
  人名など PC 全体で使う語なら `~/voice_memos/glossary/*.json`。
  wrong_variants は実際に観測された誤認識のみ追加する（仮想的な誤りは過剰置換のリスク）。
- どうしても聞き取れない箇所は `<details>` の校閲前の文も見て突き合わせる。それでも不明ならユーザーに確認。
- 辞書の置き換えの誤り（正しい語まで canonical に置き換えた等）を疑うときは、「置き換え前」の節（置き換えがあったときだけ付く）と突き合わせる。
  誤りなら、その wrong_variants を辞書から外すことを**提案**する。

## 失敗時のフォールバック

- inbox 空 & 引数なし → 音声の配置/パス指定を案内して終了。
- venv が見つからない（exit 2） → §2「初回セットアップ」の手順で `setup.ps1` を実行。
- 校閲（Claude CLI）だけ失敗 → ツールは校閲前の文字起こし（辞書の置き換え後）を本文にして保存する（`proofread: false`、理由は `proofread_error`）。§3 のとおり利用者に 1 行伝えて、本文で続行。
- 文字起こし内容が空/ノイズのみ → その旨を伝え、録り直しを提案。
