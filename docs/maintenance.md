# 保守の手引き（voice-memo-kit）

## この文書について

- 読む人: キットを保守する人（人でも AI でも）。キットを変える前、キットを更新した後、長いメモを録った後に読む。
- 出どころ: プロジェクト [#3](https://github.com/howel-jp/voice-memo-kit/issues/3)（長い辞書・長いメモでも、固有名詞の予告と校閲が黙って抜け落ちない）の申し送り（[#12](https://github.com/howel-jp/voice-memo-kit/issues/12)。0.4.0〜0.6.1）。
- Issue・PR の進め方は手順の正本に従う。この文書は、このキットに固有の「手で確かめること」「見張っていること」「測りの道具」だけを書く。
- この repo は public。ここにも Issue にも、利用者の辞書の語・メモの中身・私的なパスを書かない（数と合成の語だけ）。

## 手動で確かめること（次のフェーズへの申し送り）

ユニットテスト（`python -m unittest discover -s tests`）は、WhisperX・トークナイザ・Claude CLI をスタブにして形を固定している。
実際の録音・実際の CLI でしか分からないことを、機会があったときに手で確かめる。

| 何を確かめるか | どこで見るか | いつ | 書く先 |
|---|---|---|---|
| 45 分級の実際のメモが校閲されるか。置き換えがあれば、置き換え前の語が残るか | 出力 md の frontmatter の `proofread` がモデル名か。`<details>` の「置き換え前（Whisper の出力）」の節（置き換えが 1 箇所以上のとき） | 次に 45 分級のメモを録ったとき（持ち主） | #3 にコメント（数と「残った / 残らなかった」だけ） |
| 長いメモを `/voice-memo` で走らせたとき、Claude が前面の timeout で止められずに md まで進むか | SKILL.md §2 の `run_in_background` の手順どおりに進んだか。`voice_memos/transcripts/` に md ができたか | 同上 | 同上 |
| 実のメモで `proofread_error: incomplete` が出たとき、誤検知か本当の欠けか | `proofread_error_detail`（段落の数・`num_turns`・末尾）と、校閲前の文の突き合わせ | 出たとき | この repo の Issue（無ければ起こす）。detail の値と、誤検知か欠けかだけ |
| `[WARN] initial_prompt` が出たときに Claude が 1 行で伝えるか。`"priority": true` を付けて誤変換が減るか | ランチャーの出力と Claude の返事。印を付ける前後の誤変換の回数 | 辞書が 223 トークンを超えたとき | 同上（印の効き目は測っていない） |
| キットを更新した後に、meeting-transcript・clip-shorts が動くか | 2 つのスキルを 1 回ずつ動かす（どちらもキットの 7 関数を import する） | キットの version を上げた後 | 同上（`tests/test_kit_api.py` が確かめるのは呼び方と型だけ） |
| README の校閲の時間の目安が、実のメモにかかった時間と合うか | ランチャーの `[C] まとまり k/n 完了（n 秒）` の行と字数 | 長いメモを録ったとき | 同上（合わなければ、下の「測りの道具」で測り直して README を直す） |

これまでの手元の数（どれも数だけ）: 0.4.0 の段落のガードと 0.5.0 の末尾の確かめで、手元の transcript 13 件のうち誤って捨てるものは 0 件だった（#8 の判断 1、#2 の判断 1）。ただし手元のメモは短く（3,000 字を超えるのは 2 件）、長いメモでの率は分かっていない。

## 見張っていること（保留）

今は直さないが、条件が来たら取りかかるもの。条件が来たら、この repo に Issue を起こし（1 件ずつ）、ここから消す。

### Whisper の 1 チャンクの出力の上限（224 トークン）

- 何か: faster-whisper（ctranslate2）は、1 チャンク（VAD で 30 秒までに分けた区切り）に最大 224 トークン（initial_prompt を使い切ると 221）しか出さず、
  超える分は黙って切れる。prompt によらない。合成の密な音声では、出力の字数が元の 0.74〜0.75 になった。手元の実際のメモは
  1 チャンクの中央値 72 トークン（30 秒あたり最大 113）で、上限に約 2〜3 倍の余裕がある（[#4 の U3](https://github.com/howel-jp/voice-memo-kit/issues/4#issuecomment-6380600828)）。
- 再開の条件（[#3 の現在地 (2)](https://github.com/howel-jp/voice-memo-kit/issues/3#issuecomment-6380635953) の原文）:
  「実のメモの文字起こしで 1 チャンクの出力が 200 トークンを超えたことが分かったとき (T2 でチャンクごとの数をログに出せるようになったら、そこで見る)、
  または持ち主がチャンクの末尾の欠けに気づいたとき」
- 見え方: 0.6.0 から、文字起こしのたびに `[A] チャンクの出力（数え直し）: 最大 N トークン / M チャンク（1 チャンクの上限 224）` の行が出て、
  200 を超えたチャンクがあると `[WARN] 1 チャンクの出力が 200 トークンを超えました…` が出る。0.6.1 から、SKILL.md §5 で Claude がこの `[WARN]` を利用者に 1 行で伝える。
- 見つけたら: Issue に、`[WARN]` の行の数（チャンクの数・最大のトークン数）と、末尾が切れていたかだけを書く（メモの中身は書かない）。

### WhisperX の実行のたびに出る venv の警告

- 何か: WhisperX を動かすたびに、pyannote の `torchcodec is not installed correctly` の UserWarning と、Lightning の checkpoint の更新のメッセージが
  stderr に出る。前からある警告で、文字起こしの結果には効いていない（#1 の C1 の 2 回で、出力は期待どおり）。SKILL.md §2 は、stderr の警告を混ぜて扱わないように書いている。
- 再開の条件（[#3 の振り分けのコメント](https://github.com/howel-jp/voice-memo-kit/issues/3#issuecomment-6383515845) の原文）:
  「依存の更新 (requirements.lock.txt) をするとき、または警告がエラーになって文字起こしが止まったとき」
- 見え方: ランチャーの stderr。
- 見つけたら: 依存の更新の Issue で一緒に扱う（`requirements.lock.txt` は、手順の正本の許容範囲の外（依存パッケージ）に当たる）。
  警告がエラーになって止まったなら、その Issue を起こす。

## 測りの道具（`tools/measure/`）

README の目安の数を出した道具。実行時（`/voice-memo`）には使わない。import しただけでは、重い import・プロセスの起動・外部への接続・CLI の呼び出しをしない
（`tests/test_measure.py` の `ImportSideEffectTest`）。

| 道具 | 測るもの | 使った所 |
|---|---|---|
| `synth.py` | 合成の辞書（語の長さ S/M/L/E と、混ぜたもの。`--priority-every N` で印）・メモ風の文・音声合成に読ませる文。同じ種なら同じ結果 | 下のすべての材料 |
| `make_tts.ps1` | 合成の文から Windows の音声合成で 16 kHz の wav を作る（パスは絶対パスで渡す） | #4 の U3、各 C8 |
| `prompt_tokens.py` | initial_prompt のトークン数と、後ろ 223 トークンに届く語の数（faster-whisper の `get_prompt` でも確かめる）。`real`・`dict` は数だけを出す | #4 の U1・U2、#1 の C1 |
| `prompt_budget.py` | 1 チャンクの出力の上限（ctranslate2 の `generate` を直に呼ぶ）と、キットの経路での影響 | #4 の U3 |
| `transcript_stats.py` | 手元の transcript の字数/分・トークン/30 秒（数だけ） | #4 の U4（45 分 ≒ 11,500〜12,600 字） |
| `proofread_timing.py` | 校閲の時間・CLI の呼び出しごとの記録・段落の欠け（`--call-claude` のときだけ CLI を呼ぶ） | #4 の U4、#2 の C4 |
| `timeout_wait.py` | 時間切れの後に `subprocess.run` が返るまでの待ちと、子孫のプロセス（`claude --call-claude` のときだけ CLI を呼ぶ） | #4 の U5 |

測り直すときの前提:
- 合成の文・合成の辞書・合成の音声だけを使う。`real`・`transcript_stats.py` で手元の辞書・transcript を数えるときも、出すのは数だけ。
- 道具には作業ツリーを渡さず、`git archive <commit> | tar -x -C <空のディレクトリ>` の写しで走らせる（測ったコードとコミットが一致する）。cwd は `CLAUDE.md` の無い場所。
- トークナイザ・faster-whisper・WhisperX を使うコマンドには、`HF_HUB_OFFLINE=1` と `PYANNOTE_METRICS_ENABLED=false` を付ける（ダウンロードしない。pyannote の利用の数を送らない。#6）。
- Claude CLI を呼ぶのは `--call-claude` を付けたときだけ（定額の枠を使う）。呼んだ回数を記録する。
- 結果には、測った日・コミット・CLI の版・モデルを添える。README の目安（利用者に見える値）を変えるときは、条件を変えて測り直してから（別の run・別の日・利用者と同じ起動）。

よく使うコマンド（`<K>` は写しのディレクトリ、`<PY>` は venv の python）:

```powershell
# 届く語の数（#4 の U1。トークナイザだけ、CLI 0 回）
<PY> -I <K>/tools/measure/prompt_tokens.py synthetic                      # 印なし
<PY> -I <K>/tools/measure/prompt_tokens.py synthetic --priority-every 4   # 印あり
<PY> -I <K>/tools/measure/prompt_tokens.py dict <辞書のディレクトリ>        # 辞書 1 つの数（数だけ）

# 校閲の時間（#2 の C4。12,638 字で CLI 5 回の見込み、やり直しを含めて最大 15 回。約 7 分）
<PY> -I -u <K>/tools/measure/proofread_timing.py --call-claude --sizes 12600 --seed 0 --timeout 900 --out-dir <dir> --label <label>

# 時間切れの後の待ち（#4 の U5 の (a)。代わりの子だけ、CLI 0 回）
<PY> -I -u <K>/tools/measure/timeout_wait.py stand-in --timeout 3 --sleep 20
```

README の目安の数と、その出どころ:

| README の数 | 出どころ |
|---|---|
| initial_prompt の後ろ 223 トークン | faster-whisper 1.2.1 の `get_prompt`（`max_length // 2 - 1`）。#4 の U1 で `get_prompt` を通して確かめた |
| 前置きの文 24 トークン、語の長さで 18〜41 語 | `prompt_tokens.py synthetic`（#4 の U1） |
| 1 チャンクの上限 224（使い切ると 221） | `prompt_budget.py mechanism`（#4 の U3。GPU と CPU の 46 行） |
| 3,000 字で分けて校閲する | #4 の U4・U6（1 回で返った出力トークンが字数の 3.5〜4.2 倍。読みの上限 32,000 に余裕を残す大きさ） |
| 校閲 1,000 字あたり約 25〜45 秒 | `proofread_timing.py`（#4 の U4、#2 の C4）と、入った版のランチャー（#2 の C8）。2026-10-24 |
