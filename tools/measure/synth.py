"""合成のデータ (辞書・メモの文・音声合成に読ませる文) を、決まった乱数の種で作る。

public repo なので、測りには利用者の辞書の語・メモの中身を使わない (#3 §4)。ここで作る語と文は
音節や字を乱数で組み合わせた架空のもので、同じ種なら同じ結果になる。

    python tools/measure/synth.py dict --profile M --count 78 --out <dir>/synthetic.json
    python tools/measure/synth.py memo --chars 15000 --out <dir>/memo.txt
    python tools/measure/synth.py tts --chars 400 --out <dir>/tts.txt

import しただけでは何もしない (標準ライブラリだけを使う)。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# --- 語の部品 -------------------------------------------------------------

_KATA = list(
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワ"
    "ガギグゲゴザジズゼゾダデドバビブベボパピプペポ"
)
_KATA_TAIL = ["ン", "ー", "ル", "ス", "ト", "ア"]
_KANJI = list(
    "青森川山田中村本石松林高橋野原井上木島岡西東北南新古和光明星月花風雲海空道町"
    "丘谷浜沢池泉坂城宮寺湊里桜梅竹菊鶴亀鷹鳥"
)
_KANJI_SUFFIX = ["研究会", "推進室", "工房", "計画", "通信", "製作所", "商会", "倶楽部", "企画", "文庫"]
_ASCII_C = list("bcdfghjklmnprstvxz")
_ASCII_V = list("aeiou")
_JA_SUFFIX = ["クラウド", "ノート", "ボード", "ラボ", "ポータル", "シリーズ", "キット"]

PROFILES = ("S", "M", "L", "E")
PROFILE_LEN = {"S": (3, 5), "M": (6, 9), "L": (10, 16), "E": (6, 10)}


def _kata_word(rng: random.Random, lo: int, hi: int) -> str:
    n = rng.randint(lo, hi)
    s = "".join(rng.choice(_KATA) for _ in range(n - 1))
    return s + rng.choice(_KATA_TAIL)


def _ascii_word(rng: random.Random, lo: int, hi: int) -> str:
    n = rng.randint(lo, hi)
    out = []
    for i in range(n):
        out.append(rng.choice(_ASCII_C if i % 2 == 0 else _ASCII_V))
    return "".join(out).capitalize()


def _make_word(profile: str, rng: random.Random) -> str:
    if profile == "S":
        return _kata_word(rng, 3, 5)
    if profile == "M":
        k = "".join(rng.choice(_KANJI) for _ in range(rng.randint(2, 4)))
        mid = rng.choice(["", "の", ""])
        return k + mid + rng.choice(_KANJI_SUFFIX)
    if profile == "L":
        return (_kata_word(rng, 3, 5) + _ascii_word(rng, 2, 4).upper()
                + str(rng.randint(1, 99)) + rng.choice(_JA_SUFFIX))
    if profile == "E":
        w = _ascii_word(rng, 4, 7)
        return w + (str(rng.randint(2, 99)) if rng.random() < 0.5 else "")
    raise ValueError(f"unknown profile: {profile}")


def synth_canonicals(profile: str, count: int, seed: int = 0) -> list[str]:
    """profile (S/M/L/E) の架空の語を count 個、重複なしで返す。字数は PROFILE_LEN の範囲。"""
    if profile not in PROFILE_LEN:
        raise ValueError(f"unknown profile: {profile}")
    lo, hi = PROFILE_LEN[profile]
    rng = random.Random(f"{profile}:{seed}")
    out: list[str] = []
    seen: set[str] = set()
    guard = 0
    while len(out) < count:
        guard += 1
        if guard > count * 1000 + 1000:
            raise RuntimeError("語を作りきれない (部品が足りない)")
        w = _make_word(profile, rng)
        if not (lo <= len(w) <= hi) or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def wrong_variant(word: str, seed: int = 0) -> str:
    """誤認識の代わりに、1 字だけ別の字に替えた形を返す (校閲で直す対象にする)。"""
    rng = random.Random(f"wrong:{word}:{seed}")
    chars = list(word)
    idx = [i for i, c in enumerate(chars) if c in _KATA or c in _KANJI]
    if not idx:
        return word + "ー"
    i = rng.choice(idx)
    pool = _KATA if chars[i] in _KATA else _KANJI
    chars[i] = rng.choice([c for c in pool if c != chars[i]])
    return "".join(chars)


def synth_entries(profile: str, count: int, seed: int = 0) -> list[dict]:
    """キットの辞書と同じ形の entry を返す (canonical と、合成の wrong_variants 1 つ)。"""
    return [
        {"canonical": c, "wrong_variants": [wrong_variant(c, seed)], "notes": "synthetic", "source": "synthetic"}
        for c in synth_canonicals(profile, count, seed)
    ]


def mixed_entries(count: int, seed: int = 0) -> list[dict]:
    """S/M/L/E を順に混ぜた entry (実の辞書に近い混ざり方の代わり)。"""
    per = -(-count // len(PROFILES))
    pools = {p: synth_entries(p, per, seed) for p in PROFILES}
    out: list[dict] = []
    i = 0
    while len(out) < count:
        p = PROFILES[i % len(PROFILES)]
        out.append(pools[p][i // len(PROFILES)])
        i += 1
    return out


# --- メモ風の文 -----------------------------------------------------------

_FILLERS = ["えーと、", "あの、", "まあ、", "なんていうか、", "そうですね、", "で、", "うーん、", "えっと、"]
_TOPICS = [
    "棚の組み立て", "週末の買い出し", "画面の配置", "発表の資料", "庭の水やり", "引っ越しの段取り",
    "読書会の準備", "新しい献立", "記録の整理", "見積もりの比較", "旅行の計画", "部屋の片付け",
    "練習の予定", "通知の文言", "控えの手順", "名簿の見直し", "掲示の作り直し", "道具の手入れ",
]
_DAYS = ["月曜", "火曜", "水曜", "木曜", "金曜", "週末", "月末", "来月の頭"]
_TEMPLATES = [
    "今日は{t}について考えたことをメモしておきます。",
    "{n}の件は、来週の{d}までに一度まとめたいと思っています。",
    "{w}のほうで{k}件くらい問い合わせが来ていたので、あとで確認します。",
    "正直、{t}はもう少し時間がかかると思っていたんですけど、以外と早く終わりそうです。",
    "ただ、{n}との兼ね合いがあるので、順番は入れ替えるかもしれません。",
    "{k}個ある候補のうち、上から三つくらいに絞って比べてみるつもりです。",
    "前回は{t}で手間取ったので、今回は先に下書きを作っておきます。",
    "{n}の担当の人には、{d}の午後に連絡してみます。",
    "そもそも{t}って誰のためにやっているのか、一回整理したほうがいい気がします。",
    "細かいところは後で直すとして、まずは全体の流れを決めたいです。",
    "{w}の資料、どこに置いたか忘れちゃったので探さないといけないです。",
    "{k}時くらいから始めれば、夕方には一区切りつくと思います。",
    "あと、{t}のほうは急ぎじゃないので、余裕があるときでいいです。",
    "昨日話した{n}の案は、やっぱり少し大きすぎるかなという気がしています。",
    "費用の面で言うと、{k}千円くらいに収まれば十分だと思います。",
    "それで、{t}の前に一度だけ確認の時間を取りたいです。",
    "{n}と{n2}のどちらを先にやるか、まだ決めきれていません。",
    "思いつきなんですけど、{t}を二回に分けるのもありかもしれません。",
    "この前の懐疑で出た話は、反対の意見も多かったので、もう少し様子を見ます。",
    "{w}って名前、たまに間違えて呼んじゃうので気をつけます。",
    "とりあえず{d}は{t}に集中して、ほかのことは後回しにします。",
    "{k}ページくらいの資料になりそうなので、読む人の負担も考えないといけないです。",
    "やることを書き出してみたら、思ったより多くて少し焦っています。",
    "{n}のほうから返事が来たら、その内容を見て次を決めます。",
    "{t}は去年もやったので、そのときのメモを見返してみます。",
    "気になっているのは、{n}の締め切りが{d}だったかどうかです。",
    "もし間に合わなかったら、{k}日くらい後ろにずらしてもらえるか聞いてみます。",
    "手順としては、まず{t}、その次に確認、最後に共有という流れです。",
    "{w}の画面、文字が小さくて読みにくいって言われたので直したいです。",
    "うまくいくかはわからないですけど、試してみる価値はあると思います。",
    "{n}に関しては、前に決めたルールをそのまま使えばいいはずです。",
    "{k}人くらい集まる予定なので、場所は少し広めのところがいいです。",
    "{t}の途中で気づいたことがあったら、またここに録音しておきます。",
    "これは以外と大事なことなので、忘れないように言っておきます。",
    "{n}の説明は、もう少し短くまとめたほうが伝わりやすいと思います。",
    "{d}の朝に一度見直して、問題なければそのまま出します。",
    "{t}と{t2}を同じ日にやるのは、さすがに無理があるかもしれません。",
    "{w}から届いた案内に、{k}番の項目が抜けていたのが気になりました。",
    "まだ考えがまとまっていないので、この話はいったん置いておきます。",
    "次に録るときは、{n}の結果がどうなったかを話そうと思います。",
    "{t}について、良かった点と悪かった点を三つずつ挙げてみます。",
    "{n}の件、聞いた話だと{k}割くらいは終わっているそうです。",
    "そういえば、{t}に必要な道具がまだ届いていないです。",
    "{d}は予定が詰まっているので、{t}は別の日にしたほうがよさそうです。",
    "{n}のやり方は、前の{n2}のときと同じでいいと思います。",
    "{t}の話に戻ると、いちばん困っているのは時間の見積もりです。",
    "{n}の人たちにも、{t}のことは一言伝えておいたほうがいいですね。",
    "{k}回目の打ち合わせで、ようやく方向が見えてきた感じがします。",
]


def memo_names(seed: int = 0) -> list[str]:
    """メモの文に出す合成の固有名詞 (校閲の canonical 一覧にも渡す)。"""
    return synth_canonicals("M", 12, seed) + synth_canonicals("S", 8, seed)


def synth_memo_text(target_chars: int, seed: int = 0, names: list[str] | None = None) -> str:
    """ボイスメモの文字起こしに似せた合成の本文。段落は空行区切り (transcribe.format_body と同じ)。

    段落の切れ目で止めるので、字数は target_chars を少し超える。
    """
    rng = random.Random(f"memo:{seed}")
    names = names or memo_names(seed)
    paragraphs: list[str] = []
    total = 0
    while total < target_chars:
        sents = []
        for _ in range(rng.randint(3, 7)):
            tpl = rng.choice(_TEMPLATES)
            n, n2 = rng.sample(names, 2)
            t, t2 = rng.sample(_TOPICS, 2)
            s = tpl.format(n=n, n2=n2, w=wrong_variant(n, seed), t=t, t2=t2,
                           d=rng.choice(_DAYS), k=rng.randint(2, 30))
            if rng.random() < 0.45:
                s = rng.choice(_FILLERS) + s
            sents.append(s)
        p = "".join(sents)
        paragraphs.append(p)
        total += len(p) + (2 if len(paragraphs) > 1 else 0)
    return "\n\n".join(paragraphs)


def synth_tts_text(target_chars: int, seed: int = 0) -> str:
    """音声合成に読ませる、中身の詰まった合成の文 (カタカナの語と数が多い。空行なし)。"""
    rng = random.Random(f"tts:{seed}")
    words = synth_canonicals("S", 40, seed) + synth_canonicals("L", 20, seed)
    out: list[str] = []
    total = 0
    while total < target_chars:
        a, b, c = rng.sample(words, 3)
        s = rng.choice([
            f"{a}と{b}と{c}を、{rng.randint(2, 9)}回ずつ確かめます。",
            f"{a}の次は{b}、その次は{c}です。",
            f"{a}、{b}、{c}の三つを並べて比べます。",
            f"{rng.randint(10, 99)}番の{a}は、{b}に入れ替えます。",
        ])
        out.append(s)
        total += len(s)
    return "".join(out)


def glossary_json(entries: list[dict], title: str = "synthetic") -> str:
    """キットの辞書 JSON の形 (README の「辞書 JSON の書式」) にする。"""
    clean = [{k: v for k, v in e.items() if k in ("canonical", "wrong_variants", "notes")} for e in entries]
    return json.dumps({"title": title, "entries": clean}, ensure_ascii=False, indent=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="合成のデータを作る (#4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dict", help="合成の辞書 JSON")
    d.add_argument("--profile", choices=list(PROFILES) + ["mixed"], default="mixed")
    d.add_argument("--count", type=int, required=True)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("memo", help="合成のメモの本文")
    m.add_argument("--chars", type=int, required=True)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("tts", help="音声合成に読ませる文")
    t.add_argument("--chars", type=int, required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "dict":
        entries = (mixed_entries(args.count, args.seed) if args.profile == "mixed"
                   else synth_entries(args.profile, args.count, args.seed))
        text = glossary_json(entries, title=f"synthetic-{args.profile}-{args.count}")
    elif args.cmd == "memo":
        text = synth_memo_text(args.chars, args.seed)
    else:
        text = synth_tts_text(args.chars, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Windows でも LF で書く (字数の数え方を、作った文と揃えるため)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
