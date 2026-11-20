"""Validate generated synthetic conversations by replaying them in the TRAIN simulator.

    python3 -I synth/conv/validate.py synth/conv/out/batch_01.jsonl
Every item is replayed in a fresh ToolEnv of its card's environment. Prints one PROBLEM line per rejected item (fix and
re-run) and a summary ending in "VALIDATION: <accepted>/<total> accepted". Accepted items, with the simulator's real
results, are written to synth/conv/accepted/<same file name>.
"""
from __future__ import annotations

import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from lifemate.config import artifacts_dir, data_root, load_config  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples, tool_index, zaic_tools  # noqa: E402
from lifemate.data.splits import load_splits  # noqa: E402

ASK = "sys.ask_missing_info"
MAX_CALLS = 8
VI = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
DT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?[+-]\d{2}:\d{2}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_LIKE = re.compile(r"^[a-z][a-z0-9]*_[a-z0-9_]*\d[a-z0-9_]*$")
NOT_FOUND = ("không tìm thấy", "khong tim thay", "not found")


def norm(t: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", t or "").lower()))


NEAR_COPY = 0.5     # bigram Jaccard vs already accepted synthetic requests (template families sit at 0.5-0.9)
NEAR_REAL = 0.6     # vs real training requests
COND = re.compile(r"\bnếu\b|\bmà\b.{0,30}\bthì\b|\bkhông thì\b|\bthì thôi\b", re.I)
REASON = re.compile(r"\b(vì|nên|sợ|kẻo|cho đỡ|cho tiện|đỡ phải|bận quá|bận lắm)\b|\bđang (đi|bận|làm|nấu|lái)\b", re.I)


def bigrams(t: str) -> set:
    w = norm(t).split()
    return set(zip(w, w[1:]))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def load_cards() -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (HERE / "cards").glob("batch_*/*.json")}


ISO_WORD = re.compile(r"ISO[- ]?8601", re.I)
GOLD_FORMAT: dict[tuple[str, str], str] = {}   # (tool, param) -> "datetime" / "date" when every gold value has that form


READ_STYLE: dict[str, tuple] = {}   # read tool -> argument-key pattern used in >= 80% of gold calls (soft warning)

# Conventions measured on the 1,545 real train conversations (audit of 5 Nov):
SRC = re.compile(r"lấy từ ([a-z_]+\.[a-z_]+)")                       # "ID ..., lấy từ contacts.search" in a parameter description
NOTE_WORDS = re.compile(r"\bghi\b(?! (chú|thêm vào|vào))|nội dung|lời nhắn|\bnote\b", re.I)   # banking note only when dictated: 82/82
LABEL_WORDS = re.compile(r"\btên\b|nhãn|label", re.I)                 # alarm label: 1/78 gold alarms have one
UNDECIDED = re.compile(r"chưa (chốt|quyết|chọn|biết|nghĩ|tính|thống nhất)|để .{0,12}(sau|tính|nghĩ|xem)|nói sau|tính sau", re.I)
PHONE_NO = re.compile(r"\d{9,}")


def convention_problems(req: str, c: dict, profile: dict) -> list[str]:
    n, g = c["name"], c["arguments"]
    out = []
    if n in ("banking.transfer", "ewallet.transfer") and "note" in g and not NOTE_WORDS.search(req):
        out.append(f"{n}: remove 'note': annotators add a note only when the user dictates one (\"ghi ...\", \"nội dung ...\")")
    if n == "food_delivery.place_order" and "address" in g and g["address"].strip() == (profile.get("home_address") or "").strip():
        out.append(f"{n}: remove 'address': for delivery to the user's home annotators leave it out (27/28)")
    if n == "alarm.create" and "label" in g and not LABEL_WORDS.search(req):
        out.append(f"{n}: remove 'label': annotators add one only when the user names the alarm (\"tên ...\", \"nhãn ...\")")
    return out


def unresolved_before_ask(req: str, ask_args: dict, before: set, hist_calls: set, installed: set) -> list[tuple]:
    """Annotators first make the read that returns each needed ID, then ask (163/167 real asks)."""
    spec = INDEX[ask_args["tool_name"]]["parameters"]
    req_p = set(spec.get("required", []))
    one = {k for grp in spec.get("x-one-of", []) if len(grp) == 1 for k in grp}
    out = []
    for k, p in spec.get("properties", {}).items():
        m = SRC.search(p.get("description") or "")
        if not m:
            continue
        needed = k in ask_args["params"] or k in req_p or (k in one and not PHONE_NO.search(req))
        if needed and m.group(1) in installed and m.group(1) not in before and m.group(1) not in hist_calls:
            out.append((k, m.group(1)))
    return out


def _named(x, out: dict):
    if isinstance(x, dict):
        nm = x.get("name") or x.get("title") or x.get("label")
        if isinstance(x.get("id"), str) and isinstance(nm, str):
            out[x["id"]] = nm
        for v in x.values():
            _named(v, out)
    elif isinstance(x, list):
        for v in x:
            _named(v, out)


def wrong_entity(req: str, calls: list, results: list, history_results) -> list[str]:
    """A write uses ID X whose name is absent from the request while a sibling entity (same ID prefix, not used in this
    item) is named there: almost always a mix-up (on real annotations this only fires on self-corrections)."""
    known: dict = {}
    _named(history_results, known)
    _named(results, known)
    r = " " + norm(req) + " "
    used = set()
    for c in calls:
        for v in c["arguments"].values():
            for y in (v if isinstance(v, list) else [v]):
                used.add(y.get("item_id") if isinstance(y, dict) else y)
    pre = lambda i: re.sub(r"\d+$", "", i)  # noqa: E731
    out = []
    for c in calls:
        if c["name"] == ASK or INDEX[c["name"]].get("kind") != "write":
            continue
        for k, v in c["arguments"].items():
            for x in (v if isinstance(v, list) else [v]):
                x = x.get("item_id") if isinstance(x, dict) else x
                if not isinstance(x, str) or x not in known or f" {norm(known[x])} " in r:
                    continue
                sib = [(i, nm) for i, nm in known.items() if i not in used and pre(i) == pre(x)
                       and len(norm(nm).split()) >= 2 and f" {norm(nm)} " in r]
                if sib:
                    out.append(f"{c['name']}.{k}={x} is {known[x]!r}, but the request names {sib[0][1]!r} ({sib[0][0]})")
    return out


def learn_gold_formats(samples: list[dict]) -> None:
    seen = collections.defaultdict(list)
    keys = collections.defaultdict(collections.Counter)
    for s in samples:
        for c in s["label"]["tool_calls"]:          # the scored final turns define the read style
            keys[c["name"]][tuple(sorted(c.get("arguments") or {}))] += 1
        calls = list(s["label"]["tool_calls"]) + [c for t in s["history"] for c in (t.get("tool_calls") or [])]
        for c in calls:
            for k, v in (c.get("arguments") or {}).items():
                if isinstance(v, str):
                    seen[(c["name"], k)].append(v)
    for name, cnt in keys.items():
        pat, k = cnt.most_common(1)[0]
        if sum(cnt.values()) >= 10 and k / sum(cnt.values()) >= 0.8:
            READ_STYLE[name] = pat
    for key, vals in seen.items():
        if len(vals) >= 3 and all(DT.match(v) for v in vals):
            GOLD_FORMAT[key] = "datetime"
        elif len(vals) >= 3 and all(DATE.match(v) for v in vals):
            GOLD_FORMAT[key] = "date"


def arg_problems(name: str, args: dict, spec: dict) -> list[str]:
    out = []
    props = spec["parameters"].get("properties", {})
    for k, v in args.items():
        p = props.get(k)
        if p is None:
            out.append(f"{name}: unknown parameter {k!r}"); continue
        t = p.get("type")
        if t == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
            out.append(f"{name}.{k}: must be an integer, got {v!r}")
        if t == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
            out.append(f"{name}.{k}: must be a number, got {v!r}")
        if "enum" in p and v not in p["enum"]:
            out.append(f"{name}.{k}: {v!r} not in enum {p['enum']}")
        desc = (p.get("description") or "") + " " + (p.get("format") or "")
        if isinstance(v, str):
            fmt = GOLD_FORMAT.get((name, k))
            if fmt == "datetime" or "date-time" in desc or ISO_WORD.search(desc):
                if not DT.match(v):
                    out.append(f"{name}.{k}: {v!r} must be ISO 8601 with offset, e.g. 2026-11-29T09:00:00+07:00")
            elif fmt == "date" or p.get("format") == "date" or "YYYY-MM-DD" in desc:
                if not DATE.match(v):
                    out.append(f"{name}.{k}: {v!r} must be YYYY-MM-DD")
    for r in spec["parameters"].get("required", []):
        if r not in args:
            out.append(f"{name}: missing required parameter {r!r}")
    return out


INDEX: dict = {}


def main():
    cfg = load_config(ROOT / "configs/v1.yaml")
    root, art = data_root(cfg), artifacts_dir(cfg)
    sp = load_splits(art / "splits" / f"{cfg['splits']['name']}.json")
    train_ids = set(sp["train"])
    zt = zaic_tools(root, "train")
    index = tool_index(load_catalog(root, "train"))
    INDEX.update(index)
    learn_gold_formats(load_samples(root, "train"))
    real_requests = {norm(s["audio_text"]) for s in load_samples(root, "train")}
    real_requests |= {norm(t["audio_text"]) for s in load_samples(root, "train") for t in s["history"] if t["role"] == "user"}
    cards = load_cards()
    accepted_dir = HERE / "accepted"
    accepted_dir.mkdir(exist_ok=True)
    seen, seen_bg = set(), []          # requests already accepted in OTHER files (exact, and bigram sets for near-copies)
    for f in accepted_dir.glob("*.jsonl"):
        if Path(f).name == Path(sys.argv[1]).name:
            continue
        for l in open(f, encoding="utf-8"):
            try:
                r0 = json.loads(l)["request"]
            except (json.JSONDecodeError, KeyError):
                continue           # another validator may be writing this file right now
            seen.add(norm(r0))
            seen_bg.append((bigrams(r0), r0))
    real_bg = [(bigrams(s["audio_text"]), s["audio_text"]) for s in load_samples(root, "train")]
    total, ok_items, stats = 0, [], collections.Counter()
    tools_used = collections.Counter()
    for n, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
        if not line.strip():
            continue
        total += 1
        probs = []
        try:
            it = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"PROBLEM line {n}: not JSON ({e})"); continue
        for k in ("card", "request", "calls", "outcome"):
            if k not in it:
                probs.append(f"missing key {k!r}")
        card = cards.get(it.get("card", ""))
        if card is None:
            probs.append(f"unknown card {it.get('card')!r}")
        if probs:
            print(f"PROBLEM line {n}: " + "; ".join(probs)); continue
        if card["env_sample"] not in train_ids:
            print(f"PROBLEM line {n}: card environment is not in the train partition"); continue
        req, calls, outcome = it["request"], it["calls"], it["outcome"]
        use_hist = bool(it.get("use_history"))
        if not isinstance(req, str) or not (6 <= len(req) <= 300) or not VI.search(req):
            probs.append("request must be a Vietnamese sentence of 6-300 characters")
        elif norm(req) in real_requests:
            probs.append("request copies a real training request; write a new one")
        elif norm(req) in seen:
            probs.append("duplicate request (already used)")
        else:
            bg = bigrams(req)
            near = max(((jaccard(bg, b), r0) for b, r0 in seen_bg), default=(0, ""))
            real = max(((jaccard(bg, b), r0) for b, r0 in real_bg), default=(0, ""))
            if near[0] >= NEAR_COPY:
                probs.append(f"near-copy of an already accepted request (overlap {near[0]:.2f}): {near[1][:90]!r}. Write a "
                             f"different sentence and task, not the same frame with names, amounts or times swapped")
            elif real[0] >= NEAR_REAL and len(bg) >= 4:
                probs.append(f"near-copy of a real training request (overlap {real[0]:.2f}): {real[1][:90]!r}. Write a new one")
        if use_hist and not card["history"]:
            probs.append("use_history is true but this card has no history")
        if outcome not in ("act", "ask", "none"):
            probs.append(f"outcome must be act/ask/none, got {outcome!r}")
        if not isinstance(calls, list) or any(not isinstance(c, dict) or set(c) != {"name", "arguments"}
                                               or not isinstance(c["arguments"], dict) for c in calls):
            probs.append("calls must be a list of {name, arguments{}}")
            print(f"PROBLEM line {n}: " + "; ".join(probs)); continue
        if len(calls) > MAX_CALLS:
            probs.append(f"more than {MAX_CALLS} calls")
        names = [c["name"] for c in calls]
        if outcome == "none" and calls:
            probs.append("outcome none must have no calls")
        if outcome == "act" and (not calls or ASK in names):
            probs.append("outcome act needs >= 1 call and no ask")
        if outcome == "ask" and (not calls or names[-1] != ASK or names.count(ASK) != 1):
            probs.append("outcome ask must end with exactly one sys.ask_missing_info")
        installed = set(card["installed_tools"])
        for c in calls:
            if c["name"] not in installed:
                probs.append(f"{c['name']} is not installed in this card")
            elif c["name"] == ASK:
                a = c["arguments"]
                tn, ps = a.get("tool_name"), a.get("params")
                if set(a) != {"tool_name", "params"} or tn not in installed or tn == ASK:
                    probs.append("ask arguments must be {tool_name: <installed tool>, params: [...]}")
                elif not isinstance(ps, list) or not ps or any(p not in index[tn]["parameters"].get("properties", {}) for p in ps):
                    probs.append(f"ask params must be parameter names of {tn}")
            else:
                probs += arg_problems(c["name"], c["arguments"], index[c["name"]])
                probs += convention_problems(req, c, card["context"].get("user_profile") or {})
        if outcome == "ask" and not probs:
            if UNDECIDED.search(req):
                probs.append("ask request says the user hasn't decided yet (\"chưa chốt\", \"để sau\"...); real users just leave "
                             "the value out (2/217 real asks say so). Rewrite it without that clause")
            hist_calls = {x["name"] for t in card["history"] for x in (t.get("tool_calls") or [])} if use_hist else set()
            for k, src in unresolved_before_ask(req, calls[-1]["arguments"], set(names[:-1]), hist_calls, installed):
                probs.append(f"before asking, call {src} to resolve {k} (annotators resolve every ID they can, then ask)")
        if probs:
            print(f"PROBLEM line {n} ({it['card']}): " + "; ".join(probs[:4])); continue
        # replay in a fresh simulator of the card's environment
        env = zt.ToolEnv("train", card["env_sample"], root=root)
        known = json.dumps(card["history"], ensure_ascii=False) if use_hist else ""
        results = []
        for i, c in enumerate(calls):
            props = index[c["name"]]["parameters"].get("properties", {}) if c["name"] != ASK else {}
            for k, v in c["arguments"].items():
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    if isinstance(x, str) and (props.get(k, {}).get("x-ref") or ID_LIKE.match(x)) and f'"{x}"' not in known:
                        probs.append(f"call {i + 1} {c['name']}.{k}={x!r}: this ID was not returned by an earlier call "
                                     f"in this item{' or the history' if use_hist else ''} (add the read that finds it)")
            res = env.call(c["name"], c["arguments"])
            results.append(res)
            msg = json.dumps(res, ensure_ascii=False).lower()
            if not res.get("ok"):
                probs.append(f"call {i + 1} {c['name']} rejected by the simulator: {res.get('error')}")
            elif res.get("done") is False or any(s in msg for s in NOT_FOUND):
                probs.append(f"call {i + 1} {c['name']} did nothing: {res.get('message', '')[:120]}")
            known += json.dumps(res, ensure_ascii=False)
        if not probs:
            hres = [t.get("tool_results") for t in card["history"]] if use_hist else []
            probs += [f"wrong entity? {w}" for w in wrong_entity(req, calls, results, hres)]
        if probs:
            print(f"PROBLEM line {n} ({it['card']}): " + "; ".join(probs[:4])); continue
        for c in calls:   # soft style check: reads called with other arguments than annotators usually use
            if c["name"] == "home_automation.create_rule" and "title" not in c["arguments"]:
                print(f"WARN line {n} ({it['card']}): home_automation.create_rule has no 'title'; annotators give a short "
                      f"Vietnamese one such as \"Bật đèn sân buổi tối\" in 16/20 real rules")
            pat = READ_STYLE.get(c["name"])
            if pat is not None and tuple(sorted(c["arguments"])) != pat:
                print(f"WARN line {n} ({it['card']}): {c['name']} usually gets arguments {{{', '.join(pat)}}} in real "
                      f"annotations, here {{{', '.join(sorted(c['arguments']))}}} (accepted, but check it is needed)")
        if REASON.search(req):
            print(f"WARN line {n} ({it['card']}): request explains why the user wants it (\"vì…\", \"sợ…\", \"cho đỡ…\"); "
                  f"only 1% of real requests do. Get length from a second task or a condition instead")
        seen.add(norm(req))
        seen_bg.append((bigrams(req), req))
        ok_items.append({**it, "use_history": use_hist, "env_sample": card["env_sample"], "results": results,
                         "state_hash": env.get_state_hash()})
        stats[outcome] += 1
        stats["history"] += use_hist
        tools_used.update(c["name"] for c in calls)
    out = accepted_dir / Path(sys.argv[1]).name
    with open(out, "w", encoding="utf-8") as f:
        for it in ok_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_ok = len(ok_items)
    print(f"accepted mix: act {stats['act']}, ask {stats['ask']}, none {stats['none']} | with history {stats['history']} | "
          f"calls per accepted item {sum(len(i['calls']) for i in ok_items) / max(n_ok, 1):.2f} | distinct tools {len(tools_used)}")
    print(f"accepted items written to {out}")
    # batch-level shape targets, from the real final turns (multi-write acts 39%, conditionals 13%, reason clauses 1%)
    acts = [i for i in ok_items if i["outcome"] == "act"]
    multi = sum(sum(index[c["name"]]["kind"] == "write" for c in i["calls"]) >= 2 for i in acts) / max(len(acts), 1)
    cond = sum(bool(COND.search(i["request"])) for i in ok_items) / max(n_ok, 1)
    reason = sum(bool(REASON.search(i["request"])) for i in ok_items) / max(n_ok, 1)
    checks = [("acts with 2+ writes", multi, ">=", 0.35), ("conditional requests", cond, ">=", 0.10),
              ("requests with a reason clause", reason, "<=", 0.05)]
    parts, met = [], True
    for name, v, op, t in checks:
        good = v >= t if op == ">=" else v <= t
        met &= good
        parts.append(f"{name} {v:.0%} (need {op} {t:.0%}) {'OK' if good else 'NOT OK'}")
    print("BATCH TARGETS: " + " | ".join(parts) + (" -> TARGETS MET" if met else " -> TARGETS NOT MET"))
    print(f"VALIDATION: {n_ok}/{total} accepted")


if __name__ == "__main__":
    main()
