"""Hand-computed scorer cases. Unless stated otherwise the hash matches."""
from __future__ import annotations

import json
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.eval.metric import explain_sample, fuzzy_keys, score_sample, token_f1  # noqa: E402

FZ = {"reminders.create": frozenset({"content"}), "notes.create": frozenset({"title", "content"})}
H = "h" * 64


def call(name, **args):
    return {"name": name, "arguments": args}


a = call("alarm.list")
b = call("alarm.create", time="2026-11-11T05:30:00+07:00")
c = call("reminders.create", content="đổ xăng", time="2026-11-11T07:00:00+07:00")
x = call("weather.get_current")


def sc(S, G, variant="primary", sub_hash=H, gold_hash=H, fz=FZ):
    return score_sample(S, sub_hash, G, gold_hash, fz, variant)


def rem(content):
    return call("reminders.create", content=content, time="2026-11-11T07:00:00+07:00")


def test_01_empty_empty():
    assert sc([], []) == 1


def test_02_empty_gold_any_call():
    assert sc([a], []) == 0


def test_03_exact():
    assert sc([a, b, c], [a, b, c]) == 1


def test_04_hash_differs():
    assert sc([a, b, c], [a, b, c], sub_hash="0" * 64) == 0


def test_05_extra_read():
    assert sc([a, b, c, x], [a, b, c]) == Fraction(6, 7)


def test_06_missing_read():
    assert sc([a, c], [a, b, c]) == Fraction(4, 5)


def test_07_rejected_call_counts():
    r = call("alarm.create")  # rejected by schema (missing time), still submitted
    rec = explain_sample([a, b, c, r], H, [a, b, c], H, FZ, n_rejected=1)
    assert rec["score"] == Fraction(6, 7) and rec["n_rejected"] == 1 and rec["unmatched_sub"] == [r]


def test_08_hash_ok_but_no_match():
    assert sc([call("alarm.create", time="2026-11-12T05:30:00+07:00")], [b]) == 0


def test_09_order_ignored():
    assert sc([c, a, b], [a, b, c]) == 1


def test_10_duplicate_gold():
    assert sc([a], [a, a]) == Fraction(2, 3)


def test_11_duplicate_sub():
    assert sc([a, a], [a]) == Fraction(2, 3)


def test_12_greedy_trap():
    g1, g2 = rem("a b c d e"), rem("a b c x y")
    s1, s2 = rem("a b c d x"), rem("a b c d e")   # s1 fits g1 and g2; s2 fits only g1
    assert sc([s1, s2], [g1, g2]) == 1
    assert explain_sample([s1, s2], H, [g1, g2], H, FZ)["m"] == 2


def test_13_fuzzy_exactly_06_is_no_match():
    assert token_f1("a b c x y", "a b c z w", "word") == Fraction(3, 5)
    assert sc([rem("a b c x y")], [rem("a b c z w")]) == 0


def test_14_fuzzy_075_matches():
    assert sc([rem("a b c")], [rem("a b c d e")]) == 1


def test_15_fuzzy_case_and_nfd():
    nfd = unicodedata.normalize("NFD", "mai chạy")
    assert nfd != "mai chạy"
    assert sc([rem("Mai CHẠY")], [rem(nfd)]) == 1


def test_16_nonfuzzy_case():
    s, g = call("contacts.search", query="sếp dũng"), call("contacts.search", query="Sếp Dũng")
    assert sc([s], [g], "primary") == 0 and sc([s], [g], "pessimistic") == 0
    assert sc([s], [g], "optimistic") == 1


def test_17_true_vs_1():
    s, g = call("alarm.update", alarm_id="alm_01", enabled=True), call("alarm.update", alarm_id="alm_01", enabled=1)
    for v in ("primary", "pessimistic", "optimistic"):
        assert sc([s], [g], v) == 0, v


def test_18_int_vs_float():
    s, g = call("smart_home.climate_set", device_id="ac", temperature=2), call("smart_home.climate_set", device_id="ac", temperature=2.0)
    assert sc([s], [g], "primary") == 0
    assert sc([s], [g], "optimistic") == 1


def test_19_extra_or_missing_optional_key():
    g = call("calendar.list_events")
    s = call("calendar.list_events", date_from="2026-11-11")
    assert sc([s], [g]) == 0 and sc([g], [s]) == 0


def test_20_extra_empty_key():
    g = call("contacts.search", query="Lan")
    s = call("contacts.search", query="Lan", relation="")
    assert sc([s], [g], "primary") == 0
    assert sc([s], [g], "optimistic") == 1


def test_21_scalar_list_order():
    s = call("email.send", to=["a@x.vn", "b@x.vn"], subject="s", body="b")
    g = call("email.send", to=["b@x.vn", "a@x.vn"], subject="s", body="b")
    assert sc([s], [g], "primary") == 0
    assert sc([s], [g], "optimistic") == 1


def test_22_nested_fields_not_fuzzy():
    fz = {"food_delivery.place_order": frozenset({"note"})}
    s = call("food_delivery.place_order", restaurant_id="rest_01", items=[{"item_id": "m1", "qty": 1, "note": "ít cay nhé"}])
    g = call("food_delivery.place_order", restaurant_id="rest_01", items=[{"item_id": "m1", "qty": 1, "note": "ít cay"}])
    for v in ("primary", "pessimistic", "optimistic"):
        assert sc([s], [g], v, fz=fz) == 0, v


def test_23_ask_missing_info():
    g = call("sys.ask_missing_info", tool_name="sms.send", params=["contact_id"])
    assert sc([call("sys.ask_missing_info", tool_name="sms.send", params=["contact_id"])], [g]) == 1
    assert sc([call("sys.ask_missing_info", tool_name="sms.send", params=["content"])], [g]) == 0


def test_24_fuzzy_keys_per_split():
    from lifemate.config import data_root, load_config
    from tests.runner import Skip
    DATA = data_root(load_config())
    if not DATA.exists():
        raise Skip("data folder not present")
    tr = fuzzy_keys(json.load(open(DATA / "train/train_tools_catalog.json", encoding="utf-8")))
    pu = fuzzy_keys(json.load(open(DATA / "public_test/public_test_tools_catalog.json", encoding="utf-8")))
    assert "text" in pu["translator.translate"] and "translator.translate" not in tr
    assert any("expression" in v for v in tr.values())
    assert not any("expression" in v for v in pu.values())
    s = call("translator.translate", text="xin chào các bạn nhé", target="en")
    g = call("translator.translate", text="xin chào các bạn", target="en")
    assert sc([s], [g], fz=pu) == 1      # fuzzy under the public catalog (F1 = 8/9)
    assert sc([s], [g], fz=tr) == 0      # exact under a catalog without that marker


if __name__ == "__main__":
    from tests.runner import run
    run(globals())
