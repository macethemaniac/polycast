from polycast.bot import build_fingerprint, should_alert


def test_should_alert_new_and_improved():
    seen = {}
    now = 1_000_000
    fp = build_fingerprint("buy", "sell", "market1", 1.0, 1.5)
    ok, tag, prev = should_alert(seen, fp, edge=1.0, cooldown_sec=1800, improve_threshold=0.5, now_ts=now)
    assert ok and tag == "NEW"
    seen[fp] = {"ts": now, "edge": 1.0}
    ok, tag, prev = should_alert(seen, fp, edge=1.2, cooldown_sec=1800, improve_threshold=0.1, now_ts=now + 60)
    assert ok and tag == "IMPROVED" and prev == 1.0
    ok, tag, prev = should_alert(seen, fp, edge=1.21, cooldown_sec=1800, improve_threshold=0.5, now_ts=now + 120)
    assert not ok
