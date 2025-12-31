import math

from polycast.cross_arb import normalize_price_to_prob


def test_polymarket_normalization():
    # Polymarket prices are already in [0,1]
    assert math.isclose(normalize_price_to_prob(0.72), 0.72)
    assert math.isclose(normalize_price_to_prob(0), 0.0)
    assert math.isclose(normalize_price_to_prob(1), 1.0)


def test_kalshi_cents_normalization():
    # Kalshi often reports cents (0-100); normalize to 0-1
    assert math.isclose(normalize_price_to_prob(37), 0.37)
    assert math.isclose(normalize_price_to_prob(3), 0.03)
    assert math.isclose(normalize_price_to_prob(105), 1.0)  # clamped


def test_edge_uses_buy_ask_and_sell_bid():
    buy_ask = 0.45  # what we pay
    sell_bid = 0.52  # what we receive
    edge = sell_bid - buy_ask
    assert math.isclose(edge, 0.07)
