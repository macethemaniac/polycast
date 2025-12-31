from polycast.arbitrage.engine import ArbConfig, _edge_after_costs, _available_size, compute_opportunities


def test_edge_after_costs():
    cfg = ArbConfig(fee_buy=0.01, fee_sell=0.01, slippage=0.0)
    edge = _edge_after_costs(0.4, 0.6, cfg)
    assert round(edge, 4) == 0.18  # 0.6-0.01 - (0.4+0.01)


def test_available_size():
    from polycast.adapters import OrderBook
    ob_buy = OrderBook(best_bid=None, bid_size=None, best_ask=0.4, ask_size=10)
    ob_sell = OrderBook(best_bid=0.6, bid_size=5, best_ask=None, ask_size=None)
    assert _available_size(ob_buy, ob_sell) == 5


def test_compute_opportunities_filters(monkeypatch):
    # Mock adapters to return fixed markets/orderbooks
    class M:
        def __init__(self, mid, title):
            self.id = mid
            self.title = title
            self.outcomes = []
            self.updated_ts = None
            self.raw = {"id": mid, "title": title}

    pm_markets = [M("pm1", "Test PM")]
    kal_markets = [M("kal1", "Test Kal")]

    from polycast import adapters
    adapters_cache = {
        "pm": pm_markets,
        "kal": kal_markets,
    }

    def mock_list_markets_pm(return_debug=False, limit=100):
        return pm_markets

    def mock_list_markets_kal(return_debug=False, limit=100):
        return kal_markets

    from polycast.adapters import polymarket as pm_adapter
    from polycast.adapters import kalshi as kal_adapter

    monkeypatch.setattr(pm_adapter, "list_markets", mock_list_markets_pm)
    monkeypatch.setattr(kal_adapter, "list_markets", mock_list_markets_kal)

    from polycast.adapters import OrderBook

    ob_yes = OrderBook(best_bid=0.55, bid_size=10, best_ask=0.45, ask_size=10)
    ob_no = OrderBook(best_bid=0.6, bid_size=10, best_ask=0.52, ask_size=10)
    monkeypatch.setattr(pm_adapter, "get_orderbook", lambda mid, side: ob_yes if side == "YES" else ob_no)
    monkeypatch.setattr(kal_adapter, "get_orderbook", lambda mid, side: ob_no if side == "NO" else ob_yes)

    match_results = {
        "pm1": {
            "pm_title": "Test PM",
            "matches": [
                {"kal_id": "kal1", "kal_title": "Test Kal", "score": 0.8},
            ],
        }
    }

    cfg = ArbConfig(min_edge=0.01, min_size=1, max_staleness=3600, fee_buy=0.0, fee_sell=0.0, slippage=0.0)
    opps, rej = compute_opportunities(match_results, cfg)
    assert opps, "Should produce an opportunity"
    assert rej == {}
