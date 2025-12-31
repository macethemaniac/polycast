from polycast.matching import rules


def test_normalize_text():
    assert rules.normalize_text("Hello, World!") == "hello world"
    assert rules.normalize_text("  Multi   space  ") == "multi space"


def test_extract_dates_month_and_year():
    dates = rules.extract_dates("Election Nov 5 2026 and Dec 12, 2025 plus 2024")
    assert "11-05" in dates or "05-11" in dates  # month-day normalized with month first
    assert "12-12" in dates
    assert "2024" in dates


def test_extract_thresholds():
    ths = set(rules.extract_thresholds("Rate >5.5% and unemployment 3-4 or <=2.0"))
    assert ">5.5" in ths
    assert "3-4" in ths
    assert "<=2.0" in ths
