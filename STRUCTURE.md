# Project Structure

This document describes the organization of the Polycast repository.

## Directory Layout

```
polycast/
├── .github/                    # GitHub specific files (CI/CD workflows)
├── src/                        # Source code
│   ├── __init__.py            # Package initialization
│   ├── data/                  # Data storage
│   │   ├── __init__.py
│   │   └── alerts_chats.json # Telegram chat configurations
│   ├── exchanges/             # Exchange API integrations
│   │   ├── __init__.py
│   │   ├── binance.py        # Binance API wrapper
│   │   ├── bybit.py          # Bybit API wrapper
│   │   ├── coingecko.py      # CoinGecko API wrapper
│   │   ├── defillama.py      # DeFiLlama API wrapper
│   │   └── polymarket.py     # Polymarket API wrapper
│   └── polycast/              # Main application package
│       ├── __init__.py
│       ├── analytics/         # Arbitrage calculation logic
│       │   ├── __init__.py
│       │   ├── arbitrage.py         # General arbitrage detection
│       │   └── prediction_arbitrage.py  # Prediction market specific
│       ├── bot.py            # Telegram bot entry point
│       ├── cross_arb.py      # Cross-market arbitrage (Polymarket/Kalshi)
│       ├── kalshi_api.py     # Kalshi API integration
│       ├── main.py           # Console scanner entry point
│       ├── polymarket_api.py # Polymarket API integration
│       └── scanner.py        # Core scanning logic
├── scripts/                   # Utility scripts
│   ├── derive_polymarket_creds.py  # Credential derivation
│   ├── diag_cross_pairs.py        # Diagnostic tool for pairs
│   ├── explore_polymarket.py      # Polymarket exploration
│   ├── run_cross_arb.py          # Cross-arbitrage runner
│   └── test_bot_commands.py      # Bot command testing
├── tests/                     # Test suite (to be implemented)
│   └── __init__.py
├── docs/                      # Additional documentation
│   ├── BOT_SETUP.md          # Bot setup guide
│   └── POLYMARKET_API.md     # Polymarket API documentation
├── .dockerignore             # Docker build exclusions
├── .env.example              # Environment variable template
├── .gitattributes            # Git line ending configuration
├── .gitignore                # Git ignore rules
├── DEPLOYMENT.md             # Deployment guide
├── Dockerfile                # Docker container definition
├── docker-compose.yml        # Docker Compose configuration
├── LICENSE                   # Project license
├── MANIFEST.in               # Package manifest for distribution
├── Procfile                  # Heroku deployment configuration
├── pyproject.toml            # Modern Python project configuration
├── README.md                 # Project overview and quick start
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── runtime.txt               # Python version for Heroku
├── setup.py                  # Package setup (legacy compatibility)
├── start_bot.bat            # Windows batch startup script
├── start_bot.ps1            # PowerShell startup script
└── STRUCTURE.md             # This file
```

## Module Descriptions

### Core Modules

#### `src/polycast/bot.py`
Telegram bot interface for the arbitrage scanner. Provides commands for:
- `/start` - Welcome message
- `/scan` - Scan for crypto arbitrage
- `/price <pair>` - Get prices for specific pair
- `/cross` - Check cross-market arbitrage opportunities

#### `src/polycast/scanner.py`
Core scanning logic that orchestrates price fetching and arbitrage detection.

#### `src/polycast/main.py`
Console mode entry point for direct command-line execution.

### Exchange Integrations

All exchange modules in `src/exchanges/` provide standardized interfaces for:
- Price fetching
- Market data retrieval
- Rate limiting
- Error handling

#### `src/exchanges/coingecko.py`
Free cryptocurrency price API integration.

#### `src/exchanges/defillama.py`
DeFi protocol and price aggregator integration.

#### `src/exchanges/polymarket.py`
Polymarket prediction market API wrapper.

#### `src/exchanges/binance.py`
Binance exchange API wrapper (for future use).

#### `src/exchanges/bybit.py`
Bybit exchange API wrapper (for future use).

### Prediction Markets

#### `src/polycast/polymarket_api.py`
Polymarket-specific arbitrage detection and market fetching.

#### `src/polycast/kalshi_api.py`
Kalshi prediction market API integration.

#### `src/polycast/cross_arb.py`
Cross-market arbitrage detection between Polymarket and Kalshi.
Implements fuzzy matching to find equivalent markets.

### Analytics

#### `src/polycast/analytics/arbitrage.py`
Core arbitrage calculation algorithms:
- Spread calculation
- Percentage difference
- Profit estimation

#### `src/polycast/analytics/prediction_arbitrage.py`
Specialized arbitrage logic for prediction markets:
- YES/NO price comparison
- Complementary outcome arbitrage

### Utility Scripts

#### `scripts/derive_polymarket_creds.py`
Derives API credentials from Polymarket wallet keys.

#### `scripts/diag_cross_pairs.py`
Diagnostic tool for analyzing cross-market pairs and their matching quality.

#### `scripts/explore_polymarket.py`
Interactive exploration tool for Polymarket data.

#### `scripts/run_cross_arb.py`
Standalone script to run cross-market arbitrage detection.

#### `scripts/test_bot_commands.py`
Testing utility for Telegram bot commands.

## Configuration Files

### `pyproject.toml`
Modern Python project configuration containing:
- Package metadata
- Dependencies
- Build system configuration
- Tool configurations (black, pytest, mypy)

### `requirements.txt`
Production dependencies with version pinning.

### `requirements-dev.txt`
Development dependencies (testing, linting, type checking).

### `.env.example`
Template for environment variables. Copy to `.env` and fill in:
- `TELEGRAM_BOT_TOKEN` - Required for bot
- `TELEGRAM_CHAT_ID` - Optional for notifications
- API keys for Polymarket/Kalshi if trading

### `.gitignore`
Excludes sensitive and generated files:
- Environment files (`.env`)
- Python cache (`__pycache__`)
- Virtual environments
- Sensitive tokens

## Data Storage

### `src/data/alerts_chats.json`
Stores Telegram chat configurations for alerts.

Structure:
```json
{
  "chat_id": {
    "alerts_enabled": true,
    "threshold": 0.5
  }
}
```

## Entry Points

### Bot Mode (Telegram)
```bash
python src/polycast/bot.py
# or
polycast-bot  (after pip install)
```

### Console Mode
```bash
python src/polycast/main.py
# or
polycast-scan  (after pip install)
```

### Cross-Market Arbitrage
```bash
python scripts/run_cross_arb.py
```

## Testing Structure

Tests should be organized to mirror the source structure:

```
tests/
├── __init__.py
├── test_exchanges/
│   ├── __init__.py
│   ├── test_coingecko.py
│   └── test_defillama.py
├── test_polycast/
│   ├── __init__.py
│   ├── test_bot.py
│   ├── test_scanner.py
│   └── test_analytics/
│       ├── __init__.py
│       └── test_arbitrage.py
└── test_integration/
    ├── __init__.py
    └── test_end_to_end.py
```

## Deployment Artifacts

### Docker
- `Dockerfile` - Container image definition
- `docker-compose.yml` - Multi-container orchestration
- `.dockerignore` - Excludes unnecessary files from image

### Heroku
- `Procfile` - Process type definitions
- `runtime.txt` - Python version specification

### Systemd (Linux)
See `DEPLOYMENT.md` for service file template.

## Development Workflow

1. **Local Development:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   pip install -r requirements-dev.txt
   ```

2. **Run Tests:**
   ```bash
   pytest
   ```

3. **Code Formatting:**
   ```bash
   black src/
   ```

4. **Type Checking:**
   ```bash
   mypy src/
   ```

5. **Linting:**
   ```bash
   flake8 src/
   ```

## Adding New Features

### Adding a New Exchange

1. Create `src/exchanges/new_exchange.py`
2. Implement standard interface:
   ```python
   def get_price(symbol: str) -> float:
       # Implementation
   ```
3. Add tests in `tests/test_exchanges/test_new_exchange.py`
4. Update documentation

### Adding a New Bot Command

1. Add handler in `src/polycast/bot.py`:
   ```python
   async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
       # Implementation

   application.add_handler(CommandHandler("newcmd", new_command))
   ```
2. Update help message
3. Add tests

### Adding a New Arbitrage Strategy

1. Create module in `src/polycast/analytics/`
2. Implement detection logic
3. Integrate with scanner or bot
4. Add tests

## Dependencies Management

### Adding a Production Dependency
1. Add to `pyproject.toml` under `[project]` → `dependencies`
2. Update `requirements.txt`
3. Test in clean environment

### Adding a Development Dependency
1. Add to `pyproject.toml` under `[project.optional-dependencies]` → `dev`
2. Update `requirements-dev.txt`

## Package Distribution

### Building
```bash
python -m build
```

### Installing Locally
```bash
pip install -e .
```

### Publishing (when ready)
```bash
python -m twine upload dist/*
```

## Maintenance

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt  # Update pinned versions
```

### Security Scanning
```bash
pip install safety
safety check -r requirements.txt
```

### Database Migrations
Currently not applicable (no database). Will be documented when database is added.

## Version Control

- **Main branch:** Production-ready code
- **Develop branch:** Integration branch for features
- **Feature branches:** `feature/description`
- **Bugfix branches:** `bugfix/description`

## Contributing

See `CONTRIBUTING.md` (to be created) for contribution guidelines.

## Questions?

- Check `README.md` for general information
- See `DEPLOYMENT.md` for deployment instructions
- See `BOT_SETUP.md` for bot configuration
- See `POLYMARKET_API.md` for Polymarket integration details
