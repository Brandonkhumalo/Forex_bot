# TradingAI - ML-Powered Autonomous Forex Trading Platform

A comprehensive machine learning-based trading engine that connects to Capital.com API for autonomous 24/7 forex and commodities trading on a demo account. The system combines advanced technical analysis with per-pair machine learning predictions for intelligent trade execution.

## Overview

TradingAI is a fully autonomous trading system that analyzes 10 currency pairs and commodities every minute, using a sophisticated multi-layer decision engine. The platform features:

- **Autonomous Trading**: Runs 24/7 without manual intervention
- **Per-Pair ML Models**: Each trading pair trains its own machine learning model after 5 closed trades
- **Multi-Strategy Analysis**: Combines 5 different trading strategies for robust signal generation
- **Risk Management**: Automatic position sizing, stop-loss, and take-profit management
- **Real-Time Dashboard**: Live monitoring of trades, performance, and ML model status

### Trading Pairs
- **Forex**: GBP/USD, EUR/USD, USD/JPY, AUD/USD, USD/CAD, NZD/USD, USD/CHF, GBP/JPY, EUR/JPY
- **Commodities**: XAU/USD (Gold)

## Algorithms & Strategies

### Technical Analysis Strategies

#### 1. Smart Money Concepts (SMC)
**Location**: `backend/api/technical_analysis.py`

**Purpose**: Identifies institutional trading patterns and order flow

**Algorithms Used**:
- **Order Block Detection**: Identifies zones where large institutional orders were placed
- **Fair Value Gap (FVG)**: Detects price imbalances that tend to get filled
- **Break of Structure (BOS)**: Recognizes trend continuation/reversal signals
- **Premium/Discount Zones**: Calculates optimal entry zones based on Fibonacci retracements
- **Liquidity Pool Detection**: Identifies stop-loss clusters that institutions target

#### 2. Price Action Analysis
**Location**: `backend/api/technical_analysis.py`

**Purpose**: Reads raw price movements and patterns

**Algorithms Used**:
- **Market Structure Analysis**: Detects Higher Highs/Higher Lows (uptrend) and Lower Highs/Lower Lows (downtrend)
- **Supply/Demand Zone Detection**: Identifies areas of strong buying/selling pressure
- **Candlestick Pattern Recognition**: Detects reversal and continuation patterns (engulfing, pin bars, doji)
- **Swing Point Detection**: Identifies key pivot points for trend analysis

#### 3. Trend Following Indicators
**Location**: `backend/api/technical_analysis.py`

**Purpose**: Confirms trend direction and momentum

**Algorithms Used**:
- **MACD (Moving Average Convergence Divergence)**: Signal line crossovers with histogram analysis
- **RSI (Relative Strength Index)**: Overbought/oversold conditions with divergence detection
- **Moving Average Crossovers**: SMA 20/50/200 for trend confirmation
- **ADX (Average Directional Index)**: Measures trend strength (>25 = strong trend)
- **Bollinger Bands**: Volatility measurement and mean reversion signals

#### 4. Risk-Based Analysis
**Location**: `backend/api/technical_analysis.py`

**Purpose**: Calculates optimal entry/exit points with risk management

**Algorithms Used**:
- **ATR (Average True Range)**: Dynamic stop-loss and take-profit calculation
- **Risk/Reward Ratio**: Ensures minimum 1:1 R:R for all trades
- **Position Sizing**: Kelly Criterion-inspired sizing based on confidence and account risk

### Machine Learning System

#### Ensemble Model Architecture
**Location**: `backend/api/ml_engine.py`

**Purpose**: Predicts trade outcomes using historical data

**Models Used**:

1. **RandomForestClassifier** (Base Model 1)
   - 100 decision trees with max depth of 10
   - Handles non-linear relationships in market data
   - Provides feature importance rankings

2. **GradientBoostingClassifier** (Base Model 2)
   - Sequential tree building for error correction
   - 50 estimators with learning rate of 0.1
   - Captures complex patterns missed by RandomForest

3. **LogisticRegression** (Meta-Model)
   - Stacks predictions from RF and XGBoost
   - Learns optimal weighting of base models
   - Final probability output for trade decisions

#### ML Features (25 total)

**Price-Based Features**:
- Close price, SMA ratios (20/50/200)
- Distance from highs/lows
- Bollinger Band position and width

**Momentum Features**:
- RSI, MACD signal/histogram
- ADX trend strength
- Rate of change indicators

**Volatility Features**:
- ATR and ATR_50 ratio
- Volatility regime classification (high/normal/low)

**Market Context**:
- Session indicator (Asia=0, London=1, NY=2, Overlap=3)
- DXY (USD strength) proxy calculation
- Trend age (bars since trend start)

#### Regime Detection System
**Location**: `backend/api/ml_engine.py`

**Purpose**: Adapts trading behavior to market conditions

**Algorithms**:
- **Volatility Regime Classifier**: Uses ATR ratio to classify markets as high/normal/low volatility
- **Trend Regime Classifier**: Analyzes SMA alignment to detect trending vs ranging conditions
- **Session Context**: Adjusts expectations based on trading session characteristics

#### Confidence Formula

```
Final Confidence = (TA Confidence × 0.4) + (ML Confidence × 0.4) + (Regime Score × 0.2)
```

- **TA Confidence**: Weighted average of all technical strategy signals
- **ML Confidence**: Probability from ensemble model prediction
- **Regime Score**: Bonus for favorable market conditions (trending + aligned direction)

### Trade Execution Logic

**Location**: `backend/api/trading_engine.py`

**Decision Flow**:
1. Analyze all timeframes (15m, 30m, 1H, 4H)
2. Calculate individual strategy signals
3. Check timeframe alignment
4. If pair has 5+ closed trades, get ML prediction
5. Calculate combined confidence score
6. Execute trade if confidence ≥ 30% and capital available

**Risk Rules**:
- Maximum 20% of capital per forex pair
- Maximum 15% of capital for Gold (XAU/USD)
- 2% risk per trade
- 1:1 risk/reward ratio (2× ATR for both SL and TP)

## Technology Stack

### Backend

| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Core backend language |
| **Django 5.x** | Web framework |
| **Django REST Framework** | RESTful API endpoints |
| **PostgreSQL** | Primary database |
| **scikit-learn** | Machine learning models |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computations |
| **TA-Lib** | Technical analysis calculations |
| **APScheduler** | Background job scheduling (1-minute trading cycles) |
| **requests** | Capital.com API integration |

### Frontend

| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **TypeScript** | Type-safe JavaScript |
| **Vite** | Build tool and dev server |
| **Tailwind CSS** | Utility-first styling |
| **shadcn/ui** | Pre-built UI components |
| **TanStack Query** | Server state management |
| **Recharts** | Trading charts and analytics |
| **wouter** | Client-side routing |
| **Framer Motion** | Animations |

### Infrastructure

| Technology | Purpose |
|------------|---------|
| **Capital.com API** | Live market data and trade execution |
| **JWT Authentication** | Secure API access |
| **Neon PostgreSQL** | Cloud database hosting |

## Project Structure

```
├── backend/                 # Django backend
│   ├── api/
│   │   ├── capital_api.py       # Capital.com API integration
│   │   ├── technical_analysis.py # TA strategies and indicators
│   │   ├── ml_engine.py         # ML models and predictions
│   │   ├── trading_engine.py    # Trade execution logic
│   │   ├── scheduler.py         # Background job scheduler
│   │   ├── models.py            # Database models
│   │   └── views.py             # API endpoints
│   └── trading_ai/
│       └── settings.py          # Django configuration
│
├── src/                     # React frontend
│   ├── pages/
│   │   ├── dashboard.tsx        # Main trading dashboard
│   │   ├── analytics.tsx        # Performance analytics
│   │   └── login.tsx            # Authentication
│   ├── components/
│   │   └── app-sidebar.tsx      # Navigation
│   └── lib/
│       └── queryClient.ts       # API client
│
└── index.html               # Frontend entry point
```

## How It Works

1. **Every Minute**: The scheduler triggers a trading cycle
2. **Data Collection**: Fetches latest price data for all pairs across 4 timeframes
3. **Technical Analysis**: Runs all 5 strategies on each pair
4. **ML Prediction**: For pairs with sufficient history, generates ML predictions
5. **Signal Aggregation**: Combines TA and ML signals with regime context
6. **Trade Execution**: Opens positions that meet confidence threshold
7. **Position Monitoring**: Tracks open trades and syncs with Capital.com
8. **Model Retraining**: Automatically retrains ML models every 3 new closed trades

## Key Features

- **Adaptive Learning**: ML models improve as more trades are completed
- **Regime Awareness**: Adjusts behavior for trending vs ranging markets
- **Session Optimization**: Considers trading session characteristics
- **Backward Compatibility**: Supports both legacy and new ML model formats
- **Real-Time Sync**: Maintains consistency with broker account state
- **Comprehensive Logging**: Detailed logs for debugging and analysis

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `CAPITAL_COM_API_KEY` | Capital.com API key |
| `CAPITAL_COM_PASSWORD` | Capital.com account password |
| `CAPITAL_COM_IDENTIFIER` | Capital.com account email |
| `SESSION_SECRET` | JWT session encryption key |

## Running the Application

```bash
npm run dev
```

This starts both the Django backend (port 8000) and Vite frontend (port 5000).

## Disclaimer

This is a demo trading system for educational purposes. Always test thoroughly on a demo account before considering any real trading. Past performance does not guarantee future results. Trading forex and commodities involves significant risk of loss.
