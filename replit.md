# TradingAI - ML-Powered Forex Trading Platform

## Overview
A comprehensive ML-based trading AI engine using Capital.com API with Python/Django REST Framework backend and React/Tailwind frontend. The system performs autonomous 24/7 forex/commodities trading on a demo account, combining technical analysis with machine learning predictions.

## Architecture

### Frontend (React + TypeScript + Tailwind)
- **Location**: `client/src/`
- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS with shadcn/ui components
- **Routing**: wouter
- **State Management**: TanStack Query for server state
- **Charts**: Recharts for visualizations

### Backend (Django + Django REST Framework)
- **Location**: `backend/`
- **Framework**: Django 5.x with DRF
- **Authentication**: JWT via djangorestframework-simplejwt
- **Database**: PostgreSQL (via DATABASE_URL)
- **ML**: scikit-learn for trading predictions

### Express Proxy Server
- **Location**: `server/`
- **Purpose**: Proxies API calls to Django backend
- **Port**: 5000 (serves both frontend and proxied API)

## Key Files

### Backend
- `backend/api/models.py` - Database models (User, Trade, MLModel, etc.)
- `backend/api/views.py` - API endpoints
- `backend/api/capital_api.py` - Capital.com API integration
- `backend/api/technical_analysis.py` - RSI, MACD, Bollinger Bands analysis
- `backend/api/ml_engine.py` - Machine learning prediction engine
- `backend/api/trading_engine.py` - Autonomous trading logic

### Frontend
- `client/src/pages/dashboard.tsx` - Main trading dashboard with AI controls
- `client/src/pages/analytics.tsx` - Performance analytics and ML metrics
- `client/src/pages/login.tsx` - User authentication
- `client/src/components/app-sidebar.tsx` - Navigation sidebar

## Running the Application

### Start Frontend (Express + Vite)
```bash
npm run dev
```
This starts on port 5000 and proxies `/api/*` to Django.

### Start Django Backend
```bash
./start_django.sh
# Or manually:
cd backend && python manage.py runserver 0.0.0.0:8000
```

### Database Migrations
```bash
cd backend && python manage.py migrate
```

## API Endpoints

### Authentication
- `POST /api/auth/register/` - Create account
- `POST /api/auth/login/` - Login (returns JWT)
- `GET /api/auth/me/` - Current user info

### Trading
- `GET /api/dashboard/` - Dashboard data
- `GET /api/analytics/` - Trading analytics
- `POST /api/settings/toggle-ai/` - Start/stop AI trading
- `GET /api/trades/open/` - Active trades
- `GET /api/trades/history/` - Closed trades

### ML Status
- `GET /api/ml/status/` - ML model status and metrics

## Environment Variables Required
- `DATABASE_URL` - PostgreSQL connection string (auto-provided)
- `SESSION_SECRET` - Session encryption key
- `CAPITAL_COM_API_KEY` - Capital.com API key
- `CAPITAL_COM_PASSWORD` - Capital.com password
- `CAPITAL_COM_IDENTIFIER` - Capital.com account identifier

## ML Trading Logic

1. **Initial Phase (0-30 trades)**: Uses technical analysis only
   - RSI, MACD, Bollinger Bands
   - Moving Average crossovers
   - Supply/Demand zones
   - Market structure analysis

2. **ML Phase (30+ trades)**: Combines TA with ML predictions
   - RandomForest classifier trained on past trades
   - Auto-retrains every 10 new trades
   - Feature importance tracking
   - Confidence-weighted signals

## Trading Pairs
- GBP/USD, EUR/USD, USD/JPY, AUD/USD
- USD/CAD, NZD/USD, USD/CHF
- XAU/USD (Gold)

## Capital Limits
- Forex pairs: 20% of capital per pair
- Gold: 15% of capital
- Risk per trade: 2% of available capital

## Recent Changes (November 2025)
- Complete Django backend with JWT authentication
- Capital.com API integration with credential validation
- Technical analysis engine (RSI, MACD, Bollinger Bands, SMC)
- ML prediction system with scikit-learn (trains after 30 trades)
- 24/7 autonomous trading scheduler
- Dashboard with API status alerts and AI controls
- Analytics page with performance charts
- Database migrations run automatically on startup
- Express proxy properly forwards request bodies

## API Status Endpoint
- `GET /api/status/` - Check if Capital.com credentials are configured
- Returns: `{api_configured, api_connected, account_info, missing_credentials}`
- AI toggle is disabled until credentials are provided

## Design Guidelines
- Professional trading platform aesthetic
- Inter font for UI, JetBrains Mono for numerical data
- Green for profits/wins, red for losses
- Dark mode support via ThemeProvider
