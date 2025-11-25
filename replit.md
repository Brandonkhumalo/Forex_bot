# TradingAI - ML-Powered Forex Trading Platform

## Overview
A comprehensive ML-based trading AI engine using Capital.com API with Python/Django REST Framework backend and React/Tailwind frontend. The system performs autonomous 24/7 forex/commodities trading on a demo account, combining technical analysis with machine learning predictions.

## Project Structure

```
├── frontend/        # React frontend (TypeScript + Tailwind)
├── backend/         # Django REST Framework backend (Python)
├── server/          # Minimal dev startup script
├── client -> frontend  # Symlink for Vite compatibility
└── attached_assets/ # Static assets
```

## Architecture

### Frontend (React + TypeScript + Tailwind)
- **Location**: `frontend/src/`
- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS with shadcn/ui components
- **Routing**: wouter
- **State Management**: TanStack Query for server state
- **Charts**: Recharts for visualizations
- **API**: Direct calls to Django via CORS (configurable via VITE_API_BASE_URL)

### Backend (Django + Django REST Framework)
- **Location**: `backend/`
- **Framework**: Django 5.x with DRF
- **Authentication**: JWT via djangorestframework-simplejwt
- **Database**: PostgreSQL (via DATABASE_URL)
- **ML**: scikit-learn for trading predictions
- **Port**: 8000

### Development Startup
- **Location**: `server/index-dev.ts`
- **Purpose**: Starts both Django (port 8000) and Vite (port 5000)
- Runs database migrations automatically on startup

## Key Files

### Backend
- `backend/api/models.py` - Database models (User, Trade, MLModel, etc.)
- `backend/api/views.py` - API endpoints
- `backend/api/capital_api.py` - Capital.com API integration
- `backend/api/technical_analysis.py` - RSI, MACD, Bollinger Bands analysis
- `backend/api/ml_engine.py` - Machine learning prediction engine
- `backend/api/trading_engine.py` - Autonomous trading logic

### Frontend
- `frontend/src/pages/dashboard.tsx` - Main trading dashboard with AI controls
- `frontend/src/pages/analytics.tsx` - Performance analytics and ML metrics
- `frontend/src/pages/login.tsx` - User authentication
- `frontend/src/components/app-sidebar.tsx` - Navigation sidebar
- `frontend/src/lib/queryClient.ts` - API client with JWT auth

## Running the Application

```bash
npm run dev
```
This starts both Django backend (port 8000) and Vite frontend (port 5000).

### Start Backend Only
```bash
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
- `GET /api/status/` - API configuration status

## Environment Variables Required
- `DATABASE_URL` - PostgreSQL connection string (auto-provided)
- `SESSION_SECRET` - Session encryption key
- `CAPITAL_COM_API_KEY` - Capital.com API key
- `CAPITAL_COM_PASSWORD` - Capital.com password
- `CAPITAL_COM_IDENTIFIER` - Capital.com account identifier
- `VITE_API_BASE_URL` - Frontend API base URL (defaults to http://localhost:8000)

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
- Restructured project: frontend/ (React) + backend/ (Django)
- Direct Django API calls via CORS (no Express proxy needed)
- Simplified server folder to minimal startup script
- JWT authentication with localStorage token storage
- Database migrations run automatically on startup

## API Status Endpoint
- `GET /api/status/` - Check if Capital.com credentials are configured
- Returns: `{api_configured, api_connected, account_info, missing_credentials}`
- AI toggle is disabled until credentials are provided

## Design Guidelines
- Professional trading platform aesthetic
- Inter font for UI, JetBrains Mono for numerical data
- Green for profits/wins, red for losses
- Dark mode support via ThemeProvider
