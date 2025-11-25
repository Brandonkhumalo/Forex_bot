# TradingAI - ML-Powered Forex Trading Platform

## Overview
A comprehensive ML-based trading AI engine using Capital.com API with Python/Django REST Framework backend and React/Tailwind frontend. The system performs autonomous 24/7 forex/commodities trading on a demo account, combining technical analysis with machine learning predictions.

## Project Structure

```
├── src/             # React frontend (TypeScript + Tailwind)
├── backend/         # Django REST Framework backend (Python)
├── index.html       # Frontend entry point
└── attached_assets/ # Static assets
```

## Architecture

### Frontend (React + TypeScript + Tailwind)
- **Location**: `src/`
- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS with shadcn/ui components
- **Routing**: wouter
- **State Management**: TanStack Query for server state
- **Charts**: Recharts for visualizations
- **Port**: 5000 (Vite dev server with proxy to Django)

### Backend (Django + Django REST Framework)
- **Location**: `backend/`
- **Framework**: Django 5.x with DRF
- **Authentication**: JWT via djangorestframework-simplejwt
- **Database**: PostgreSQL (via DATABASE_URL)
- **ML**: scikit-learn for trading predictions
- **Port**: 8000

## Key Files

### Backend
- `backend/api/models.py` - Database models (User, Trade, MLModel, etc.)
- `backend/api/views.py` - API endpoints
- `backend/api/capital_api.py` - Capital.com API integration
- `backend/api/technical_analysis.py` - RSI, MACD, Bollinger Bands analysis
- `backend/api/ml_engine.py` - Machine learning prediction engine
- `backend/api/trading_engine.py` - Autonomous trading logic

### Frontend
- `src/pages/dashboard.tsx` - Main trading dashboard with AI controls
- `src/pages/analytics.tsx` - Performance analytics and ML metrics
- `src/pages/login.tsx` - User authentication
- `src/components/app-sidebar.tsx` - Navigation sidebar
- `src/lib/queryClient.ts` - API client with JWT auth

## Running the Application

```bash
npm run dev
```
This starts both Django backend (port 8000) and Vite frontend (port 5000).

### Start Backend Only
```bash
cd backend && python manage.py runserver 0.0.0.0:8000
```

### Start Frontend Only
```bash
npm run dev:frontend
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

## Separating Frontend and Backend

To split this project into separate repositories:

**Frontend repo:**
1. Copy: `src/`, `index.html`, `vite.config.ts`, `tailwind.config.ts`, `tsconfig.json`, `package.json`
2. Update `vite.config.ts` proxy target to point to your backend URL

**Backend repo:**
1. Copy: `backend/` folder
2. Add CORS settings for your frontend domain

## Design Guidelines
- Professional trading platform aesthetic
- Inter font for UI, JetBrains Mono for numerical data
- Green for profits/wins, red for losses
- Dark mode support via ThemeProvider
