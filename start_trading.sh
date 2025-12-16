#!/bin/bash

cd backend

echo "Starting AI Trading Engine..."
echo "This will analyze markets and execute trades automatically."
echo ""

python -c "
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_ai.settings')
django.setup()

from api.trading_engine import TradingEngine
from api.models import User

# Get the main user
user = User.objects.get(id=2)
print(f'Trading as: {user.email}')

# Start the trading engine
engine = TradingEngine(user)
engine.run_trading_cycle()
"
