#!/bin/bash

cd /home/runner/workspace/backend

python manage.py migrate --run-syncdb

python manage.py run_trading_scheduler &

python manage.py runserver 0.0.0.0:8000
