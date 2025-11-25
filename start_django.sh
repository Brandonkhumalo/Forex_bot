#!/bin/bash
cd backend
echo "Running Django migrations..."
python manage.py migrate --run-syncdb
echo "Starting Django server on port 8000..."
exec python manage.py runserver 0.0.0.0:8000
