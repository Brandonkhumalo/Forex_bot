#!/bin/bash

cd backend

echo "Starting Django server on port 8000..."
echo "Frontend and API both served from: http://0.0.0.0:8000"

python manage.py runserver 0.0.0.0:8000
