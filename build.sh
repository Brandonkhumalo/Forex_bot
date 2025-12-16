#!/bin/bash

echo "Building React frontend..."
npm run build

echo "Copying frontend to Django static folder..."
rm -rf backend/static/*
cp -r dist/* backend/static/

echo "Collecting static files for Django..."
cd backend && python manage.py collectstatic --noinput

echo "Build complete! Frontend is now served by Django."
