#!/usr/bin/env python
import os
import sys
import subprocess

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_ai.settings')
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError:
        raise ImportError("Django is not installed")
    
    print("Running Django migrations...")
    execute_from_command_line(['manage.py', 'migrate', '--run-syncdb'])
    
    print("Starting Django server on port 8000...")
    execute_from_command_line(['manage.py', 'runserver', '0.0.0.0:8000'])

if __name__ == '__main__':
    main()
