from django.apps import AppConfig
import threading
import os


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    scheduler_started = False
    
    def ready(self):
        if os.environ.get('RUN_MAIN') == 'true' and not ApiConfig.scheduler_started:
            ApiConfig.scheduler_started = True
            from .scheduler import start_scheduler
            thread = threading.Thread(target=start_scheduler, daemon=True)
            thread.start()
