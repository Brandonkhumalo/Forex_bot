import logging
from django.core.management.base import BaseCommand
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from api.models import User, TradingSettings
from api.trading_engine import TradingEngine

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run the 24/7 trading scheduler'
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting trading scheduler...'))
        
        scheduler = BlockingScheduler()
        
        scheduler.add_job(
            self.run_trading_cycle,
            IntervalTrigger(minutes=1),
            id='trading_cycle',
            name='Run trading cycle for all active users',
            replace_existing=True,
        )
        
        scheduler.add_job(
            self.update_economic_events,
            IntervalTrigger(hours=1),
            id='economic_events',
            name='Update economic calendar',
            replace_existing=True,
        )
        
        try:
            self.stdout.write(self.style.SUCCESS('Scheduler started. Press Ctrl+C to exit.'))
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.stdout.write(self.style.WARNING('Scheduler stopped.'))
    
    def run_trading_cycle(self):
        active_settings = TradingSettings.objects.filter(ai_enabled=True)
        
        for settings in active_settings:
            try:
                engine = TradingEngine(settings.user)
                engine.run_trading_cycle()
            except Exception as e:
                logger.error(f"Error in trading cycle for {settings.user.email}: {str(e)}")
    
    def update_economic_events(self):
        from api.capital_api import CapitalComAPI
        from api.models import EconomicEvent
        
        api = CapitalComAPI()
        if api.authenticate():
            events = api.get_economic_calendar(currencies=['USD', 'GBP'])
            if events:
                for event_data in events:
                    EconomicEvent.objects.update_or_create(
                        currency=event_data['currency'],
                        event_name=event_data['name'],
                        scheduled_at=event_data['scheduledAt'],
                        defaults={
                            'impact': event_data.get('impact', 'medium'),
                            'forecast': event_data.get('forecast', ''),
                            'previous': event_data.get('previous', ''),
                            'actual': event_data.get('actual', ''),
                        }
                    )
                logger.info(f"Updated {len(events)} economic events")
