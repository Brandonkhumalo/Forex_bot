import time
from django.core.management.base import BaseCommand
from api.trading_engine import TradingEngine
from api.models import User


class Command(BaseCommand):
    help = 'Run the AI trading engine continuously'

    def add_arguments(self, parser):
        parser.add_argument(
            '--interval',
            type=int,
            default=300,
            help='Interval between trading cycles in seconds (default: 300)'
        )
        parser.add_argument(
            '--user-id',
            type=int,
            default=2,
            help='User ID to run trading for (default: 2)'
        )

    def handle(self, *args, **options):
        interval = options['interval']
        user_id = options['user_id']
        
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            self.stderr.write(self.style.ERROR(f'User with ID {user_id} not found'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Starting AI Trading Engine for {user.email}'))
        self.stdout.write(f'Trading cycle interval: {interval} seconds')
        
        engine = TradingEngine(user)
        
        while True:
            try:
                self.stdout.write(f'Running trading cycle at {time.strftime("%Y-%m-%d %H:%M:%S")}')
                engine.run_trading_cycle()
                self.stdout.write(self.style.SUCCESS('Trading cycle completed'))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f'Error in trading cycle: {str(e)}'))
            
            self.stdout.write(f'Sleeping for {interval} seconds...')
            time.sleep(interval)
