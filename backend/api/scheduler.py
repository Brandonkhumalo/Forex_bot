import logging
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

def run_trading_cycles():
    from .models import User, TradingSettings
    from .trading_engine import TradingEngine
    
    try:
        active_users = TradingSettings.objects.filter(ai_enabled=True).select_related('user')
        
        for settings in active_users:
            try:
                engine = TradingEngine(settings.user)
                if engine.initialize():
                    result = engine.run_trading_cycle()
                    logger.info(f"Trading cycle for {settings.user.email}: {result.get('trades_executed', 0)} trades")
            except Exception as e:
                logger.error(f"Error in trading cycle for {settings.user.email}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in scheduler: {str(e)}")

def start_scheduler():
    logger.info("Starting trading scheduler...")
    
    time.sleep(5)
    
    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        run_trading_cycles,
        IntervalTrigger(minutes=1),
        id='trading_cycle',
        name='Run trading cycle for all active users',
        replace_existing=True,
    )
    
    scheduler.start()
    logger.info("Trading scheduler started - running every 1 minute")
    
    while True:
        time.sleep(60)
