import os
import requests
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


class CapitalComAPI:
    def __init__(self):
        self.api_key = os.environ.get('CAPITAL_COM_API_KEY', '')
        self.password = os.environ.get('CAPITAL_COM_PASSWORD', '')
        self.identifier = os.environ.get('CAPITAL_COM_IDENTIFIER', '')
        self.base_url = os.environ.get('CAPITAL_COM_DEMO_URL', 'https://demo-api-capital.backend-capital.com')
        self.session_token = None
        self.cst_token = None
        self.session = requests.Session()
        
        self.pair_mapping = {
            'GBP/USD': 'GBPUSD',
            'EUR/USD': 'EURUSD',
            'USD/JPY': 'USDJPY',
            'AUD/USD': 'AUDUSD',
            'USD/CAD': 'USDCAD',
            'NZD/USD': 'NZDUSD',
            'USD/CHF': 'USDCHF',
            'XAU/USD': 'GOLD',
        }
        
        self.timeframe_mapping = {
            '15m': 'MINUTE_15',
            '30m': 'MINUTE_30',
            '1H': 'HOUR',
            '4H': 'HOUR_4',
            'D': 'DAY',
        }
    
    def authenticate(self) -> bool:
        if not all([self.api_key, self.password, self.identifier]):
            logger.warning("Capital.com API credentials not configured")
            return False
        
        try:
            headers = {
                'X-CAP-API-KEY': self.api_key,
                'Content-Type': 'application/json',
            }
            
            payload = {
                'identifier': self.identifier,
                'password': self.password,
                'encryptedPassword': False,
            }
            
            response = self.session.post(
                f'{self.base_url}/api/v1/session',
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                self.cst_token = response.headers.get('CST')
                self.session_token = response.headers.get('X-SECURITY-TOKEN')
                logger.info("Successfully authenticated with Capital.com")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            'X-CAP-API-KEY': self.api_key,
            'CST': self.cst_token or '',
            'X-SECURITY-TOKEN': self.session_token or '',
            'Content-Type': 'application/json',
        }
    
    def get_account_info(self) -> Optional[Dict]:
        try:
            response = self.session.get(
                f'{self.base_url}/api/v1/accounts',
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('accounts'):
                    account = data['accounts'][0]
                    return {
                        'accountId': account.get('accountId'),
                        'accountName': account.get('accountName'),
                        'balance': account.get('balance', {}).get('balance', 0),
                        'available': account.get('balance', {}).get('available', 0),
                        'deposit': account.get('balance', {}).get('deposit', 0),
                        'profitLoss': account.get('balance', {}).get('profitLoss', 0),
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_prices(self) -> Optional[Dict]:
        try:
            epics = list(self.pair_mapping.values())
            response = self.session.get(
                f'{self.base_url}/api/v1/markets',
                params={'searchTerm': ','.join(epics[:5])},
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                prices = {}
                for market in data.get('markets', []):
                    epic = market.get('epic')
                    for pair, mapped_epic in self.pair_mapping.items():
                        if epic == mapped_epic:
                            prices[pair] = {
                                'bid': market.get('bid', 0),
                                'ask': market.get('offer', 0),
                                'spread': market.get('offer', 0) - market.get('bid', 0),
                                'change': market.get('netChange', 0),
                                'changePercent': market.get('percentageChange', 0),
                            }
                            break
                return prices
            return None
        except Exception as e:
            logger.error(f"Error getting prices: {str(e)}")
            return None
    
    def get_market_info(self, pair: str) -> Optional[Dict]:
        try:
            epic = self.pair_mapping.get(pair, pair)
            response = self.session.get(
                f'{self.base_url}/api/v1/markets/{epic}',
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error getting market info: {str(e)}")
            return None
    
    def get_historical_prices(self, pair: str, timeframe: str, num_candles: int = 200) -> Optional[List[Dict]]:
        try:
            epic = self.pair_mapping.get(pair, pair)
            resolution = self.timeframe_mapping.get(timeframe, 'HOUR')
            
            response = self.session.get(
                f'{self.base_url}/api/v1/prices/{epic}',
                params={
                    'resolution': resolution,
                    'max': num_candles,
                },
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                candles = []
                for price in data.get('prices', []):
                    candles.append({
                        'timestamp': price.get('snapshotTime'),
                        'open': (price.get('openPrice', {}).get('bid', 0) + price.get('openPrice', {}).get('ask', 0)) / 2,
                        'high': (price.get('highPrice', {}).get('bid', 0) + price.get('highPrice', {}).get('ask', 0)) / 2,
                        'low': (price.get('lowPrice', {}).get('bid', 0) + price.get('lowPrice', {}).get('ask', 0)) / 2,
                        'close': (price.get('closePrice', {}).get('bid', 0) + price.get('closePrice', {}).get('ask', 0)) / 2,
                        'volume': price.get('lastTradedVolume', 0),
                    })
                return candles
            return None
        except Exception as e:
            logger.error(f"Error getting historical prices: {str(e)}")
            return None
    
    def open_position(self, pair: str, direction: str, size: float, 
                      stop_loss: float = None, take_profit: float = None) -> Optional[Dict]:
        try:
            epic = self.pair_mapping.get(pair, pair)
            
            payload = {
                'epic': epic,
                'direction': 'BUY' if direction == 'buy' else 'SELL',
                'size': size,
                'guaranteedStop': False,
                'trailingStop': False,
            }
            
            if stop_loss:
                payload['stopLevel'] = stop_loss
            if take_profit:
                payload['profitLevel'] = take_profit
            
            response = self.session.post(
                f'{self.base_url}/api/v1/positions',
                json=payload,
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                return {
                    'dealId': data.get('dealId'),
                    'status': data.get('dealStatus'),
                    'epic': epic,
                    'direction': direction,
                    'size': size,
                }
            else:
                logger.error(f"Failed to open position: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
            return None
    
    def close_position(self, deal_id: str, size: float = None) -> Optional[Dict]:
        try:
            payload = {}
            if size:
                payload['size'] = size
            
            response = self.session.delete(
                f'{self.base_url}/api/v1/positions/{deal_id}',
                json=payload if payload else None,
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to close position: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return None
    
    def get_open_positions(self) -> Optional[List[Dict]]:
        try:
            response = self.session.get(
                f'{self.base_url}/api/v1/positions',
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                positions = []
                for pos in data.get('positions', []):
                    position = pos.get('position', {})
                    market = pos.get('market', {})
                    positions.append({
                        'dealId': position.get('dealId'),
                        'epic': market.get('epic'),
                        'direction': position.get('direction'),
                        'size': position.get('size'),
                        'openLevel': position.get('level'),
                        'currentLevel': market.get('bid') if position.get('direction') == 'SELL' else market.get('offer'),
                        'profitLoss': position.get('upl', 0),
                        'stopLevel': position.get('stopLevel'),
                        'profitLevel': position.get('limitLevel'),
                    })
                return positions
            return None
        except Exception as e:
            logger.error(f"Error getting open positions: {str(e)}")
            return None
    
    def get_economic_calendar(self, currencies: List[str] = None) -> Optional[List[Dict]]:
        try:
            from_date = datetime.utcnow()
            to_date = from_date + timedelta(days=7)
            
            response = self.session.get(
                f'{self.base_url}/api/v1/economicCalendar',
                params={
                    'from': from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'to': to_date.strftime('%Y-%m-%dT%H:%M:%S'),
                },
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                events = []
                for event in data.get('economicEvents', []):
                    currency = event.get('currency', '')
                    if currencies and currency not in currencies:
                        continue
                    events.append({
                        'id': event.get('id'),
                        'currency': currency,
                        'name': event.get('name'),
                        'impact': event.get('importance', 'medium').lower(),
                        'forecast': event.get('forecast', ''),
                        'previous': event.get('previous', ''),
                        'actual': event.get('actual', ''),
                        'scheduledAt': event.get('date'),
                    })
                return events
            return None
        except Exception as e:
            logger.error(f"Error getting economic calendar: {str(e)}")
            return None
