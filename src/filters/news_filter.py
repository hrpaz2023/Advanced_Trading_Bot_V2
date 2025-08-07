import investpy
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging

class NewsFilter:
    """
    Filtro para evitar operar durante eventos económicos de alto impacto.
    """
    def __init__(self, symbols_to_monitor: list, halt_minutes=5):
        self.logger = logging.getLogger(__name__)
        self.halt_window = timedelta(minutes=halt_minutes)
        self.high_impact_events = self._fetch_high_impact_events(symbols_to_monitor)

    def _get_countries_for_symbols(self, symbols):
        # Mapeo simple de divisa a país para la API de investpy
        currency_country_map = {
            'USD': ['united states'], 'EUR': ['euro zone', 'germany', 'france'],
            'GBP': ['united kingdom'], 'JPY': ['japan'], 'AUD': ['australia'],
            'CAD': ['canada'], 'CHF': ['switzerland'], 'NZD': ['new zealand']
        }
        countries = set()
        for symbol in symbols:
            c1, c2 = symbol[:3], symbol[3:]
            if c1 in currency_country_map: countries.update(currency_country_map[c1])
            if c2 in currency_country_map: countries.update(currency_country_map[c2])
        return list(countries)

    def _fetch_high_impact_events(self, symbols_to_monitor):
        countries = self._get_countries_for_symbols(symbols_to_monitor)
        try:
            df = investpy.news.economic_calendar(
                countries=countries,
                importances=['high']
            )
            # El 'time' viene como texto 'HH:MM', lo convertimos a datetime
            df['datetime_utc'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True).dt.tz_localize('GMT').dt.tz_convert('UTC')
            
            self.logger.info(f"Se encontraron {len(df)} eventos de alto impacto para hoy.")
            return df
        except Exception as e:
            self.logger.error(f"No se pudo obtener el calendario económico: {e}. El filtro de noticias estará inactivo.")
            return pd.DataFrame()

    def is_trading_halted(self, symbol: str) -> tuple[bool, str]:
        """
        Verifica si el trading para un símbolo específico debe detenerse.
        """
        if self.high_impact_events.empty:
            return False, "Filtro de noticias inactivo."

        now_utc = datetime.now(pytz.UTC)
        
        # Obtener las divisas del símbolo
        currencies_in_pair = {symbol[:3], symbol[3:]}

        for index, event in self.high_impact_events.iterrows():
            event_time_utc = event['datetime_utc']
            halt_starts = event_time_utc - self.halt_window
            halt_ends = event_time_utc + self.halt_window

            if halt_starts <= now_utc <= halt_ends:
                # Si la divisa del evento está en nuestro par, detenemos el trading
                if event['currency'].upper() in currencies_in_pair:
                    return True, f"Trading HALTED para {symbol} por noticia: '{event['event']}' a las {event_time_utc.strftime('%H:%M')} UTC."
        
        return False, "No hay noticias de alto impacto en este momento."