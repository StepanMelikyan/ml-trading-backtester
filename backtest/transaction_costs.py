# backtest/transaction_costs.py
"""
Моделирование транзакционных издержек: комиссии, спреды, свопы, проскальзывания.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import SYMBOLS
from utils.logger import log
from utils.helpers import safe_divide


class TransactionCosts:
    """
    Расчет транзакционных издержек для разных инструментов.
    """

    def __init__(self, symbol: str):
        """
        Инициализация калькулятора издержек.

        Args:
            symbol: торговый инструмент
        """
        self.symbol = symbol
        self.symbol_info = SYMBOLS.get(symbol, {})

        # Базовая комиссия (в процентах от объема)
        self.base_commission = 0.001  # 0.1%

        # Спреды по умолчанию
        self.default_spreads = {
            'XAUUSD': 0.3,  # 30 центов
            'Brent': 0.02,  # 2 цента
            'WTI': 0.02,  # 2 цента
            'default': 0.0001  # 1 pip для forex
        }

        # Свопы (плата за перенос позиции на следующий день)
        self.swap_rates = {
            'XAUUSD': {'long': -8.5, 'short': 2.5},  # в пунктах
            'Brent': {'long': -3.2, 'short': 1.8},
            'WTI': {'long': -3.2, 'short': 1.8},
            'default': {'long': -1.5, 'short': 0.5}
        }

        # Минимальная комиссия
        self.min_commission = 1.0  # $1

    def get_spread(self) -> float:
        """
        Возвращает спред для инструмента.

        Returns:
            Спред в пунктах
        """
        return self.default_spreads.get(
            self.symbol,
            self.default_spreads['default']
        )

    def get_spread_pips(self) -> float:
        """
        Возвращает спред в пипсах.

        Returns:
            Спред в пипсах
        """
        spread = self.get_spread()
        pip_size = self.symbol_info.get('pip_size', 0.0001)
        return spread / pip_size

    def get_swap(self, position_type: str = 'long') -> float:
        """
        Возвращает своп для инструмента.

        Args:
            position_type: тип позиции ('long' или 'short')

        Returns:
            Своп в пунктах
        """
        swaps = self.swap_rates.get(
            self.symbol,
            self.swap_rates['default']
        )
        return swaps.get(position_type, 0)

    def calculate_commission(self, volume: float, price: float) -> float:
        """
        Расчет комиссии за сделку.

        Args:
            volume: объем в лотах
            price: цена

        Returns:
            Комиссия в долларах
        """
        commission = volume * price * self.base_commission
        return max(commission, self.min_commission)

    def calculate_spread_cost(self, volume: float) -> float:
        """
        Расчет стоимости спреда.

        Args:
            volume: объем в лотах

        Returns:
            Стоимость спреда в долларах
        """
        spread = self.get_spread()
        pip_size = self.symbol_info.get('pip_size', 0.0001)
        return volume * spread * pip_size

    def calculate_slippage(self, volume: float, price: float,
                           volatility: Optional[float] = None) -> float:
        """
        Расчет проскальзывания.

        Args:
            volume: объем в лотах
            price: цена
            volatility: волатильность (опционально)

        Returns:
            Проскальзывание в долларах
        """
        if volatility is None:
            volatility = 0.001  # 0.1% по умолчанию

        # Чем выше волатильность и объем, тем больше проскальзывание
        slippage_pct = 0.0001 + volatility * 0.1 + (volume / 100) * 0.0001
        return volume * price * slippage_pct

    def calculate_total_cost(self, volume: float, price: float,
                             is_entry: bool = True,
                             volatility: Optional[float] = None) -> Dict:
        """
        Расчет полных издержек для сделки.

        Args:
            volume: объем в лотах
            price: цена
            is_entry: True для входа, False для выхода
            volatility: волатильность

        Returns:
            Словарь с издержками
        """
        costs = {
            'commission': self.calculate_commission(volume, price),
            'spread': self.calculate_spread_cost(volume),
            'slippage': 0
        }

        if volatility is not None:
            costs['slippage'] = self.calculate_slippage(volume, price, volatility)

        costs['total'] = sum(costs.values())

        # Для выхода из позиции добавляем те же издержки
        if not is_entry:
            costs['total'] *= 2
            costs['round_trip'] = costs['total']

        # Издержки в процентах
        costs['total_pct'] = (costs['total'] / (volume * price)) * 100

        return costs

    def calculate_swap_cost(self, volume: float, price: float,
                            position_type: str, days_held: int) -> float:
        """
        Расчет свопов за удержание позиции.

        Args:
            volume: объем в лотах
            price: цена
            position_type: тип позиции ('long' или 'short')
            days_held: количество дней удержания

        Returns:
            Стоимость свопов
        """
        swap_rate = self.get_swap(position_type)
        pip_size = self.symbol_info.get('pip_size', 0.0001)
        pip_value = volume * pip_size * price

        return swap_rate * pip_value * days_held

    def get_break_even_points(self, entry_price: float, volume: float,
                              position_type: str = 'long') -> Dict:
        """
        Расчет точек безубыточности с учетом издержек.

        Args:
            entry_price: цена входа
            volume: объем
            position_type: тип позиции

        Returns:
            Информация о точке безубыточности
        """
        costs = self.calculate_total_cost(volume, entry_price, is_entry=True)
        total_cost = costs['total']

        pip_size = self.symbol_info.get('pip_size', 0.0001)
        pip_value = volume * pip_size * entry_price

        points_to_breakeven = total_cost / pip_value if pip_value > 0 else 0

        if position_type == 'long':
            breakeven_price = entry_price + points_to_breakeven * pip_size
        else:
            breakeven_price = entry_price - points_to_breakeven * pip_size

        return {
            'entry_price': entry_price,
            'breakeven_price': breakeven_price,
            'points_to_breakeven': points_to_breakeven,
            'cost_in_points': total_cost / pip_value if pip_value > 0 else 0,
            'cost_in_money': total_cost,
            'cost_as_percent': (total_cost / (volume * entry_price)) * 100
        }

    def calculate_required_move(self, entry_price: float, target_pnl: float,
                                volume: float, position_type: str) -> float:
        """
        Рассчитывает необходимое движение цены для достижения целевой прибыли.

        Args:
            entry_price: цена входа
            target_pnl: целевая прибыль
            volume: объем
            position_type: тип позиции

        Returns:
            Необходимое движение цены
        """
        costs = self.calculate_total_cost(volume, entry_price, is_entry=True)
        total_cost = costs['total']

        pip_size = self.symbol_info.get('pip_size', 0.0001)
        pip_value = volume * pip_size * entry_price

        required_pips = (target_pnl + total_cost) / pip_value if pip_value > 0 else 0

        return required_pips * pip_size

    def get_cost_summary(self) -> Dict:
        """
        Возвращает сводку по издержкам для инструмента.

        Returns:
            Словарь с информацией об издержках
        """
        example_volume = 1.0  # 1 лот
        example_price = 100.0

        entry_costs = self.calculate_total_cost(example_volume, example_price, is_entry=True)
        round_trip_costs = self.calculate_total_cost(example_volume, example_price, is_entry=False)

        return {
            'symbol': self.symbol,
            'spread_pips': self.get_spread_pips(),
            'spread_cost': self.calculate_spread_cost(example_volume),
            'swap_long': self.get_swap('long'),
            'swap_short': self.get_swap('short'),
            'commission_pct': self.base_commission * 100,
            'commission_example': self.calculate_commission(example_volume, example_price),
            'entry_cost_example': entry_costs['total'],
            'round_trip_cost_example': round_trip_costs['round_trip'],
            'cost_as_pct': entry_costs['total_pct'],
            'breakeven': self.get_break_even_points(example_price, example_volume)
        }

    def apply_costs_to_trade(self, trade_data: Dict) -> Dict:
        """
        Применяет издержки к данным сделки.

        Args:
            trade_data: данные сделки (entry_price, exit_price, volume, position_type)

        Returns:
            Обновленные данные сделки с учетом издержек
        """
        entry_price = trade_data.get('entry_price', 0)
        exit_price = trade_data.get('exit_price', 0)
        volume = trade_data.get('volume', 1)
        position_type = trade_data.get('position_type', 'long')

        # Издержки на вход и выход
        entry_costs = self.calculate_total_cost(volume, entry_price, is_entry=True)
        exit_costs = self.calculate_total_cost(volume, exit_price, is_entry=True)

        total_costs = entry_costs['total'] + exit_costs['total']

        # Расчет прибыли без учета издержек
        if position_type == 'long':
            gross_pnl = (exit_price - entry_price) * volume
        else:
            gross_pnl = (entry_price - exit_price) * volume

        net_pnl = gross_pnl - total_costs

        result = trade_data.copy()
        result['gross_pnl'] = gross_pnl
        result['net_pnl'] = net_pnl
        result['total_costs'] = total_costs
        result['costs_pct'] = (total_costs / abs(gross_pnl)) * 100 if gross_pnl != 0 else 0

        return result