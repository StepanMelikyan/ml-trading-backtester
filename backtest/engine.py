# backtest/engine.py
"""
Движок для бэктестинга торговых стратегий.
Поддерживает мульти-инструменты, различные типы ордеров и риск-менеджмент.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import BACKTEST_CONFIG
from utils.logger import log
from utils.helpers import safe_divide
from .transaction_costs import TransactionCosts
from .metrics import TradingMetrics


@dataclass
class Trade:
    """Класс для хранения информации о сделке"""
    entry_time: datetime
    entry_price: float
    position: str  # 'long' or 'short'
    size: float
    symbol: str = "XAUUSD"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    fees: float = 0.0

    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'symbol': self.symbol,
            'position': self.position,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason,
            'fees': self.fees
        }


class BacktestEngine:
    """
    Движок для бэктестинга торговых стратегий.
    """

    def __init__(self,
                 initial_capital: float = None,
                 commission: float = None,
                 slippage: float = None,
                 risk_per_trade: float = None,
                 leverage: float = 1.0,
                 symbol: str = "XAUUSD"):
        """
        Инициализация движка бэктестинга.

        Args:
            initial_capital: начальный капитал
            commission: комиссия в % от объема
            slippage: проскальзывание в %
            risk_per_trade: риск на сделку в % от капитала
            leverage: кредитное плечо
            symbol: торговый инструмент
        """
        self.initial_capital = initial_capital or BACKTEST_CONFIG['initial_capital']
        self.commission = commission or BACKTEST_CONFIG.get('commission', 0.001)
        self.slippage = slippage or BACKTEST_CONFIG.get('slippage', 0.0001)
        self.risk_per_trade = risk_per_trade or BACKTEST_CONFIG.get('risk_per_trade', 0.02)
        self.leverage = leverage
        self.symbol = symbol

        self.current_capital = self.initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [self.initial_capital]
        self.current_position: Optional[str] = None
        self.current_trade: Optional[Trade] = None

        self.transaction_costs = TransactionCosts(symbol)

        self.metrics_calculator = TradingMetrics()

    def calculate_position_size(self, price: float, stop_loss: Optional[float] = None) -> float:
        """
        Расчет размера позиции на основе риск-менеджмента.

        Args:
            price: текущая цена
            stop_loss: уровень стоп-лосса

        Returns:
            Размер позиции
        """
        risk_amount = self.current_capital * self.risk_per_trade

        if stop_loss is None:
            # Если стоп не указан, используем фиксированный процент
            return (risk_amount / price) * self.leverage

        stop_distance = abs(price - stop_loss)

        if stop_distance == 0:
            return 0

        position_size = (risk_amount / stop_distance) * self.leverage

        # Проверяем, что не превышаем капитал с учетом плеча
        max_size = (self.current_capital * self.leverage) / price
        position_size = min(position_size, max_size)

        return position_size

    def execute_order(self, price: float, time: datetime,
                      order_type: str, stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None) -> bool:
        """
        Исполнение ордера.

        Args:
            price: цена исполнения
            time: время исполнения
            order_type: тип ордера ('buy' или 'sell')
            stop_loss: уровень стоп-лосса
            take_profit: уровень тейк-профита

        Returns:
            True если ордер исполнен
        """
        if self.current_position is not None:
            log.warning(f"⚠️ Позиция уже открыта: {self.current_position}")
            return False

        # Учитываем проскальзывание
        if order_type == 'buy':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)

        # Расчет размера позиции
        size = self.calculate_position_size(execution_price, stop_loss)

        # Расчет стоимости с учетом комиссии
        cost = size * execution_price * (1 + self.commission)

        if cost > self.current_capital:
            log.warning(f"⚠️ Недостаточно средств: {cost:.2f} > {self.current_capital:.2f}")
            return False

        self.current_capital -= cost

        self.current_trade = Trade(
            entry_time=time,
            entry_price=execution_price,
            position=order_type,
            size=size,
            symbol=self.symbol,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.current_position = order_type

        log.info(f"🟢 Открыта позиция: {order_type.upper()} {size:.4f} @ {execution_price:.2f}")

        return True

    def buy(self, price: float, time: datetime,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None) -> bool:
        """Покупка (лонг)"""
        return self.execute_order(price, time, 'buy', stop_loss, take_profit)

    def sell(self, price: float, time: datetime,
             stop_loss: Optional[float] = None,
             take_profit: Optional[float] = None) -> bool:
        """Продажа (шорт)"""
        return self.execute_order(price, time, 'sell', stop_loss, take_profit)

    def close_position(self, price: float, time: datetime, reason: str = "signal") -> bool:
        """
        Закрытие текущей позиции.

        Args:
            price: цена закрытия
            time: время закрытия
            reason: причина закрытия

        Returns:
            True если позиция закрыта
        """
        if self.current_position is None or self.current_trade is None:
            return False

        trade = self.current_trade

        # Учитываем проскальзывание
        if self.current_position == 'buy':
            execution_price = price * (1 - self.slippage)
            pnl = (execution_price - trade.entry_price) * trade.size
        else:  # sell
            execution_price = price * (1 + self.slippage)
            pnl = (trade.entry_price - execution_price) * trade.size

        # Комиссия при закрытии
        close_fee = trade.size * execution_price * self.commission
        pnl -= close_fee

        # Обновляем капитал
        self.current_capital += trade.size * execution_price + pnl

        # Сохраняем сделку
        trade.exit_time = time
        trade.exit_price = execution_price
        trade.pnl = pnl
        trade.pnl_pct = (pnl / (trade.entry_price * trade.size)) * 100 if trade.entry_price * trade.size > 0 else 0
        trade.exit_reason = reason
        trade.fees = trade.size * trade.entry_price * self.commission + close_fee

        self.trades.append(trade)
        self.equity_curve.append(self.current_capital)

        log.info(f"🔴 Закрыта позиция: {reason} @ {execution_price:.2f} | P&L: ${pnl:.2f}")

        self.current_position = None
        self.current_trade = None

        return True

    def check_stops(self, high: float, low: float, time: datetime):
        """
        Проверка стоп-лоссов и тейк-профитов.

        Args:
            high: максимум свечи
            low: минимум свечи
            time: время свечи
        """
        if self.current_position is None or self.current_trade is None:
            return

        trade = self.current_trade

        if self.current_position == 'buy':
            # Проверка стоп-лосса
            if trade.stop_loss and low <= trade.stop_loss:
                self.close_position(trade.stop_loss, time, "stop_loss")

            # Проверка тейк-профита
            elif trade.take_profit and high >= trade.take_profit:
                self.close_position(trade.take_profit, time, "take_profit")

        else:  # sell
            # Проверка стоп-лосса
            if trade.stop_loss and high >= trade.stop_loss:
                self.close_position(trade.stop_loss, time, "stop_loss")

            # Проверка тейк-профита
            elif trade.take_profit and low <= trade.take_profit:
                self.close_position(trade.take_profit, time, "take_profit")

    def run(self, df: pd.DataFrame, signal_column: str = 'signal',
            use_atr_stops: bool = True, atr_multiplier: float = 2.0) -> Dict:
        """
        Запуск бэктестинга на исторических данных.

        Args:
            df: DataFrame с ценами и сигналами
            signal_column: колонка с сигналами (1=buy, -1=sell, 0=close)
            use_atr_stops: использовать ли ATR для стопов
            atr_multiplier: множитель ATR для стопов

        Returns:
            Словарь с результатами
        """
        log.info(f"🚀 Запуск бэктестинга для {self.symbol}")
        log.info(f"  Начальный капитал: ${self.initial_capital:,.2f}")
        log.info(f"  Период: {df['time'].iloc[0]} - {df['time'].iloc[-1]}")
        log.info(f"  Количество свечей: {len(df)}")

        # Сбрасываем состояние
        self.current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.current_position = None
        self.current_trade = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            signal = row[signal_column] if signal_column in row else 0

            # Проверка стопов
            self.check_stops(row['high'], row['low'], row['time'])

            # Расчет стопов на основе ATR
            stop_loss = None
            take_profit = None

            if use_atr_stops and 'ATR_14' in df.columns:
                atr = row['ATR_14']

                if signal == 1:  # buy
                    stop_loss = row['close'] - atr_multiplier * atr
                    take_profit = row['close'] + atr_multiplier * atr * 1.5
                elif signal == -1:  # sell
                    stop_loss = row['close'] + atr_multiplier * atr
                    take_profit = row['close'] - atr_multiplier * atr * 1.5

            # Сигнал на покупку
            if signal == 1 and self.current_position is None:
                self.buy(row['close'], row['time'], stop_loss, take_profit)

            # Сигнал на продажу
            elif signal == -1 and self.current_position is None:
                self.sell(row['close'], row['time'], stop_loss, take_profit)

            # Сигнал на закрытие
            elif signal == 0 and self.current_position is not None:
                self.close_position(row['close'], row['time'], "signal")

        # Закрываем все открытые позиции в конце
        if self.current_position is not None:
            self.close_position(df.iloc[-1]['close'], df.iloc[-1]['time'], "end_of_data")

        # Расчет метрик
        results = self.metrics_calculator.calculate_all(self.trades, self.equity_curve)

        log.info(f"✅ Бэктестинг завершен")
        log.info(f"  Конечный капитал: ${self.current_capital:,.2f}")
        log.info(f"  Прибыль: ${self.current_capital - self.initial_capital:,.2f}")
        log.info(f"  Сделок: {len(self.trades)}")

        return results

    def get_trades_df(self) -> pd.DataFrame:
        """
        Возвращает DataFrame со всеми сделками.

        Returns:
            DataFrame с историей сделок
        """
        if not self.trades:
            return pd.DataFrame()

        trades_dict = [t.to_dict() for t in self.trades]
        return pd.DataFrame(trades_dict)

    def get_summary(self) -> Dict:
        """
        Возвращает краткую сводку по бэктестингу.

        Returns:
            Словарь с основными показателями
        """
        if not self.trades:
            return {}

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': self.current_capital - self.initial_capital,
            'total_return_pct': ((self.current_capital / self.initial_capital) - 1) * 100,
            'num_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t.pnl > 0]) / len(self.trades) * 100,
            'avg_pnl': np.mean([t.pnl for t in self.trades]),
            'max_drawdown': self.metrics_calculator.calculate_max_drawdown(self.equity_curve)
        }