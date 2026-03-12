# tests/test_backtest.py
"""
Тесты для модулей бэктестинга.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.engine import BacktestEngine, Trade
from backtest.metrics import TradingMetrics
from backtest.risk_metrics import AdvancedRiskMetrics
from backtest.transaction_costs import TransactionCosts


@pytest.fixture
def sample_price_data():
    """Создает тестовые данные цен."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    n = len(dates)

    # Генерируем случайные цены с трендом
    np.random.seed(42)
    returns = np.random.randn(n) * 0.001
    price = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'time': dates,
        'open': price * (1 + np.random.randn(n) * 0.0005),
        'high': price * (1 + np.abs(np.random.randn(n) * 0.001)),
        'low': price * (1 - np.abs(np.random.randn(n) * 0.001)),
        'close': price,
        'volume': np.random.randint(100, 1000, n),
        'signal': np.random.choice([-1, 0, 1], n, p=[0.2, 0.6, 0.2])
    })

    # Добавляем ATR для тестов
    df['ATR_14'] = np.random.randn(n) * 0.5 + 1

    return df


class TestBacktestEngine:
    """Тесты для движка бэктестинга."""

    def test_initialization(self):
        """Тест инициализации."""
        engine = BacktestEngine(
            initial_capital=10000,
            commission=0.001,
            slippage=0.0001,
            risk_per_trade=0.02
        )

        assert engine.initial_capital == 10000
        assert engine.current_capital == 10000
        assert engine.commission == 0.001
        assert engine.slippage == 0.0001
        assert engine.risk_per_trade == 0.02
        assert len(engine.trades) == 0

    def test_position_sizing(self):
        """Тест расчета размера позиции."""
        engine = BacktestEngine(initial_capital=10000, risk_per_trade=0.02)

        # Без стоп-лосса
        size = engine.calculate_position_size(price=100, stop_loss=None)
        expected = (10000 * 0.02) / 100  # risk_amount / price
        assert size == pytest.approx(expected, rel=1e-3)

        # Со стоп-лоссом
        size = engine.calculate_position_size(price=100, stop_loss=95)
        expected = (10000 * 0.02) / 5  # risk_amount / stop_distance
        assert size == pytest.approx(expected, rel=1e-3)

    def test_buy_order(self, sample_price_data):
        """Тест ордера на покупку."""
        engine = BacktestEngine(initial_capital=10000)
        row = sample_price_data.iloc[0]

        result = engine.buy(
            price=row['close'],
            time=row['time'],
            stop_loss=95,
            take_profit=105
        )

        assert result is True
        assert engine.current_position == 'buy'
        assert engine.current_trade is not None
        assert engine.current_trade.entry_price > 0
        assert engine.current_capital < 10000

    def test_sell_order(self, sample_price_data):
        """Тест ордера на продажу."""
        engine = BacktestEngine(initial_capital=10000)
        row = sample_price_data.iloc[0]

        result = engine.sell(
            price=row['close'],
            time=row['time'],
            stop_loss=105,
            take_profit=95
        )

        assert result is True
        assert engine.current_position == 'sell'

    def test_close_position(self, sample_price_data):
        """Тест закрытия позиции."""
        engine = BacktestEngine(initial_capital=10000)
        row1 = sample_price_data.iloc[0]
        row2 = sample_price_data.iloc[1]

        # Открываем позицию
        engine.buy(price=row1['close'], time=row1['time'])

        # Закрываем
        result = engine.close_position(price=row2['close'], time=row2['time'])

        assert result is True
        assert engine.current_position is None
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_price == row2['close']

    def test_stop_loss(self, sample_price_data):
        """Тест срабатывания стоп-лосса."""
        engine = BacktestEngine(initial_capital=10000)
        row = sample_price_data.iloc[0]

        # Открываем позицию со стопом
        engine.buy(price=100, time=row['time'], stop_loss=99)

        # Цена пробивает стоп
        engine.check_stops(high=100.5, low=98.5, time=row['time'])

        assert engine.current_position is None
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "stop_loss"
        assert engine.trades[0].exit_price == 99

    def test_take_profit(self, sample_price_data):
        """Тест срабатывания тейк-профита."""
        engine = BacktestEngine(initial_capital=10000)
        row = sample_price_data.iloc[0]

        # Открываем позицию с тейком
        engine.buy(price=100, time=row['time'], take_profit=101)

        # Цена достигает тейка
        engine.check_stops(high=101.5, low=100.5, time=row['time'])

        assert engine.current_position is None
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "take_profit"
        assert engine.trades[0].exit_price == 101

    def test_full_backtest(self, sample_price_data):
        """Тест полного бэктестинга."""
        engine = BacktestEngine(initial_capital=10000)

        results = engine.run(sample_price_data, signal_column='signal')

        assert results is not None
        assert 'total_trades' in results
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results

        # Проверяем, что сделки записались
        assert len(engine.trades) > 0

        # Проверяем кривую капитала
        assert len(engine.equity_curve) > 1
        assert engine.equity_curve[-1] != engine.initial_capital


class TestTradingMetrics:
    """Тесты для метрик производительности."""

    def test_calculate_metrics(self):
        """Тест расчета метрик."""
        # Создаем тестовые сделки
        trades = []
        for i in range(10):
            trade = Trade(
                entry_time=datetime.now(),
                exit_time=datetime.now() + timedelta(hours=1),
                entry_price=100,
                exit_price=100 + (1 if i % 2 == 0 else -1),
                position='buy',
                size=1,
                pnl=1 if i % 2 == 0 else -1,
                pnl_pct=1 if i % 2 == 0 else -1,
                exit_reason='signal'
            )
            trades.append(trade)

        equity_curve = [10000, 10050, 10020, 10080, 10040, 10090]

        metrics = TradingMetrics.calculate_all(trades, equity_curve)

        assert metrics['total_trades'] == 10
        assert metrics['winning_trades'] == 5
        assert metrics['losing_trades'] == 5
        assert metrics['win_rate'] == 50
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

    def test_max_drawdown(self):
        """Тест расчета максимальной просадки."""
        equity = [10000, 10500, 10300, 10800, 10600, 10400, 10200, 11000]

        dd = TradingMetrics.calculate_max_drawdown(equity)

        # Максимальная просадка должна быть положительной
        assert dd >= 0

        # Пик был 10800, минимум 10200 -> просадка ~5.56%
        expected = ((10800 - 10200) / 10800) * 100
        assert dd == pytest.approx(expected, rel=1e-2)


class TestAdvancedRiskMetrics:
    """Тесты для продвинутых метрик риска."""

    def test_var_calculation(self):
        """Тест расчета VaR."""
        returns = np.random.randn(1000) * 0.01

        var = AdvancedRiskMetrics.calculate_var(returns, method='historical')

        assert 'var' in var
        assert 'cvar' in var
        assert var['var'] < 0  # VaR должен быть отрицательным

    def test_drawdown_metrics(self):
        """Тест анализа просадок."""
        # Создаем кривую с несколькими просадками
        equity = [10000]
        for i in range(1, 100):
            if 20 < i < 40:
                # Просадка
                equity.append(equity[-1] * 0.99)
            elif 60 < i < 80:
                # Еще одна просадка
                equity.append(equity[-1] * 0.98)
            else:
                # Рост
                equity.append(equity[-1] * 1.01)

        metrics = AdvancedRiskMetrics.calculate_drawdown_metrics(equity)

        assert 'num_drawdowns' in metrics
        assert metrics['num_drawdowns'] >= 2
        assert 'max_drawdown' in metrics
        assert metrics['max_drawdown'] > 0

    def test_risk_adjusted_returns(self):
        """Тест скорректированной на риск доходности."""
        returns = np.random.randn(1000) * 0.01

        metrics = AdvancedRiskMetrics.calculate_risk_adjusted_returns(returns)

        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'information_ratio' in metrics

    def test_stress_test(self):
        """Тест стресс-тестирования."""
        returns = np.random.randn(500) * 0.01

        results = AdvancedRiskMetrics.calculate_stress_test(returns)

        assert len(results) > 0
        assert 'Normal' in results
        assert 'Severe Crash' in results


class TestTransactionCosts:
    """Тесты для транзакционных издержек."""

    def test_spread(self):
        """Тест спреда."""
        costs = TransactionCosts("XAUUSD")
        spread = costs.get_spread()
        assert spread > 0

        spread_pips = costs.get_spread_pips()
        assert spread_pips > 0

    def test_commission(self):
        """Тест комиссии."""
        costs = TransactionCosts("XAUUSD")
        commission = costs.calculate_commission(volume=1, price=100)
        assert commission > 0

    def test_total_costs(self):
        """Тест полных издержек."""
        costs = TransactionCosts("XAUUSD")

        entry_costs = costs.calculate_total_cost(volume=1, price=100, is_entry=True)
        assert 'total' in entry_costs
        assert entry_costs['total'] > 0

        round_trip = costs.calculate_total_cost(volume=1, price=100, is_entry=False)
        assert round_trip['total'] > entry_costs['total']

    def test_breakeven(self):
        """Тест точки безубыточности."""
        costs = TransactionCosts("XAUUSD")

        be = costs.get_break_even_points(entry_price=100, volume=1)

        assert 'breakeven_price' in be
        assert be['breakeven_price'] > 100  # Для long позиции
        assert 'points_to_breakeven' in be
        assert be['points_to_breakeven'] > 0