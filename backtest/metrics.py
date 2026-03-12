# backtest/metrics.py
"""
Метрики производительности торговых стратегий.
Более 30 метрик для оценки качества стратегии.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log
from utils.helpers import safe_divide


class TradingMetrics:
    """
    Калькулятор метрик производительности торговых стратегий.
    """

    @staticmethod
    def calculate_all(trades: List, equity_curve: List[float]) -> Dict:
        """
        Расчет всех метрик.

        Args:
            trades: список сделок
            equity_curve: кривая капитала

        Returns:
            Словарь со всеми метриками
        """
        metrics = {}

        # Базовые метрики
        metrics.update(TradingMetrics.calculate_basic_metrics(trades, equity_curve))

        # Метрики сделок
        metrics.update(TradingMetrics.calculate_trade_metrics(trades))

        # Метрики риска
        metrics.update(TradingMetrics.calculate_risk_metrics(equity_curve))

        # Статистические метрики
        metrics.update(TradingMetrics.calculate_statistical_metrics(equity_curve, trades))

        return metrics

    @staticmethod
    def calculate_basic_metrics(trades: List, equity_curve: List[float]) -> Dict:
        """
        Базовые метрики.

        Args:
            trades: список сделок
            equity_curve: кривая капитала

        Returns:
            Словарь с базовыми метриками
        """
        if not trades or len(equity_curve) < 2:
            return {}

        initial_capital = equity_curve[0]
        final_capital = equity_curve[-1]

        metrics = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': final_capital - initial_capital,
            'total_return_pct': ((final_capital / initial_capital) - 1) * 100,
            'num_trades': len(trades),
        }

        return metrics

    @staticmethod
    def calculate_trade_metrics(trades: List) -> Dict:
        """
        Метрики по сделкам.

        Args:
            trades: список сделок

        Returns:
            Словарь с метриками сделок
        """
        if not trades:
            return {}

        profits = [t.pnl for t in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]

        metrics = {
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': abs(np.mean(losing_trades)) if losing_trades else 0,
            'largest_win': max(profits) if profits else 0,
            'largest_loss': min(profits) if profits else 0,
            'profit_factor': safe_divide(sum(winning_trades), abs(sum(losing_trades))),
            'expectancy': np.mean(profits),
            'avg_bars_held': np.mean([(t.exit_time - t.entry_time).total_seconds() / 60
                                      for t in trades if t.exit_time]) if trades else 0,
        }

        # Серии побед/поражений
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for p in profits:
            if p > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        metrics['max_consecutive_wins'] = max_consecutive_wins
        metrics['max_consecutive_losses'] = max_consecutive_losses

        return metrics

    @staticmethod
    def calculate_risk_metrics(equity_curve: List[float], risk_free_rate: float = 0.02) -> Dict:
        """
        Метрики риска.

        Args:
            equity_curve: кривая капитала
            risk_free_rate: безрисковая ставка

        Returns:
            Словарь с метриками риска
        """
        if len(equity_curve) < 2:
            return {}

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        metrics = {}

        # Максимальная просадка
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        metrics['max_drawdown'] = np.max(drawdown)
        metrics['max_drawdown_pct'] = metrics['max_drawdown']

        # Длительность просадки
        drawdown_start = 0
        max_duration = 0
        current_duration = 0

        for i, dd in enumerate(drawdown):
            if dd > 0:
                if current_duration == 0:
                    drawdown_start = i
                current_duration += 1
            else:
                if current_duration > 0:
                    max_duration = max(max_duration, current_duration)
                    current_duration = 0

        metrics['max_drawdown_duration'] = max_duration

        # Коэффициент Шарпа
        if len(returns) > 0 and np.std(returns) > 0:
            excess_returns = returns - risk_free_rate / 252
            metrics['sharpe_ratio'] = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
        else:
            metrics['sharpe_ratio'] = 0

        # Коэффициент Сортино (учитывает только отрицательную волатильность)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            metrics['sortino_ratio'] = np.sqrt(252) * np.mean(returns) / np.std(downside_returns)
        else:
            metrics['sortino_ratio'] = 0

        # Коэффициент Калмара
        total_return_pct = ((equity[-1] / equity[0]) - 1) * 100
        metrics['calmar_ratio'] = safe_divide(total_return_pct, metrics['max_drawdown'])

        # Волатильность
        metrics['volatility'] = np.std(returns) * np.sqrt(252) * 100

        # VaR (Value at Risk)
        metrics['var_95'] = np.percentile(returns, 5) * 100
        metrics['cvar_95'] = np.mean(returns[returns <= np.percentile(returns, 5)]) * 100

        return metrics

    @staticmethod
    def calculate_statistical_metrics(equity_curve: List[float], trades: List) -> Dict:
        """
        Статистические метрики.

        Args:
            equity_curve: кривая капитала
            trades: список сделок

        Returns:
            Словарь со статистическими метриками
        """
        metrics = {}

        if len(equity_curve) < 2:
            return metrics

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Асимметрия и эксцесс
        if len(returns) > 1:
            metrics['skewness'] = stats.skew(returns)
            metrics['kurtosis'] = stats.kurtosis(returns)

            # Тест на нормальность
            if len(returns) > 3:
                stat, p_value = stats.normaltest(returns)
                metrics['normality_pvalue'] = p_value

        # Метрики восстановления
        if metrics.get('max_drawdown', 0) > 0:
            metrics['recovery_factor'] = safe_divide(
                equity[-1] - equity[0],
                equity[0] * metrics['max_drawdown'] / 100
            )

        # Месячная доходность
        if len(equity_curve) > 30:
            monthly_returns = []
            for i in range(0, len(equity_curve) - 21, 21):
                monthly_return = (equity_curve[i + 21] - equity_curve[i]) / equity_curve[i] * 100
                monthly_returns.append(monthly_return)

            if monthly_returns:
                metrics['monthly_avg_return'] = np.mean(monthly_returns)
                metrics['monthly_std_return'] = np.std(monthly_returns)
                metrics['monthly_sharpe'] = safe_divide(np.mean(monthly_returns), np.std(monthly_returns)) * np.sqrt(12)

        return metrics

    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """
        Расчет максимальной просадки.

        Args:
            equity_curve: кривая капитала

        Returns:
            Максимальная просадка в %
        """
        if len(equity_curve) < 2:
            return 0

        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100

        return np.max(drawdown)

    @staticmethod
    def print_report(metrics: Dict, symbol: str):
        """
        Красивый вывод отчета.

        Args:
            metrics: словарь с метриками
            symbol: торговый инструмент
        """
        print(f"\n{'=' * 60}")
        print(f"📊 ТОРГОВЫЙ ОТЧЕТ: {symbol}")
        print(f"{'=' * 60}")

        print(f"\n💰 КАПИТАЛ:")
        print(f"  Начальный:  ${metrics.get('initial_capital', 0):,.2f}")
        print(f"  Конечный:   ${metrics.get('final_capital', 0):,.2f}")
        print(f"  Прибыль:    ${metrics.get('total_return', 0):,.2f} ({metrics.get('total_return_pct', 0):+.2f}%)")

        print(f"\n📈 СТАТИСТИКА СДЕЛОК:")
        print(f"  Всего сделок:     {metrics.get('num_trades', 0)}")
        print(f"  Прибыльных:       {metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0):.1f}%)")
        print(f"  Убыточных:        {metrics.get('losing_trades', 0)}")
        print(f"  Профит-фактор:    {metrics.get('profit_factor', 0):.2f}")
        print(f"  Средняя прибыль:  ${metrics.get('avg_win', 0):.2f}")
        print(f"  Средний убыток:   ${metrics.get('avg_loss', 0):.2f}")
        print(f"  Макс. прибыль:    ${metrics.get('largest_win', 0):.2f}")
        print(f"  Макс. убыток:     ${metrics.get('largest_loss', 0):.2f}")
        print(f"  Ожидание сделки:  ${metrics.get('expectancy', 0):.2f}")
        print(f"  Серия побед:      {metrics.get('max_consecutive_wins', 0)}")
        print(f"  Серия поражений:  {metrics.get('max_consecutive_losses', 0)}")

        print(f"\n📉 МЕТРИКИ РИСКА:")
        print(f"  Макс. просадка:      {metrics.get('max_drawdown', 0):.2f}%")
        print(f"  Длительность:        {metrics.get('max_drawdown_duration', 0)} дней")
        print(f"  Коэф. Шарпа:         {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Коэф. Сортино:       {metrics.get('sortino_ratio', 0):.2f}")
        print(f"  Коэф. Калмара:       {metrics.get('calmar_ratio', 0):.2f}")
        print(f"  Волатильность:       {metrics.get('volatility', 0):.2f}%")
        print(f"  VaR (95%):           {metrics.get('var_95', 0):.2f}%")
        print(f"  CVaR (95%):          {metrics.get('cvar_95', 0):.2f}%")

        print(f"\n📊 СТАТИСТИКА:")
        print(f"  Асимметрия:          {metrics.get('skewness', 0):.2f}")
        print(f"  Эксцесс:             {metrics.get('kurtosis', 0):.2f}")
        print(f"  Recovery Factor:     {metrics.get('recovery_factor', 0):.2f}")

        if 'monthly_avg_return' in metrics:
            print(f"\n📆 МЕСЯЧНАЯ СТАТИСТИКА:")
            print(f"  Средняя доходность:  {metrics['monthly_avg_return']:.2f}%")
            print(f"  Волатильность:       {metrics['monthly_std_return']:.2f}%")
            print(f"  Шарп (мес):          {metrics['monthly_sharpe']:.2f}")

        print(f"{'=' * 60}")