# backtest/risk_metrics.py
"""
Продвинутые метрики риска: VaR, CVaR, Stress Testing, Monte Carlo симуляции.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log
from utils.helpers import safe_divide


class AdvancedRiskMetrics:
    """
    Продвинутые метрики риска для анализа торговых стратегий.
    """

    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95,
                      method: str = 'historical', **kwargs) -> Dict:
        """
        Value at Risk (VaR) - максимальные ожидаемые потери.

        Args:
            returns: список доходностей
            confidence_level: уровень доверия (0.95 = 95%)
            method: метод расчета ('historical', 'parametric', 'monte_carlo')
            **kwargs: дополнительные параметры

        Returns:
            Словарь с VaR метриками
        """
        returns = np.array(returns)

        if len(returns) == 0:
            return {'var': 0, 'cvar': 0}

        result = {}

        if method == 'historical':
            # Исторический метод
            var = np.percentile(returns, (1 - confidence_level) * 100)
            result['var'] = var
            result['cvar'] = np.mean(returns[returns <= var])

        elif method == 'parametric':
            # Параметрический метод (предполагает нормальное распределение)
            mean = np.mean(returns)
            std = np.std(returns)
            var = stats.norm.ppf(1 - confidence_level, mean, std)
            result['var'] = var

            # CVaR для нормального распределения
            from scipy.stats import norm
            z_score = stats.norm.ppf(1 - confidence_level)
            cvar = mean - std * norm.pdf(z_score) / (1 - confidence_level)
            result['cvar'] = cvar

        elif method == 'monte_carlo':
            # Метод Монте-Карло
            n_simulations = kwargs.get('n_simulations', 10000)
            horizon = kwargs.get('horizon', 1)

            # Подгоняем распределение
            mean = np.mean(returns)
            std = np.std(returns)

            # Симулируем
            simulated = np.random.normal(mean, std, (n_simulations, horizon))
            if horizon > 1:
                simulated = np.sum(simulated, axis=1)

            result['var'] = np.percentile(simulated, (1 - confidence_level) * 100)
            result['cvar'] = np.mean(simulated[simulated <= result['var']])

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        # Конвертируем в проценты
        result['var_pct'] = result['var'] * 100
        result['cvar_pct'] = result['cvar'] * 100
        result['confidence_level'] = confidence_level
        result['method'] = method

        return result

    @staticmethod
    def calculate_expected_shortfall(returns: List[float],
                                     confidence_level: float = 0.95) -> float:
        """
        Expected Shortfall (CVaR) - средние потери в хвосте распределения.

        Args:
            returns: список доходностей
            confidence_level: уровень доверия

        Returns:
            Expected Shortfall
        """
        returns = np.array(returns)
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return np.mean(returns[returns <= var])

    @staticmethod
    def calculate_drawdown_metrics(equity_curve: List[float]) -> Dict:
        """
        Детальный анализ просадок.

        Args:
            equity_curve: кривая капитала

        Returns:
            Словарь с метриками просадок
        """
        if len(equity_curve) < 2:
            return {}

        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100

        # Находим все просадки
        drawdowns = []
        in_drawdown = False
        start_idx = 0
        current_max_dd = 0

        for i in range(len(drawdown)):
            if drawdown[i] > 0 and not in_drawdown:
                # Начало просадки
                in_drawdown = True
                start_idx = i
                current_max_dd = drawdown[i]
            elif in_drawdown:
                if drawdown[i] > current_max_dd:
                    current_max_dd = drawdown[i]

                if drawdown[i] == 0:
                    # Конец просадки
                    in_drawdown = False
                    drawdowns.append({
                        'start': start_idx,
                        'end': i,
                        'max_drawdown': current_max_dd,
                        'duration': i - start_idx
                    })

        # Статистика по просадкам
        if drawdowns:
            dd_durations = [d['duration'] for d in drawdowns]
            dd_depths = [d['max_drawdown'] for d in drawdowns]

            stats = {
                'num_drawdowns': len(drawdowns),
                'avg_drawdown': np.mean(dd_depths),
                'max_drawdown': np.max(dd_depths),
                'avg_duration': np.mean(dd_durations),
                'max_duration': np.max(dd_durations),
                'total_duration': sum(dd_durations),
                'drawdowns': drawdowns
            }
        else:
            stats = {
                'num_drawdowns': 0,
                'avg_drawdown': 0,
                'max_drawdown': 0,
                'avg_duration': 0,
                'max_duration': 0,
                'total_duration': 0,
                'drawdowns': []
            }

        # Ulcer Index
        stats['ulcer_index'] = np.sqrt(np.mean(drawdown ** 2))

        # Pain Index
        stats['pain_index'] = np.mean(drawdown)

        return stats

    @staticmethod
    def calculate_risk_adjusted_returns(returns: List[float],
                                        risk_free_rate: float = 0.02) -> Dict:
        """
        Расчет скорректированной на риск доходности.

        Args:
            returns: список доходностей
            risk_free_rate: безрисковая ставка

        Returns:
            Словарь с метриками
        """
        returns = np.array(returns)

        if len(returns) == 0:
            return {}

        # Коэффициент Шарпа
        excess_returns = returns - risk_free_rate / 252
        sharpe = safe_divide(np.mean(excess_returns), np.std(returns)) * np.sqrt(252)

        # Коэффициент Сортино (учитывает только отрицательную волатильность)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino = safe_divide(np.mean(excess_returns), np.std(downside_returns)) * np.sqrt(252)
        else:
            sortino = float('inf')

        # Коэффициент Трейнора
        # Предполагаем бета = 1 для упрощения
        treynor = np.mean(excess_returns) * 252

        # Информационное соотношение
        info_ratio = safe_divide(np.mean(returns), np.std(returns)) * np.sqrt(252)

        # Коэффициент Келли
        win_rate = len(returns[returns > 0]) / len(returns)
        avg_win = np.mean(returns[returns > 0]) if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if len(returns[returns < 0]) > 0 else 0

        if avg_loss > 0:
            kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        else:
            kelly = 1

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'treynor_ratio': treynor,
            'information_ratio': info_ratio,
            'kelly_criterion': kelly
        }

    @staticmethod
    def calculate_tail_risk(returns: List[float]) -> Dict:
        """
        Анализ хвостовых рисков.

        Args:
            returns: список доходностей

        Returns:
            Словарь с метриками хвостового риска
        """
        returns = np.array(returns)

        if len(returns) == 0:
            return {}

        # Эксцесс (островершинность)
        kurtosis = stats.kurtosis(returns)

        # Асимметрия
        skewness = stats.skew(returns)

        # Тест на нормальность (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(returns)

        # Максимальные потери
        max_loss = np.min(returns) * 100
        max_gain = np.max(returns) * 100

        # 99-й процентиль потерь
        var_99 = np.percentile(returns, 1) * 100
        cvar_99 = np.mean(returns[returns <= var_99 / 100]) * 100

        # Tail Ratio (отношение хвостов)
        tail_95 = np.percentile(np.abs(returns), 95)
        tail_99 = np.percentile(np.abs(returns), 99)
        tail_ratio = safe_divide(tail_99, tail_95)

        return {
            'kurtosis': kurtosis,
            'skewness': skewness,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05,
            'max_loss_pct': max_loss,
            'max_gain_pct': max_gain,
            'var_99_pct': var_99,
            'cvar_99_pct': cvar_99,
            'tail_ratio': tail_ratio
        }

    @staticmethod
    def calculate_stress_test(returns: List[float],
                              scenarios: Optional[List[Dict]] = None) -> Dict:
        """
        Стресс-тестирование стратегии.

        Args:
            returns: список доходностей
            scenarios: список сценариев

        Returns:
            Результаты стресс-тестов
        """
        if scenarios is None:
            scenarios = AdvancedRiskMetrics.generate_stress_scenarios()

        results = {}
        cumulative_return = np.sum(returns) * 100

        for scenario in scenarios:
            name = scenario.get('name', 'Unknown')
            shock = scenario.get('shock', 0)
            vol_mult = scenario.get('vol_mult', 1.0)

            # Применяем шок
            shocked_returns = np.array(returns) * (1 + shock)

            # Увеличиваем волатильность
            mean_ret = np.mean(shocked_returns)
            shocked_returns = mean_ret + (shocked_returns - mean_ret) * vol_mult

            # Рассчитываем метрики для сценария
            total_return = np.sum(shocked_returns) * 100
            volatility = np.std(shocked_returns) * np.sqrt(252) * 100
            max_drawdown = AdvancedRiskMetrics._calculate_max_dd_from_returns(shocked_returns)

            results[name] = {
                'shock_pct': shock * 100,
                'vol_mult': vol_mult,
                'total_return_pct': total_return,
                'volatility_pct': volatility,
                'max_drawdown_pct': max_drawdown,
                'change_from_normal': total_return - cumulative_return,
                'survived': total_return > -50  # Выжили, если потеряли меньше 50%
            }

        return results

    @staticmethod
    def _calculate_max_dd_from_returns(returns: List[float]) -> float:
        """Рассчитывает максимальную просадку из доходностей."""
        equity = 100 * np.exp(np.cumsum(returns))
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        return np.max(drawdown)

    @staticmethod
    def generate_stress_scenarios() -> List[Dict]:
        """
        Генерирует стандартные стресс-сценарии.

        Returns:
            Список сценариев
        """
        return [
            {'name': 'Normal', 'shock': 0, 'vol_mult': 1.0},
            {'name': 'Mild Crash', 'shock': -0.2, 'vol_mult': 1.5},  # -20% доходность, +50% волатильность
            {'name': 'Severe Crash', 'shock': -0.5, 'vol_mult': 2.0},  # -50% доходность, x2 волатильность
            {'name': 'Black Monday', 'shock': -0.7, 'vol_mult': 3.0},  # -70% доходность, x3 волатильность
            {'name': 'Financial Crisis', 'shock': -0.8, 'vol_mult': 4.0},  # -80% доходность, x4 волатильность
            {'name': 'Bull Market', 'shock': 0.3, 'vol_mult': 0.8},  # +30% доходность, -20% волатильность
            {'name': 'Low Volatility', 'shock': 0, 'vol_mult': 0.5},  # Волатильность /2
            {'name': 'High Volatility', 'shock': 0, 'vol_mult': 2.0},  # Волатильность x2
        ]

    @staticmethod
    def calculate_beta(returns: List[float], market_returns: List[float]) -> float:
        """
        Рассчитывает бета-коэффициент стратегии.

        Args:
            returns: доходности стратегии
            market_returns: доходности рынка

        Returns:
            Бета-коэффициент
        """
        if len(returns) != len(market_returns) or len(returns) == 0:
            return 0

        returns = np.array(returns)
        market_returns = np.array(market_returns)

        covariance = np.cov(returns, market_returns)[0, 1]
        variance = np.var(market_returns)

        return safe_divide(covariance, variance)

    @staticmethod
    def calculate_correlation(returns: List[float], market_returns: List[float]) -> float:
        """
        Рассчитывает корреляцию стратегии с рынком.

        Args:
            returns: доходности стратегии
            market_returns: доходности рынка

        Returns:
            Коэффициент корреляции
        """
        if len(returns) != len(market_returns) or len(returns) == 0:
            return 0

        return np.corrcoef(returns, market_returns)[0, 1]

    @staticmethod
    def calculate_information_ratio(returns: List[float],
                                    benchmark_returns: List[float]) -> float:
        """
        Информационное соотношение (активная доходность / tracking error).

        Args:
            returns: доходности стратегии
            benchmark_returns: доходности бенчмарка

        Returns:
            Information Ratio
        """
        if len(returns) != len(benchmark_returns) or len(returns) == 0:
            return 0

        active_returns = np.array(returns) - np.array(benchmark_returns)

        return safe_divide(np.mean(active_returns), np.std(active_returns)) * np.sqrt(252)