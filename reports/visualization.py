# reports/visualization.py
"""
Модуль для визуализации результатов торговли и анализа данных.
Создает профессиональные графики для отчетов.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PLOTS_DIR
from utils.logger import log

# Настройка стилей
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TradingVisualizer:
    """
    Класс для создания визуализаций торговых результатов.
    """

    def __init__(self, symbol: str, save_plots: bool = True, plots_dir: Optional[Path] = None):
        """
        Инициализация визуализатора.

        Args:
            symbol: торговый инструмент
            save_plots: сохранять ли графики
            plots_dir: директория для сохранения
        """
        self.symbol = symbol
        self.save_plots = save_plots
        self.plots_dir = plots_dir or PLOTS_DIR
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Цветовая схема
        self.colors = {
            'price': '#2E86AB',
            'buy': '#2ECC71',
            'sell': '#E74C3C',
            'equity': '#9B59B6',
            'drawdown': '#E67E22',
            'volume': '#3498DB'
        }

    def plot_price_with_signals(self, df: pd.DataFrame,
                                predictions: Optional[np.ndarray] = None,
                                trades: Optional[List] = None,
                                title: Optional[str] = None,
                                show: bool = True,
                                save: bool = True) -> plt.Figure:
        """
        График цены с сигналами модели и сделками.

        Args:
            df: DataFrame с ценами
            predictions: предсказания модели
            trades: список сделок
            title: заголовок
            show: показывать ли график
            save: сохранять ли график

        Returns:
            Объект фигуры matplotlib
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10),
                                 gridspec_kw={'height_ratios': [3, 1, 1]})

        # Верхний график - цена и сигналы
        ax1 = axes[0]
        ax1.plot(df['time'], df['close'], label='Close Price',
                 color=self.colors['price'], linewidth=1.5, alpha=0.7)

        # Добавляем скользящие средние
        if 'SMA_20' in df.columns:
            ax1.plot(df['time'], df['SMA_20'], label='SMA 20',
                     color='orange', linewidth=1, alpha=0.7, linestyle='--')
        if 'SMA_50' in df.columns:
            ax1.plot(df['time'], df['SMA_50'], label='SMA 50',
                     color='red', linewidth=1, alpha=0.7, linestyle='--')

        # Сигналы модели
        if predictions is not None:
            buy_signals = df[predictions == 1]
            sell_signals = df[predictions == 0]

            ax1.scatter(buy_signals['time'], buy_signals['close'],
                        color=self.colors['buy'], marker='^', s=100,
                        label='Buy Signal', zorder=5, alpha=0.8)
            ax1.scatter(sell_signals['time'], sell_signals['close'],
                        color=self.colors['sell'], marker='v', s=100,
                        label='Sell Signal', zorder=5, alpha=0.8)

        # Отмечаем сделки
        if trades:
            for trade in trades[-50:]:  # Последние 50 сделок
                if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                    color = self.colors['buy'] if trade.pnl > 0 else self.colors['sell']
                    ax1.axvspan(trade.entry_time, trade.exit_time,
                                alpha=0.1, color=color)

        ax1.set_title(title or f'{self.symbol} - Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Средний график - RSI
        ax2 = axes[1]
        if 'RSI_14' in df.columns:
            ax2.plot(df['time'], df['RSI_14'], color='purple', linewidth=1)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

        # Нижний график - Объем
        ax3 = axes[2]
        if 'volume' in df.columns:
            colors = [self.colors['buy'] if close > open_ else self.colors['sell']
                      for close, open_ in zip(df['close'], df['open'])]
            ax3.bar(df['time'], df['volume'], color=colors, alpha=0.7, width=0.8)
            ax3.set_ylabel('Volume')
            ax3.set_xlabel('Time')
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save and self.save_plots:
            filename = self.plots_dir / f"{self.symbol}_price_signals.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            log.info(f"📊 График сохранен: {filename}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_equity_curve(self, equity_curve: List[float],
                          trades: Optional[List] = None,
                          title: Optional[str] = None,
                          show: bool = True,
                          save: bool = True) -> plt.Figure:
        """
        График кривой капитала.

        Args:
            equity_curve: кривая капитала
            trades: список сделок
            title: заголовок
            show: показывать ли график
            save: сохранять ли график

        Returns:
            Объект фигуры matplotlib
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 8),
                                 gridspec_kw={'height_ratios': [2, 1]})

        # Кривая капитала
        ax1 = axes[0]
        x = range(len(equity_curve))
        ax1.plot(x, equity_curve, color=self.colors['equity'],
                 linewidth=2, label='Equity Curve')

        # Начальный капитал
        ax1.axhline(y=equity_curve[0], color='gray', linestyle='--',
                    alpha=0.5, label=f'Initial: ${equity_curve[0]:,.2f}')

        # Отмечаем сделки
        if trades:
            for i, trade in enumerate(trades):
                if hasattr(trade, 'entry_time'):
                    # Это упрощение - в реальности нужно сопоставить время с индексом
                    color = self.colors['buy'] if trade.pnl > 0 else self.colors['sell']
                    ax1.axvline(x=i, color=color, alpha=0.3, linewidth=0.5)

        ax1.set_title(title or f'{self.symbol} - Equity Curve')
        ax1.set_ylabel('Capital ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Просадка
        ax2 = axes[1]
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100

        ax2.fill_between(x, 0, drawdown, color=self.colors['drawdown'],
                         alpha=0.5, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Trade Number')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save and self.save_plots:
            filename = self.plots_dir / f"{self.symbol}_equity_curve.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            log.info(f"📊 График сохранен: {filename}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_drawdown_analysis(self, equity_curve: List[float],
                               show: bool = True, save: bool = True) -> plt.Figure:
        """
        Детальный анализ просадок.

        Args:
            equity_curve: кривая капитала
            show: показывать ли график
            save: сохранять ли график

        Returns:
            Объект фигуры matplotlib
        """
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100

        # Находим все просадки
        in_drawdown = False
        start_idx = 0
        drawdown_periods = []

        for i in range(len(drawdown)):
            if drawdown[i] > 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif drawdown[i] == 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append((start_idx, i, max(drawdown[start_idx:i])))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Кривая просадки
        ax1 = axes[0, 0]
        ax1.fill_between(range(len(drawdown)), 0, drawdown,
                         color=self.colors['drawdown'], alpha=0.5)
        ax1.set_title('Drawdown Over Time')
        ax1.set_ylabel('Drawdown (%)')
        ax1.set_xlabel('Time')
        ax1.grid(True, alpha=0.3)

        # 2. Распределение просадок
        ax2 = axes[0, 1]
        if drawdown_periods:
            dd_depths = [dd[2] for dd in drawdown_periods]
            ax2.hist(dd_depths, bins=20, color=self.colors['drawdown'],
                     alpha=0.7, edgecolor='black')
            ax2.set_title('Drawdown Distribution')
            ax2.set_xlabel('Drawdown Depth (%)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

        # 3. Длительность просадок
        ax3 = axes[1, 0]
        if drawdown_periods:
            dd_durations = [(dd[1] - dd[0]) for dd in drawdown_periods]
            ax3.bar(range(len(dd_durations)), dd_durations,
                    color=self.colors['drawdown'], alpha=0.7)
            ax3.set_title('Drawdown Duration')
            ax3.set_xlabel('Drawdown Number')
            ax3.set_ylabel('Duration (periods)')
            ax3.grid(True, alpha=0.3)

        # 4. Underwater chart
        ax4 = axes[1, 1]
        underwater = (equity / peak - 1) * 100
        ax4.fill_between(range(len(underwater)), 0, underwater,
                         color='darkred', alpha=0.5)
        ax4.set_title('Underwater Chart')
        ax4.set_ylabel('Return from Peak (%)')
        ax4.set_xlabel('Time')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save and self.save_plots:
            filename = self.plots_dir / f"{self.symbol}_drawdown_analysis.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_monthly_returns(self, df: pd.DataFrame,
                             show: bool = True, save: bool = True) -> plt.Figure:
        """
        Тепловая карта месячных доходностей.

        Args:
            df: DataFrame с колонкой 'close' и 'time'
            show: показывать ли график
            save: сохранять ли график

        Returns:
            Объект фигуры matplotlib
        """
        # Рассчитываем доходности
        df = df.copy()
        df['returns'] = df['close'].pct_change() * 100

        # Добавляем временные признаки
        df['year'] = pd.DatetimeIndex(df['time']).year
        df['month'] = pd.DatetimeIndex(df['time']).month

        # Группируем по месяцам
        monthly_returns = df.groupby(['year', 'month'])['returns'].sum().unstack()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Тепловая карта
        sns.heatmap(monthly_returns, annot=True, fmt='.2f',
                    cmap='RdYlGn', center=0, ax=ax,
                    cbar_kws={'label': 'Return (%)'})

        ax.set_title(f'{self.symbol} - Monthly Returns (%)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')

        plt.tight_layout()

        if save and self.save_plots:
            filename = self.plots_dir / f"{self.symbol}_monthly_returns.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_feature_importance(self, feature_names: List[str],
                                importance_values: np.ndarray,
                                title: Optional[str] = None,
                                top_n: int = 20,
                                show: bool = True,
                                save: bool = True) -> plt.Figure:
        """
        График важности признаков.

        Args:
            feature_names: названия признаков
            importance_values: значения важности
            title: заголовок
            top_n: количество отображаемых признаков
            show: показывать ли график
            save: сохранять ли график

        Returns:
            Объект фигуры matplotlib
        """
        # Сортируем по важности
        indices = np.argsort(importance_values)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

        ax.barh(range(len(indices)), importance_values[indices], color=colors[::-1])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(title or f'{self.symbol} - Top {top_n} Feature Importance')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save and self.save_plots:
            filename = self.plots_dir / f"{self.symbol}_feature_importance.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              show: bool = True, save: bool = True) -> plt.Figure:
        """
        Матрица ошибок классификации.

        Args:
            y_true: истинные значения
            y_pred: предсказанные значения
            show: показывать ли график
            save: сохранять ли график

        Returns:
            Объект фигуры matplotlib
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Sell', 'Buy'],
                    yticklabels=['Sell', 'Buy'])

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{self.symbol} - Confusion Matrix')

        # Добавляем проценты
        total = np.sum(cm)
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'{percentage:.1f}%',
                        ha='center', va='center',
                        color='white' if cm[i, j] > total / 2 else 'black')

        plt.tight_layout()

        if save and self.save_plots:
            filename = self.plots_dir / f"{self.symbol}_confusion_matrix.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_trades_distribution(self, trades: List,
                                 show: bool = True, save: bool = True) -> plt.Figure:
        """
        Распределение прибыли по сделкам.

        Args:
            trades: список сделок
            show: показывать ли график
            save: сохранять ли график

        Returns:
            Объект фигуры matplotlib
        """
        if not trades:
            return None

        pnls = [t.pnl for t in trades]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Гистограмма прибыли
        ax1 = axes[0, 0]
        ax1.hist(pnls, bins=30, color=self.colors['equity'],
                 alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.axvline(x=np.mean(pnls), color='green', linestyle='--',
                    linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')
        ax1.set_title('PnL Distribution')
        ax1.set_xlabel('PnL ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. PnL по порядку
        ax2 = axes[0, 1]
        ax2.plot(range(len(pnls)), pnls, 'o-', color=self.colors['price'],
                 alpha=0.7, markersize=4)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('PnL by Trade')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('PnL ($)')
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative PnL
        ax3 = axes[1, 0]
        cumulative = np.cumsum(pnls)
        ax3.fill_between(range(len(cumulative)), 0, cumulative,
                         color=self.colors['buy'] if cumulative[-1] > 0 else self.colors['sell'],
                         alpha=0.5)
        ax3.plot(range(len(cumulative)), cumulative, color='black', linewidth=1)
        ax3.set_title('Cumulative PnL')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Cumulative PnL ($)')
        ax3.grid(True, alpha=0.3)

        # 4. Win/Loss ratio
        ax4 = axes[1, 1]
        wins = len([p for p in pnls if p > 0])
        losses = len([p for p in pnls if p <= 0])
        ax4.pie([wins, losses], labels=[f'Wins ({wins})', f'Losses ({losses})'],
                colors=[self.colors['buy'], self.colors['sell']],
                autopct='%1.1f%%', startangle=90)
        ax4.set_title('Win/Loss Ratio')

        plt.tight_layout()

        if save and self.save_plots:
            filename = self.plots_dir / f"{self.symbol}_trades_distribution.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def create_dashboard(self, df: pd.DataFrame, trades: List,
                         equity_curve: List[float], metrics: Dict) -> plt.Figure:
        """
        Создает дашборд с основными графиками.

        Args:
            df: DataFrame с данными
            trades: список сделок
            equity_curve: кривая капитала
            metrics: метрики производительности

        Returns:
            Объект фигуры matplotlib
        """
        fig = plt.figure(figsize=(20, 12))

        # Создаем сетку для графиков
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Цена и сигналы (верхний, на всю ширину)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['time'], df['close'], color=self.colors['price'], linewidth=1.5)
        ax1.set_title(f'{self.symbol} - Price Chart')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)

        # 2. Кривая капитала
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(equity_curve, color=self.colors['equity'], linewidth=2)
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Capital ($)')
        ax2.grid(True, alpha=0.3)

        # 3. Распределение PnL
        ax3 = fig.add_subplot(gs[1, 1])
        if trades:
            pnls = [t.pnl for t in trades]
            ax3.hist(pnls, bins=20, color=self.colors['buy'], alpha=0.7)
            ax3.set_title('PnL Distribution')
            ax3.set_xlabel('PnL ($)')
            ax3.grid(True, alpha=0.3)

        # 4. Месячные доходности (упрощенно)
        ax4 = fig.add_subplot(gs[1, 2])
        if 'RSI_14' in df.columns:
            ax4.plot(df['time'], df['RSI_14'], color='purple')
            ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax4.set_title('RSI')
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)

        # 5-9. Метрики в текстовом виде
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        metrics_text = f"""
        Total Return: ${metrics.get('total_return', 0):,.2f}
        Return %: {metrics.get('total_return_pct', 0):.2f}%
        Total Trades: {metrics.get('num_trades', 0)}
        Win Rate: {metrics.get('win_rate', 0):.1f}%
        Profit Factor: {metrics.get('profit_factor', 0):.2f}
        """
        ax5.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        ax5.set_title('Performance Metrics')

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        risk_text = f"""
        Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
        Volatility: {metrics.get('volatility', 0):.2f}%
        VaR (95%): {metrics.get('var_95', 0):.2f}%
        """
        ax6.text(0.1, 0.5, risk_text, fontsize=12, verticalalignment='center')
        ax6.set_title('Risk Metrics')

        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        trade_text = f"""
        Avg Win: ${metrics.get('avg_win', 0):.2f}
        Avg Loss: ${metrics.get('avg_loss', 0):.2f}
        Largest Win: ${metrics.get('largest_win', 0):.2f}
        Largest Loss: ${metrics.get('largest_loss', 0):.2f}
        Expectancy: ${metrics.get('expectancy', 0):.2f}
        """
        ax7.text(0.1, 0.5, trade_text, fontsize=12, verticalalignment='center')
        ax7.set_title('Trade Statistics')

        plt.suptitle(f'{self.symbol} - Trading Dashboard', fontsize=16, y=0.98)

        if self.save_plots:
            filename = self.plots_dir / f"{self.symbol}_dashboard.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            log.info(f"📊 Дашборд сохранен: {filename}")

        plt.show()

        return fig