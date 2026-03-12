# reports/html_report.py
"""
Генератор интерактивных HTML отчетов с графиками и метриками.
"""

import pandas as pd
import numpy as np
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple  # ← ДОБАВЛЕНО!
import matplotlib.pyplot as plt
import jinja2
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import REPORTS_DIR
from utils.logger import log
from .visualization import TradingVisualizer


class HTMLReport:
    """
    Генератор профессиональных HTML отчетов.
    """

    def __init__(self, symbol: str, output_dir: Optional[Path] = None):
        """
        Инициализация генератора отчетов.

        Args:
            symbol: торговый инструмент
            output_dir: директория для сохранения отчетов
        """
        self.symbol = symbol
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.viz = TradingVisualizer(symbol, save_plots=False)
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)

        # Создаем шаблон, если его нет
        self._ensure_template()

    def _ensure_template(self):
        """Создает HTML шаблон, если он не существует."""
        template_path = self.template_dir / "report_template.html"

        if not template_path.exists():
            template_content = self._get_default_template()
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)

    def _get_default_template(self) -> str:
        """Возвращает HTML шаблон по умолчанию."""
        return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Торговый отчет - {{ symbol }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .content { padding: 40px; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        .metric-card:hover { transform: translateY(-5px); }
        .metric-title { 
            color: #667eea; 
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            font-weight: 600;
        }
        .metric-value { 
            font-size: 2.5em; 
            font-weight: bold; 
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .section-title {
            font-size: 2em;
            margin: 40px 0 20px;
            color: #333;
            border-left: 5px solid #667eea;
            padding-left: 20px;
        }
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .chart-container h3 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .chart-img {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            font-weight: 600;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }
        tr:hover { background-color: #f9f9ff; }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .badge-success { background: #10b981; color: white; }
        .badge-danger { background: #ef4444; color: white; }
        .badge-warning { background: #f59e0b; color: white; }
        .footer {
            text-align: center;
            padding: 30px;
            background: #f9f9ff;
            color: #666;
            font-size: 0.9em;
        }
        @media (max-width: 768px) {
            .header { padding: 20px; }
            .content { padding: 20px; }
            .metric-value { font-size: 2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Торговый отчет: {{ symbol }}</h1>
            <p>Сгенерирован: {{ date }}</p>
        </div>

        <div class="content">
            <!-- Ключевые метрики -->
            <h2 class="section-title">📈 Ключевые метрики</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Начальный капитал</div>
                    <div class="metric-value">${{ "%.2f"|format(metrics.initial_capital) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Конечный капитал</div>
                    <div class="metric-value">${{ "%.2f"|format(metrics.final_capital) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Общая прибыль</div>
                    <div class="metric-value {{ 'positive' if metrics.total_return > 0 else 'negative' }}">
                        ${{ "%.2f"|format(metrics.total_return) }}
                    </div>
                    <div>{{ "%.2f"|format(metrics.total_return_pct) }}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Коэф. Шарпа</div>
                    <div class="metric-value">{{ "%.2f"|format(metrics.sharpe_ratio) }}</div>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Всего сделок</div>
                    <div class="metric-value">{{ metrics.num_trades }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Винрейт</div>
                    <div class="metric-value">{{ "%.1f"|format(metrics.win_rate) }}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Профит-фактор</div>
                    <div class="metric-value">{{ "%.2f"|format(metrics.profit_factor) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Макс. просадка</div>
                    <div class="metric-value negative">{{ "%.2f"|format(metrics.max_drawdown) }}%</div>
                </div>
            </div>

            <!-- Графики -->
            <h2 class="section-title">📊 Графики</h2>

            {% if equity_curve_img %}
            <div class="chart-container">
                <h3>Кривая капитала</h3>
                <img class="chart-img" src="data:image/png;base64,{{ equity_curve_img }}" alt="Equity Curve">
            </div>
            {% endif %}

            {% if price_signals_img %}
            <div class="chart-container">
                <h3>Цена и сигналы</h3>
                <img class="chart-img" src="data:image/png;base64,{{ price_signals_img }}" alt="Price and Signals">
            </div>
            {% endif %}

            {% if drawdown_img %}
            <div class="chart-container">
                <h3>Анализ просадок</h3>
                <img class="chart-img" src="data:image/png;base64,{{ drawdown_img }}" alt="Drawdown Analysis">
            </div>
            {% endif %}

            {% if feature_importance_img %}
            <div class="chart-container">
                <h3>Важность признаков</h3>
                <img class="chart-img" src="data:image/png;base64,{{ feature_importance_img }}" alt="Feature Importance">
            </div>
            {% endif %}

            <!-- Детальная статистика -->
            <h2 class="section-title">📋 Детальная статистика</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Статистика сделок</div>
                    <table style="width: 100%; margin-top: 10px;">
                        <tr><td>Прибыльных:</td><td><strong>{{ metrics.winning_trades }}</strong></td></tr>
                        <tr><td>Убыточных:</td><td><strong>{{ metrics.losing_trades }}</strong></td></tr>
                        <tr><td>Средняя прибыль:</td><td><strong>${{ "%.2f"|format(metrics.avg_win) }}</strong></td></tr>
                        <tr><td>Средний убыток:</td><td><strong>${{ "%.2f"|format(metrics.avg_loss) }}</strong></td></tr>
                        <tr><td>Макс. прибыль:</td><td><strong>${{ "%.2f"|format(metrics.largest_win) }}</strong></td></tr>
                        <tr><td>Макс. убыток:</td><td><strong>${{ "%.2f"|format(metrics.largest_loss) }}</strong></td></tr>
                        <tr><td>Ожидание:</td><td><strong>${{ "%.2f"|format(metrics.expectancy) }}</strong></td></tr>
                    </table>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Риск-метрики</div>
                    <table style="width: 100%; margin-top: 10px;">
                        <tr><td>Волатильность:</td><td><strong>{{ "%.2f"|format(metrics.volatility) }}%</strong></td></tr>
                        <tr><td>Сортино:</td><td><strong>{{ "%.2f"|format(metrics.sortino_ratio) }}</strong></td></tr>
                        <tr><td>Калмар:</td><td><strong>{{ "%.2f"|format(metrics.calmar_ratio) }}</strong></td></tr>
                        <tr><td>VaR (95%):</td><td><strong>{{ "%.2f"|format(metrics.var_95) }}%</strong></td></tr>
                        <tr><td>CVaR (95%):</td><td><strong>{{ "%.2f"|format(metrics.cvar_95) }}%</strong></td></tr>
                        <tr><td>Асимметрия:</td><td><strong>{{ "%.2f"|format(metrics.skewness) }}</strong></td></tr>
                        <tr><td>Эксцесс:</td><td><strong>{{ "%.2f"|format(metrics.kurtosis) }}</strong></td></tr>
                    </table>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Серии</div>
                    <table style="width: 100%; margin-top: 10px;">
                        <tr><td>Макс. побед подряд:</td><td><strong>{{ metrics.max_consecutive_wins }}</strong></td></tr>
                        <tr><td>Макс. поражений подряд:</td><td><strong>{{ metrics.max_consecutive_losses }}</strong></td></tr>
                        <tr><td>Средняя длительность:</td><td><strong>{{ "%.1f"|format(metrics.avg_bars_held) }} мин</strong></td></tr>
                    </table>
                </div>
            </div>

            <!-- Последние сделки -->
            <h2 class="section-title">📋 Последние 20 сделок</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Время входа</th>
                            <th>Время выхода</th>
                            <th>Направление</th>
                            <th>Цена входа</th>
                            <th>Цена выхода</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Причина</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for trade in trades[-20:] %}
                        <tr>
                            <td>{{ trade.entry_time.strftime('%Y-%m-%d %H:%M') }}</td>
                            <td>{{ trade.exit_time.strftime('%Y-%m-%d %H:%M') }}</td>
                            <td>
                                <span class="badge {{ 'badge-success' if trade.position == 'buy' else 'badge-danger' }}">
                                    {{ trade.position.upper() }}
                                </span>
                            </td>
                            <td>{{ "%.2f"|format(trade.entry_price) }}</td>
                            <td>{{ "%.2f"|format(trade.exit_price) }}</td>
                            <td class="{{ 'positive' if trade.pnl > 0 else 'negative' }}">
                                ${{ "%.2f"|format(trade.pnl) }}
                            </td>
                            <td class="{{ 'positive' if trade.pnl > 0 else 'negative' }}">
                                {{ "%.2f"|format(trade.pnl_pct) }}%
                            </td>
                            <td>
                                <span class="badge {{ 'badge-success' if trade.exit_reason == 'take_profit' else 'badge-warning' if trade.exit_reason == 'stop_loss' else 'badge' }}">
                                    {{ trade.exit_reason }}
                                </span>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <!-- Модели (если есть) -->
            {% if models %}
            <h2 class="section-title">🤖 Модели</h2>
            <div class="metrics-grid">
                {% for name, model in models.items() %}
                <div class="metric-card">
                    <div class="metric-title">{{ name }}</div>
                    <table style="width: 100%; margin-top: 10px;">
                        {% for metric, value in model.metrics.items() %}
                        {% if metric in ['accuracy', 'f1_score', 'precision', 'recall'] %}
                        <tr>
                            <td>{{ metric }}:</td>
                            <td><strong>{{ "%.4f"|format(value) }}</strong></td>
                        </tr>
                        {% endif %}
                        {% endfor %}
                    </table>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>

        <div class="footer">
            Сгенерировано с помощью ML Trading Backtester • {{ date }}
        </div>
    </div>
</body>
</html>
"""

    def _fig_to_base64(self, fig) -> str:
        """Конвертирует matplotlib фигуру в base64 строку."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def generate(self, df: pd.DataFrame, metrics: Dict, trades: List,
                 equity_curve: List[float], models: Optional[Dict] = None,
                 feature_importance: Optional[Tuple] = None) -> Path:
        """
        Генерирует HTML отчет.

        Args:
            df: DataFrame с данными
            metrics: метрики производительности
            trades: список сделок
            equity_curve: кривая капитала
            models: словарь моделей
            feature_importance: (feature_names, importance_values)

        Returns:
            Путь к сгенерированному отчету
        """
        log.info(f"📝 Генерация HTML отчета для {self.symbol}...")

        # Создаем графики
        print("  Создание графиков...")

        # Кривая капитала
        fig1 = self.viz.plot_equity_curve(equity_curve, trades, show=False, save=False)
        equity_curve_img = self._fig_to_base64(fig1)

        # Цена и сигналы
        predictions = df['signal'].values if 'signal' in df.columns else None
        fig2 = self.viz.plot_price_with_signals(
            df.iloc[-500:], predictions[-500:] if predictions is not None else None,
            trades[-30:], show=False, save=False
        )
        price_signals_img = self._fig_to_base64(fig2)

        # Анализ просадок
        fig3 = self.viz.plot_drawdown_analysis(equity_curve, show=False, save=False)
        drawdown_img = self._fig_to_base64(fig3)

        # Важность признаков
        feature_importance_img = None
        if feature_importance is not None:
            feature_names, importance_values = feature_importance
            fig4 = self.viz.plot_feature_importance(
                feature_names, importance_values, show=False, save=False
            )
            feature_importance_img = self._fig_to_base64(fig4)

        # Подготовка данных для шаблона
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_dir))
        template = env.get_template('report_template.html')

        # Конвертируем trades в список словарей для шаблона
        trades_dict = []
        for t in trades[-20:]:
            trades_dict.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'position': t.position,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason
            })

        # Подготавливаем модели для отображения
        models_dict = {}
        if models:
            for name, model in models.items():
                if hasattr(model, 'metrics'):
                    models_dict[name] = {'metrics': model.metrics}

        html_content = template.render(
            symbol=self.symbol,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=metrics,
            trades=trades_dict,
            models=models_dict,
            equity_curve_img=equity_curve_img,
            price_signals_img=price_signals_img,
            drawdown_img=drawdown_img,
            feature_importance_img=feature_importance_img
        )

        # Сохраняем отчет
        report_path = self.output_dir / f"{self.symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        log.info(f"✅ HTML отчет сохранен: {report_path}")

        return report_path