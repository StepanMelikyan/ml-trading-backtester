# reports/telegram_bot.py
"""
Telegram бот для отправки уведомлений о сделках и отчетах.
"""

import asyncio
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log

load_dotenv()


class TelegramNotifier:
    """
    Отправка уведомлений о результатах торговли в Telegram.
    """

    def __init__(self):
        """Инициализация Telegram бота."""
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bot = None
        self.application = None

        if self.token and self.chat_id:
            self.bot = Bot(token=self.token)
            self._setup_commands()
            log.info("✅ Telegram бот инициализирован")
        else:
            log.warning("⚠️ Telegram не настроен. Укажите TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в .env")

    def _setup_commands(self):
        """Настройка команд бота."""
        self.application = Application.builder().token(self.token).build()

        # Добавляем обработчики команд
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("report", self.report_command))
        self.application.add_handler(CommandHandler("trades", self.trades_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("drawdown", self.drawdown_command))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start."""
        await update.message.reply_text(
            "🤖 <b>Привет! Я бот для мониторинга торговой стратегии.</b>\n\n"
            "📊 <b>Доступные команды:</b>\n"
            "/status - текущий статус (баланс, открытые позиции)\n"
            "/report - последний отчет\n"
            "/trades - последние сделки\n"
            "/stats - статистика за сегодня\n"
            "/drawdown - информация о просадке\n"
            "/help - помощь\n\n"
            "🔔 Я буду автоматически присылать уведомления о новых сделках.",
            parse_mode='HTML'
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help."""
        await update.message.reply_text(
            "📚 <b>Справка по командам:</b>\n\n"
            "<b>/status</b> - текущий баланс и открытые позиции\n"
            "<b>/report</b> - получить последний HTML отчет\n"
            "<b>/trades [N]</b> - последние N сделок (по умолчанию 5)\n"
            "<b>/stats</b> - статистика за сегодня\n"
            "<b>/drawdown</b> - информация о текущей просадке\n"
            "<b>/help</b> - эта справка",
            parse_mode='HTML'
        )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /status."""
        # Здесь должна быть логика получения текущего статуса
        # Это заглушка, нужно будет подключить к реальным данным
        await update.message.reply_text(
            "📊 <b>Текущий статус:</b>\n"
            "━━━━━━━━━━━━━━━\n"
            "💰 Баланс: $10,000.00\n"
            "📈 Открытых позиций: 0\n"
            "📊 Прибыль сегодня: $0.00\n"
            "📉 Прибыль за неделю: $0.00\n"
            "⚠️ Текущая просадка: 0.00%",
            parse_mode='HTML'
        )

    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /report - отправляет последний отчет."""
        await update.message.reply_text("🔍 Поиск последнего отчета...")

        # Ищем последний HTML отчет
        reports_dir = Path("reports")
        if reports_dir.exists():
            html_files = list(reports_dir.glob("*.html"))
            if html_files:
                latest_report = max(html_files, key=lambda p: p.stat().st_mtime)

                # Отправляем как документ
                await update.message.reply_document(
                    document=open(latest_report, 'rb'),
                    filename=latest_report.name,
                    caption=f"📊 <b>Торговый отчет</b>\nФайл: {latest_report.name}",
                    parse_mode='HTML'
                )

                # Также отправляем превью (первую картинку, если есть)
                preview_path = reports_dir / "plots" / "latest_preview.png"
                if preview_path.exists():
                    await update.message.reply_photo(
                        photo=open(preview_path, 'rb'),
                        caption="📈 Предварительный просмотр"
                    )
                return

        await update.message.reply_text("❌ Отчеты не найдены")

    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /trades - информация о последних сделках."""
        # Получаем количество сделок из аргументов
        n = 5
        if context.args and context.args[0].isdigit():
            n = min(int(context.args[0]), 20)  # Не больше 20

        # Здесь должна быть логика получения последних сделок
        # Это заглушка
        trades_text = f"📈 <b>Последние {n} сделок:</b>\n━━━━━━━━━━━━━━━\n"

        for i in range(n):
            trades_text += (
                f"\n<b>Сделка {i + 1}:</b>\n"
                f"  • Вход: BUY XAUUSD @ 1950.30\n"
                f"  • Выход: @ 1955.45\n"
                f"  • P&L: <b>+$51.50</b> (+2.64%)\n"
                f"  • Время: 2024-01-15 14:30\n"
                f"  • Результат: ✅ Take Profit\n"
            )

        await update.message.reply_text(trades_text, parse_mode='HTML')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats - статистика за сегодня."""
        # Здесь должна быть логика получения статистики
        # Это заглушка
        await update.message.reply_text(
            "📊 <b>Статистика за сегодня:</b>\n"
            "━━━━━━━━━━━━━━━\n"
            "💰 Прибыль: +$124.50\n"
            "📊 Сделок: 3\n"
            "✅ Прибыльных: 2 (66.7%)\n"
            "❌ Убыточных: 1 (33.3%)\n"
            "📈 Лучшая сделка: +$85.30\n"
            "📉 Худшая сделка: -$23.40\n"
            "⚡ Профит-фактор: 2.15",
            parse_mode='HTML'
        )

    async def drawdown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /drawdown - информация о просадке."""
        # Здесь должна быть логика получения информации о просадке
        # Это заглушка
        await update.message.reply_text(
            "📉 <b>Анализ просадки:</b>\n"
            "━━━━━━━━━━━━━━━\n"
            "📊 Текущая просадка: 3.2%\n"
            "📈 Максимальная просадка: 12.5%\n"
            "⏱ Длительность: 5 дней\n"
            "🔄 Восстановление: 80%\n"
            "📅 Дата макс. просадки: 2024-01-10\n\n"
            "⚠️ <i>Текущая просадка в пределах нормы</i>",
            parse_mode='HTML'
        )

    async def send_message(self, message: str, parse_mode: str = 'HTML'):
        """
        Отправляет сообщение в Telegram.

        Args:
            message: текст сообщения
            parse_mode: режим парсинга ('HTML' или 'Markdown')
        """
        if not self.bot or not self.chat_id:
            log.warning("❌ Telegram не настроен")
            return

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            log.info("✅ Сообщение отправлено в Telegram")
        except Exception as e:
            log.error(f"❌ Ошибка отправки в Telegram: {e}")

    async def send_photo(self, photo_path: Path, caption: str = ""):
        """
        Отправляет фото в Telegram.

        Args:
            photo_path: путь к фото
            caption: подпись к фото
        """
        if not self.bot or not self.chat_id:
            return

        try:
            with open(photo_path, 'rb') as photo:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo,
                    caption=caption
                )
            log.info(f"✅ Фото отправлено в Telegram: {photo_path.name}")
        except Exception as e:
            log.error(f"❌ Ошибка отправки фото: {e}")

    async def send_document(self, document_path: Path, caption: str = ""):
        """
        Отправляет документ в Telegram.

        Args:
            document_path: путь к документу
            caption: подпись к документу
        """
        if not self.bot or not self.chat_id:
            return

        try:
            with open(document_path, 'rb') as doc:
                await self.bot.send_document(
                    chat_id=self.chat_id,
                    document=doc,
                    filename=document_path.name,
                    caption=caption
                )
            log.info(f"✅ Документ отправлен в Telegram: {document_path.name}")
        except Exception as e:
            log.error(f"❌ Ошибка отправки документа: {e}")

    def send_trade_notification(self, trade_data: dict):
        """
        Отправляет уведомление о новой сделке.

        Args:
            trade_data: данные сделки
        """
        emoji = "🟢" if trade_data.get('pnl', 0) > 0 else "🔴"

        message = (
            f"{emoji} <b>Новая сделка</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Инструмент: {trade_data.get('symbol', 'Unknown')}\n"
            f"Направление: {trade_data.get('position', 'Unknown').upper()}\n"
            f"Цена входа: ${trade_data.get('entry_price', 0):.2f}\n"
            f"Цена выхода: ${trade_data.get('exit_price', 0):.2f}\n"
            f"Объем: {trade_data.get('size', 0):.2f} лотов\n"
            f"P&L: <b>${trade_data.get('pnl', 0):.2f}</b> ({trade_data.get('pnl_pct', 0):+.2f}%)\n"
            f"Причина: {trade_data.get('exit_reason', 'Unknown')}\n"
            f"Время: {trade_data.get('exit_time', datetime.now()).strftime('%Y-%m-%d %H:%M')}"
        )

        asyncio.create_task(self.send_message(message))

    def send_daily_report(self, metrics: dict):
        """
        Отправляет ежедневный отчет.

        Args:
            metrics: метрики за день
        """
        message = (
            "📊 <b>Ежедневный отчет</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📅 Дата: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"💰 Баланс: ${metrics.get('balance', 0):,.2f}\n"
            f"📈 Прибыль сегодня: ${metrics.get('daily_pnl', 0):,.2f}\n"
            f"📊 Сделок сегодня: {metrics.get('trades_today', 0)}\n"
            f"✅ Винрейт: {metrics.get('win_rate', 0) * 100:.1f}%\n"
            f"📉 Просадка: {metrics.get('drawdown', 0):.2f}%\n"
            f"⚡ Коэф. Шарпа: {metrics.get('sharpe', 0):.2f}"
        )

        asyncio.create_task(self.send_message(message))

    def send_alert(self, message: str, alert_type: str = 'info'):
        """
        Отправляет alert сообщение.

        Args:
            message: текст сообщения
            alert_type: тип alert ('info', 'warning', 'error')
        """
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }

        emoji = emoji_map.get(alert_type, 'ℹ️')

        full_message = f"{emoji} <b>{alert_type.upper()}</b>\n{message}"
        asyncio.create_task(self.send_message(full_message))

    def run_polling(self):
        """Запускает бота в режиме polling."""
        if self.application:
            log.info("🤖 Telegram бот запущен в режиме polling...")
            self.application.run_polling()

    def stop(self):
        """Останавливает бота."""
        if self.application:
            self.application.stop()
            log.info("🤖 Telegram бот остановлен")


# Создаем глобальный экземпляр
telegram = TelegramNotifier()