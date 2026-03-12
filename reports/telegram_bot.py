# reports/telegram_bot.py
"""
Telegram бот с системой подписки для уведомлений о сделках
"""

import asyncio
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import pandas as pd
from pathlib import Path
import os
import json
from dotenv import load_dotenv
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log

load_dotenv()


class TelegramNotifier:
    """
    Отправка уведомлений с поддержкой множественных подписчиков
    """

    def __init__(self):
        """Инициализация Telegram бота"""
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.admin_chat_id = os.getenv('TELEGRAM_ADMIN_ID')
        self.bot = None
        self.application = None

        # Создаем папку data если её нет
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)

        self.subscribers_file = data_dir / "subscribers.json"
        self.subscribers = set()

        # Статистика
        self.signals_sent_today = 0
        self.last_signal_time = None
        self.last_signal_price = 0

        if self.token:
            self.bot = Bot(token=self.token)
            self._load_subscribers()
            self._setup_commands()

            # Добавляем админа автоматически
            if self.admin_chat_id and self.admin_chat_id.isdigit():
                admin_id = int(self.admin_chat_id)
                if admin_id not in self.subscribers:
                    self.subscribers.add(admin_id)
                    self._save_subscribers()
                    print(f"✅ Админ {admin_id} автоматически добавлен в подписчики")

            log.info(f"✅ Telegram бот инициализирован. Подписчиков: {len(self.subscribers)}")
        else:
            log.warning("⚠️ Telegram не настроен. Укажите TELEGRAM_BOT_TOKEN в .env")

    def _load_subscribers(self):
        """Загружает список подписчиков из файла"""
        try:
            print(f"🔍 Загрузка подписчиков из: {self.subscribers_file}")

            if self.subscribers_file.exists():
                with open(self.subscribers_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.subscribers = set(data.get('subscribers', []))
                print(f"✅ Загружено {len(self.subscribers)} подписчиков")
            else:
                print(f"📁 Файл {self.subscribers_file} не существует, будет создан при первой подписке")
                self.subscribers = set()
        except Exception as e:
            print(f"⚠️ Ошибка загрузки подписчиков: {e}")
            self.subscribers = set()

    def _save_subscribers(self):
        """Сохраняет список подписчиков в файл"""
        try:
            print(f"💾 Сохранение {len(self.subscribers)} подписчиков в {self.subscribers_file}")

            # Убедимся что папка существует
            self.subscribers_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.subscribers_file, 'w', encoding='utf-8') as f:
                json.dump({'subscribers': list(self.subscribers)}, f, indent=2)

            print(f"✅ Подписчики сохранены")

            # Проверим что файл создался
            if self.subscribers_file.exists():
                print(f"📁 Файл создан: {self.subscribers_file}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения подписчиков: {e}")

    def _setup_commands(self):
        """Настройка команд бота"""
        self.application = Application.builder().token(self.token).build()

        # Основные команды
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))

        # Команды подписки
        self.application.add_handler(CommandHandler("subscribe", self.subscribe_command))
        self.application.add_handler(CommandHandler("unsubscribe", self.unsubscribe_command))

        # Информационные команды
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("last", self.last_signal_command))
        self.application.add_handler(CommandHandler("subs", self.list_subscribers_command))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start - приветствие и автоматическая подписка"""
        user = update.effective_user
        chat_id = update.effective_chat.id

        print(f"\n👤 ПОЛУЧЕН START от {user.first_name} (ID: {chat_id})")
        print(f"Текущих подписчиков до добавления: {len(self.subscribers)}")

        # Автоматически подписываем пользователя
        if chat_id not in self.subscribers:
            self.subscribers.add(chat_id)
            self._save_subscribers()
            print(f"✅ Новый подписчик добавлен! Теперь подписчиков: {len(self.subscribers)}")
            log.info(f"✅ Новый подписчик: {user.first_name} (ID: {chat_id})")

            # Отправляем приветственное сообщение админу
            if self.admin_chat_id and self.admin_chat_id.isdigit():
                try:
                    await self.bot.send_message(
                        chat_id=int(self.admin_chat_id),
                        text=f"👤 Новый подписчик: {user.first_name} (@{user.username})\nID: {chat_id}",
                        parse_mode='HTML'
                    )
                except:
                    pass
        else:
            print(f"⏺ Пользователь уже был подписан")

        welcome_message = (
            f"👋 <b>Привет, {user.first_name}!</b>\n\n"
            f"✅ Вы подписаны на торговые сигналы!\n"
            f"📊 Теперь вы будете получать уведомления о сделках.\n\n"
            f"<b>Доступные команды:</b>\n"
            f"/help - список всех команд\n"
            f"/status - текущий статус\n"
            f"/stats - статистика бота\n"
            f"/last - последний сигнал\n"
            f"/unsubscribe - отписаться от уведомлений"
        )

        await update.message.reply_text(welcome_message, parse_mode='HTML')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = (
            "📚 <b>Справка по командам:</b>\n\n"
            "<b>📊 Основные:</b>\n"
            "/status - текущий статус\n"
            "/last - последний сигнал\n\n"
            "<b>🔔 Подписка:</b>\n"
            "/subscribe - подписаться\n"
            "/unsubscribe - отписаться\n\n"
            "<b>📈 Информация:</b>\n"
            "/stats - статистика бота\n"
            "/subs - список подписчиков (только админ)\n"
            "/help - эта справка"
        )
        await update.message.reply_text(help_text, parse_mode='HTML')

    async def subscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ручная подписка на уведомления"""
        chat_id = update.effective_chat.id

        if chat_id in self.subscribers:
            await update.message.reply_text("✅ Вы уже подписаны на уведомления!")
        else:
            self.subscribers.add(chat_id)
            self._save_subscribers()
            await update.message.reply_text(
                "✅ Вы успешно подписались на торговые сигналы!\n"
                "Теперь вы будете получать уведомления о сделках."
            )
            log.info(f"✅ Ручная подписка: {chat_id}")

    async def unsubscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Отписка от уведомлений"""
        chat_id = update.effective_chat.id

        if chat_id in self.subscribers:
            self.subscribers.remove(chat_id)
            self._save_subscribers()
            await update.message.reply_text(
                "❌ Вы отписались от уведомлений.\n"
                "Чтобы подписаться снова, используйте /subscribe"
            )
            log.info(f"❌ Отписка: {chat_id}")
        else:
            await update.message.reply_text(
                "Вы не подписаны на уведомления.\n"
                "Используйте /subscribe для подписки."
            )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Текущий статус бота"""
        status_text = (
            "📊 <b>Текущий статус:</b>\n"
            "━━━━━━━━━━━━━━━\n"
            f"👥 Подписчиков: {len(self.subscribers)}\n"
            f"🟢 Бот активен\n"
            f"📊 Сигналов сегодня: {self.signals_sent_today}\n"
            f"🕒 Последний сигнал: {self.last_signal_time.strftime('%H:%M') if self.last_signal_time else 'нет'}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await update.message.reply_text(status_text, parse_mode='HTML')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика бота"""
        stats_text = (
            "📈 <b>Статистика бота:</b>\n"
            "━━━━━━━━━━━━━━━\n"
            f"👥 Всего подписчиков: {len(self.subscribers)}\n"
            f"📊 Сигналов сегодня: {self.signals_sent_today}\n"
            f"🕒 Последний сигнал: {self.last_signal_time.strftime('%H:%M') if self.last_signal_time else 'нет'}\n"
            f"💰 Последняя цена: ${self.last_signal_price:.2f}\n"
            f"🤖 Версия: 2.1 (с анти-дублем)"
        )
        await update.message.reply_text(stats_text, parse_mode='HTML')

    async def last_signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Последний отправленный сигнал"""
        if self.last_signal_time:
            await update.message.reply_text(
                f"🔄 <b>Последний сигнал:</b>\n"
                f"Время: {self.last_signal_time.strftime('%H:%M:%S')}\n"
                f"Цена: ${self.last_signal_price:.2f}",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text("❌ Сигналов пока не было")

    async def list_subscribers_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Список подписчиков (только для админа)"""
        chat_id = update.effective_chat.id

        # Проверяем что это админ
        if str(chat_id) != self.admin_chat_id:
            await update.message.reply_text("⛔ Доступ запрещен")
            return

        subscribers_list = "\n".join([f"• {id}" for id in self.subscribers])
        await update.message.reply_text(
            f"📋 <b>Список подписчиков ({len(self.subscribers)}):</b>\n{subscribers_list}",
            parse_mode='HTML'
        )

    async def send_message(self, message: str, parse_mode: str = 'HTML'):
        """
        Отправляет сообщение ВСЕМ подписчикам с обработкой ошибок
        """
        if not self.bot:
            print("❌ Бот не инициализирован")
            return

        if not self.subscribers:
            print("❌ Нет подписчиков для рассылки")
            return

        print(f"📨 Рассылка сообщения {len(self.subscribers)} подписчикам...")
        success_count = 0
        fail_count = 0
        removed = []

        for chat_id in list(self.subscribers):
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
                success_count += 1
            except Exception as e:
                fail_count += 1
                error_text = str(e)
                print(f"❌ Ошибка для {chat_id}: {error_text}")

                # Если бот заблокирован или чат не найден - удаляем из подписчиков
                if "Forbidden" in error_text or "bot was blocked" in error_text or "Chat not found" in error_text:
                    removed.append(chat_id)

        # Удаляем заблокировавших бота
        for chat_id in removed:
            self.subscribers.remove(chat_id)
            print(f"🗑 Удален неактивный подписчик {chat_id}")

        if removed:
            self._save_subscribers()

        print(f"✅ Рассылка завершена: {success_count} успешно, {fail_count} ошибок")

        # Отправляем отчет админу
        if self.admin_chat_id and self.admin_chat_id.isdigit() and fail_count > 0:
            try:
                await self.bot.send_message(
                    chat_id=int(self.admin_chat_id),
                    text=f"📊 Отчет о рассылке:\n✅ Успешно: {success_count}\n❌ Ошибок: {fail_count}\n🗑 Удалено: {len(removed)}",
                    parse_mode='HTML'
                )
            except:
                pass

    async def send_photo(self, photo_path: Path, caption: str = ""):
        """Отправляет фото всем подписчикам"""
        if not self.bot or not self.subscribers:
            return

        print(f"📸 Отправка фото {len(self.subscribers)} подписчикам...")
        success_count = 0
        removed = []

        for chat_id in list(self.subscribers):
            try:
                with open(photo_path, 'rb') as photo:
                    await self.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=caption
                    )
                success_count += 1
            except Exception as e:
                print(f"❌ Ошибка отправки фото для {chat_id}: {e}")
                if "Forbidden" in str(e) or "bot was blocked" in str(e):
                    removed.append(chat_id)

        for chat_id in removed:
            self.subscribers.remove(chat_id)

        if removed:
            self._save_subscribers()

        print(f"✅ Фото отправлено {success_count} подписчикам")

    def send_signal(self, signal_data: dict):
        """
        Отправляет торговый сигнал всем подписчикам
        """
        action = signal_data.get('action', signal_data.get('trend', 'UNKNOWN'))
        emoji = "🟢" if action == 'BUY' else "🔴" if action == 'SELL' else "⚪"

        # Обновляем статистику
        self.signals_sent_today += 1
        self.last_signal_time = datetime.now()
        self.last_signal_price = signal_data.get('price', 0)

        # Формируем сообщение (оно уже отформатировано в dynamic_level_bot)
        message = signal_data.get('formatted_message', '')

        if message:
            asyncio.create_task(self.send_message(message))
        else:
            # Если сообщение не передано, используем простой формат
            price = signal_data.get('price', 0)
            zone = signal_data.get('zone', {})

            simple_message = (
                f"{emoji} <b>ТОРГОВЫЙ СИГНАЛ</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📊 <b>Действие:</b> {action}\n"
                f"💰 <b>Цена:</b> ${price:.2f}\n"
                f"🎯 <b>Зона:</b> ${zone.get('low', 0):.2f} - ${zone.get('high', 0):.2f}\n"
                f"⏰ <b>Время:</b> {datetime.now().strftime('%H:%M:%S')}"
            )
            asyncio.create_task(self.send_message(simple_message))

    def run_polling(self):
        """Запускает бота в режиме polling"""
        if self.application:
            print("🤖 Telegram бот запущен в режиме polling...")
            self.application.run_polling()

    def stop(self):
        """Останавливает бота"""
        if self.application:
            self.application.stop()
            self._save_subscribers()
            print("🤖 Telegram бот остановлен")


# Создаем глобальный экземпляр
telegram = TelegramNotifier()