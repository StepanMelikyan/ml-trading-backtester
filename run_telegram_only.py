# run_telegram_only.py
from reports.telegram_bot import telegram
import time

print("=" * 50)
print("🚀 TELEGRAM БОТ ЗАПУЩЕН ОТДЕЛЬНО")
print("=" * 50)
print(f"Подписчиков: {len(telegram.subscribers)}")
print("Нажмите Ctrl+C для остановки")
print("=" * 50)

try:
    telegram.run_polling()
except KeyboardInterrupt:
    telegram.stop()
    print("\n🛑 Бот остановлен")