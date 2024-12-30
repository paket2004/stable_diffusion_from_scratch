import telebot
import requests

FASTAPI_ENDPOINT ="http://127.0.0.1:8000/generate_image"

TELEGRAM_BOT_TOKEN = "ENTER YOUR TOKEN"

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome_message(message):
    bot.reply_to(message, "Hello! Send /generate to create an image")

@bot.message_handler(commands=['generate'])
def generate_image(message):
    
    prompt = message.text.replace('/generate', '').strip()

    if not prompt:
        bot.reply_to(message, 'Please provide a prompt after /generate.')
        return


    bot.reply_to(message, "Generating an image... Wait a bit.")

    response = requests.get(f"{FASTAPI_ENDPOINT}?text={prompt}")
    if response.status_code == 200:
        image = response.content
        bot.send_photo(message.chat.id, photo=image)
    else:
        bot.reply_to(message, "Failed to generate image")

bot.polling()
