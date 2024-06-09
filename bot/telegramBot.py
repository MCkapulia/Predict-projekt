import telebot
from price import tinkoffPrice
from price import ModelCreate
from telebot import types
from dotenv import load_dotenv
import os
load_dotenv()
TOKEN = os.getenv('TOKEN1')
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def echo(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    itembtn1 = types.KeyboardButton('Прогноз')
    itembtn2 = types.KeyboardButton('Текущий курс')
    item3 = types.KeyboardButton("Переобучение модели")
    markup.add(itembtn1, itembtn2)
    markup.add(item3)
    bot.send_message(message.chat.id,
"""Привет! Я бот. Я помогу тебе заработать много деняк!!!
Я могу:
- Считать прогноз на 1, 3 и 5 дней вперёд;
- Показать текщую цену акции "Роснефть";
- Переобучить модель, чтобы прогнозы были более точными с помощью команды.
Жду указаний!)""", reply_markup=markup)

@bot.message_handler(content_types='text')
def message_reply(message):
    if message.text == "Прогноз":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = types.KeyboardButton("Прогноз на 1 день")
        item2 = types.KeyboardButton("Прогноз на 3 дня")
        item3 = types.KeyboardButton("Прогноз на 5 дней")
        back_button = types.InlineKeyboardButton(text="Назад", callback_data="back")
        markup.add(item1, item2, item3, back_button)
        bot.send_message(message.chat.id, 'Выберите насколько дней вы бы хотели узнать прогноз', reply_markup=markup)
    elif message.text == "Текущий курс":
        bot.send_message(message.chat.id, tinkoffPrice.currPrice())
    elif message.text == "Назад":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        itembtn1 = types.KeyboardButton('Прогноз')
        itembtn2 = types.KeyboardButton('Текущий курс')
        item3 = types.KeyboardButton("Переобучение модели")
        markup.add(itembtn1, itembtn2)
        markup.add(item3)
        bot.send_message(message.chat.id, "Вы вернулись в главное меню.", reply_markup=markup)
    elif message.text == "Прогноз на 1 день":
        bot.send_message(message.chat.id, 'Выполняется расчёт прогноза...')
        predict = ModelCreate.predict_1day()
        bot.send_message(message.chat.id, predict)
    elif message.text == "Прогноз на 3 дня":
        bot.send_message(message.chat.id, 'Выполняется расчёт прогноза...')
        predict = ModelCreate.predict_3day()
        bot.send_message(message.chat.id, predict[2])
    elif message.text == "Прогноз на 5 дней":
        bot.send_message(message.chat.id, 'Выполняется расчёт прогноза...')
        predict = ModelCreate.predict_5day()
        bot.send_message(message.chat.id,predict[4])
    elif message.text == "Переобучение модели":
        bot.send_message(message.chat.id, 'Подождите, мы переобучаем модель!')
        ModelCreate.create_model('ROSN.me')
        bot.send_message(message.chat.id, 'Модель готова к использованию!')
    else:
        bot.send_message(message.chat.id,'Неизвестная команда!')

bot.polling(none_stop=True)