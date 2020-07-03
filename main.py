import os
from io import BytesIO

from telegram import bot
from telegram.bot import Bot

from model import StyleTransferModel


model = StyleTransferModel()
first_image_file = {}


# Handle '/start' and '/help'
def hello(bot: Bot, update):
    chat_id = update.message.chat_id
    bot.send_message(chat_id, f'Приветик, {update.message.from_user.first_name}!\nЯ ArtStyleTransferBot '
                              f'и могу переносить стиль с одной картинки на другую. Для начала отправь, что будем менять')

def help(bot: Bot, update):
    chat_id = update.message.chat_id
    bot.send_message(chat_id, 'Я могу переносить стиль с одной картинки на другую, но делаю это не очень быстро.')

def send_prediction_on_photo(bot: Bot, update):
    # Нам нужно получить две картинки, чтобы произвести перенос стиля, но каждая картинка приходит в
    # отдельном апдейте, поэтому в простейшем случае мы будем сохранять id первой картинки в память,
    # чтобы, когда уже придет вторая, мы могли загрузить в память уже сами картинки и обработать их.
    chat_id = update.message.chat_id

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)

    if chat_id in first_image_file:

        # первая картинка, которая к нам пришла станет content image, а вторая style image
        content_image_stream = BytesIO()
        first_image_file[chat_id].download(out=content_image_stream)
        bot.send_message(chat_id, 'Крутяк! Исходная картинка есть, теперь отправляй картинку со стилем')
        del first_image_file[chat_id]

        style_image_stream = BytesIO()
        image_file.download(out=style_image_stream)
        bot.send_message(chat_id, 'Стиль получен! Пошёл работать.\nМожешь выпить пока чайку, потому что я считаю на cpu(')

        output = model.transfer_style(
            model.open_image(content_image_stream),
            model.open_image(style_image_stream)
        )

        # теперь отправим назад фото
        output_stream = BytesIO()
        output.save(output_stream, format='PNG')
        output_stream.seek(0)

        bot.send_message(chat_id, 'TADAAAM!')
        bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")
    else:
        first_image_file[chat_id] = image_file



if __name__ == '__main__':
    print('kek')
    from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    # используем прокси, так как без него у меня ничего не работало(
    updater = Updater(token=os.getenv('TELEGRAM_TOKEN'))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, send_prediction_on_photo))
    updater.dispatcher.add_handler(CommandHandler('start', hello))
    updater.dispatcher.add_handler(CommandHandler('help', help))
    updater.start_polling()

