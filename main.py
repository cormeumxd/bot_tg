import logging
import os

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import model

HEROKU_APP_NAME = os.getenv('HEROKU_APP_NAME')
bot_token = os.getenv('TOKEN')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send picture of cat or dog')


def get_message(update: Update, context: CallbackContext) -> None:
    user = update.message.from_user
    update.message.reply_text('Only picture!')
    logger.info("Message from %s:", user.first_name)


def get_photo(update: Update, context: CallbackContext) -> None:
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    # photo_file.download('image.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'image.jpg')
    update.message.reply_text(
        'Okay now wait a few seconds!!!'
    )
    path = photo_file.download("image.jpg")
    predict = model.predict(open(path, 'rb'))
    if (predict == 1):
        update.message.reply_text(
            'This is dog'
        )
    else:
        update.message.reply_text(
            'This is cat'
        )




def main():
    # Create the Updater and pass it your bot's token.
    TOKEN = bot_token  # place your token here
    updater = Updater(TOKEN, use_context=True)
    PORT = int(os.environ.get('PORT', '8443'))

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    # dispatcher.add_handler(CommandHandler("help", help_command))

    dispatcher.add_handler(MessageHandler(Filters.text, get_message))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, get_photo))

    updater.start_webhook(listen="0.0.0.0",
                          port=PORT,
                          url_path=TOKEN)
    updater.bot.set_webhook(f'https://{HEROKU_APP_NAME}.herokuapp.com/' + TOKEN)
    updater.idle()


if __name__ == '__main__':
    main()
