import logging
import os

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

from credentionals import bot_token


HEROKU_APP_NAME = os.getenv('HEROKU_APP_NAME')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send picture of cat or dog')

def photo(update: Update, context: CallbackContext) -> None:
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('image.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'image.jpg')
    update.message.reply_text(
        'Okay now wait a few seconds!!!'
    )

def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    TOKEN = bot_token # place your token here
    updater = Updater(TOKEN, use_context=True)
    PORT = int(os.environ.get('PORT', '5000'))

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    #dispatcher.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))

    updater.start_webhook(listen="0.0.0.0",
                      port=PORT,
                      url_path=TOKEN)
    updater.bot.set_webhook(f'https://{HEROKU_APP_NAME}.herokuapp.com/' + TOKEN)
    updater.idle()

if __name__ == '__main__':
    main()