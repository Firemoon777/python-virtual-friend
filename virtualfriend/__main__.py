import base64
import logging
import os

import json
from io import BytesIO

import openai
import torch

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from virtualfriend.dialogue import DialogueCore
from virtualfriend.sd import generate_img2img

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

openai.api_key = os.environ["OPENAI_TOKEN"]
DialogueCore(device="cuda:0", stop_words=["AI:", "Human:"])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")


async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_chat_action(ChatAction.TYPING)

    prompt = update.message.text

    result = DialogueCore().text(prompt)

    await update.message.reply_text(result)
    return

    template = (
        "Amelie is a chatbot that reluctantly answers questions with sarcastic responses:"
        "You: {request}\n"
        "Amelie: "
    )

    # response = openai.Completion.create(
    #     model="text-curie-001",
    #     prompt=template.format(request=update.message.text.replace("\n", " ")),
    #     temperature=0.5,
    #     max_tokens=60,
    #     top_p=0.3,
    #     frequency_penalty=0.5,
    #     presence_penalty=0.0,
    #     stop=["Amelie:", "You:"]
    # )
    # print(response)
    # first_response = response["choices"][0]["text"]
    # caption = first_response + "\n\n"
    caption = ""

    # await update.message.reply_text(first_response)
    print(gen(update.message.text))

    request_emotions = emotion_task(update.message.text)[0]
    response_emotions = emotion_task(update.message.text)[0]
    emotion_dict = {e['label']: e['score'] for e in response_emotions if e['score'] > 0.5}

    caption += "\n".join([f"{k}: {v:.2f}" for k, v in emotion_dict.items()])


    # modifier = " ".join([f"({k}:{v:.2f})" for k, v in emotion_dict.items()])
    # response = generate_img2img("portrait of beautiful woman " + modifier + " mountains")
    #
    # image = BytesIO(base64.b64decode(response.images[0]))
    # await update.message.reply_photo(image, caption)


if __name__ == '__main__':
    application = ApplicationBuilder().token(os.environ["BOT_TOKEN"]).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    text_handler = MessageHandler(filters.TEXT, text)
    application.add_handler(text_handler)

    application.run_polling()