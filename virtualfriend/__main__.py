import base64
import logging
import os

import json
import re
import subprocess
import time
from io import BytesIO
from pathlib import Path

import openai
import torch

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from virtualfriend.dialogue import DialogueCore
from virtualfriend.model import PromptModel, EmotionModel

import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

Live2DExecutable = os.environ["LIVE2D_EXECUTABLE"]
openai.api_key = os.environ.get("OPENAI_TOKEN", "")
DialogueCore(device="cuda:0", stop_words=["You:", "Friend:"])

tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", run_opts={"device":"cuda"})
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder", run_opts={"device":"cuda"})


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")


async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_chat_action(ChatAction.TYPING)

    if "context" not in context.user_data:
        context.user_data["context"] = PromptModel.create_from_user(update.message.from_user)

    prompt: PromptModel = context.user_data["context"]

    prompt.data_in = update.message.text.strip()
    d = DialogueCore()
    result = d.reply(prompt)

    if not re.sub(r"[^A-Za-z ]*", "", result.data_out):
        await update.message.reply_text(result.data_out)
        return

    answer = result.data_out

    # Running the TTS
    logging.info("Generating wav...")
    sentence_wavs = []
    for i, sentence in enumerate(re.split("[.!?\n]", answer)):
        sentence = sentence.strip()
        if not sentence:
            continue

        print(sentence)

        mel_output, mel_length, alignment = tacotron2.encode_text(sentence)
        waveforms = hifi_gan.decode_batch(mel_output)
        wav_file = Path(f"/tmp/{update.message.from_user.id}_{update.message.id}_{i:03}.wav")
        torchaudio.save(wav_file, waveforms.squeeze(1).to("cpu") * 4, 22050, encoding="PCM_S", bits_per_sample=16)
        meta = torchaudio.info(wav_file)
        wav_length = meta.num_frames / meta.sample_rate
        emotional = EmotionModel.from_classifier(d.emotion_classifier(sentence)[0])
        sentence_wavs.append((wav_file, wav_length, emotional))

    wav_file = Path(f"/tmp/{update.message.from_user.id}_{update.message.id}.wav")
    cmd = "sox " + " ".join([str(p.absolute()) for p, _, _ in sentence_wavs]) + " " + str(wav_file.absolute())
    os.system(cmd)

    for p, _, _ in sentence_wavs:
        p.unlink()
    meta = torchaudio.info(wav_file)

    logging.info("Rendering video...")
    await update.message.reply_chat_action(ChatAction.RECORD_VIDEO_NOTE)
    video_file = Path(f"/tmp/{update.message.from_user.id}_{update.message.id}.mp4")
    p = subprocess.Popen(
        [
            Live2DExecutable,
            video_file.absolute()
        ],
        stdin=subprocess.PIPE,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
        encoding="ascii"
    )

    p.stdin.write(f"s{wav_file.absolute()}\n")
    p.stdin.flush()

    for _, length, emotion in sentence_wavs:
        await update.message.reply_chat_action(ChatAction.RECORD_VIDEO_NOTE)
        emotion_code = emotion.get_suitable_emotion_code()

        p.stdin.write(f"e{emotion_code}\n")
        p.stdin.flush()
        print(f"Waiting for {length}")
        time.sleep(length)

    p.stdin.write("q\n")
    p.stdin.flush()
    p.wait()

    logging.info("Merging video and audio...")
    merge_file = Path(f"/tmp/{update.message.from_user.id}_{update.message.id}.merge.mp4")
    print(merge_file)
    os.system(
        f"ffmpeg "
        f"-ss 00:00:01 "
        f"-i {video_file.absolute()} "
        f"-i {wav_file.absolute()} "
        f"-c:v copy "
        f"-c:a aac "
        f"{merge_file.absolute()} "
        f">/dev/null 2>/dev/null"
    )

    await update.message.reply_video_note(str(merge_file.absolute()))

    wav_file.unlink()
    video_file.unlink()
    merge_file.unlink()


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "context" in context.user_data:
        del context.user_data["context"]

    await update.message.reply_text("Done")


if __name__ == '__main__':
    application = ApplicationBuilder().token(os.environ["BOT_TOKEN"]).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    reset_handler = CommandHandler('reset', reset)
    application.add_handler(reset_handler)

    text_handler = MessageHandler(filters.TEXT, text)
    application.add_handler(text_handler)

    application.run_polling()
