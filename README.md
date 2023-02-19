# python-virtual-friend

Uses a couple of neural networks to produce response as video notes.

For education purposes.
Inspired by [hackdaddy8000/unsuperior-ai-waifu](https://github.com/hackdaddy8000/unsuperior-ai-waifu)

## Collection

- Textual model: OpenAI GPT-3 or [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B)
- Voice model: [speechbrain/tts-tacotron2-ljspeech + speechbrain/tts-hifigan-ljspeech](https://huggingface.co/speechbrain/tts-tacotron2-ljspeech)
- Emotion classifier: [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- Talking face model: [Live2D Haru](https://www.live2d.com/en/download/sample-data/)

## Requirements

- Linux with X
- Python 3.8+
- OpenAI token (for GPT-3) or GPU with 15+ GB VRAM (for GPT-J)
- [Live2D Samples](https://github.com/Live2D/CubismNativeSamples/tree/0542d7a70cf40e0a122cb7e8f61105a78126e16b)

## Installation

1. Install virtual python environment with huggingface transformers. [doc](https://huggingface.co/docs/transformers/installation)
2. Install python requirements `python3 -m pip install -r requirements.txt`
3. Clone [Live2D Samples](https://github.com/Live2D/CubismNativeSamples/tree/0542d7a70cf40e0a122cb7e8f61105a78126e16b) and apply git patch.
4. Build `Samples/OpenGL/Demo/proj.linux.cmake` project.

## Run

Provide environment variables:

| ENV               | Optional?                   | Description                                   |
|-------------------|-----------------------------|-----------------------------------------------|
| BOT_TOKEN         | No                          | Token for Telegram Bot API                    |
| OPENAI_TOKEN      | Yes (if GPT-J used instead) | Token for GPT-3                               |
| DISPLAY           | No                          | X Display variable. Used for OpenGL rendering |
| LIVE2D_EXECUTABLE | No                          | Path for Live2D Demo executable               |

Run app:

```bash
python3 -m virtualfriend
```