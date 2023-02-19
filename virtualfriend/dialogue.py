import logging
import os.path
from typing import List

import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList

from virtualfriend.model import PromptModel, QAModel, EmotionModel


class WordStopCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
        print(self.stop_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_sequence in self.stop_token_ids:
            l = len(stop_sequence)
            if input_ids[0][-l:].tolist() == stop_sequence:
                return True

        return False


class IGPT:
    is_initialized: bool = False
    emotion_classifier = None
    logger = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(IGPT, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.logger.info("Loading emotion classifier...")
        emotion_model_path = "j-hartmann/emotion-english-distilroberta-base"
        self.emotion_classifier = pipeline("text-classification", model=emotion_model_path, return_all_scores=True)


class GPTJ(IGPT):
    stop_words: List[str]
    device: str

    tokenizer = None
    model = None
    generator = None

    def __init__(self, stop_words: List[str] = None, device: str = None):
        if self.__class__.instance.is_initialized: return
        super().__init__()

        self.stop_words = stop_words if stop_words else ["\n"]
        self.device = device

        self.logger.info("Loading GPT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", device=self.device)

        if not os.path.exists("gptj.pt"):
            self.logger.info("No torch model found. Downloading...")
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16)

            self.logger.info("Saving in torch format...")
            torch.save(model, "gptj.pt")

        self.logger.info("Loading GPT-j model...")
        self.model = torch.load("gptj.pt")

        self.logger.info("Creating text-generation pipeline...")
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)

        self.logger.info("Building essentials...")
        stop_word_criteria = WordStopCriteria(self.tokenizer(stop_words).input_ids)
        self.stop_criteria_list = StoppingCriteriaList([stop_word_criteria])

        self.logger.info("Loaded!")
        self.is_initialized = True

    def text(self, prompt: str, stop_criteria=None, max_new_tokens=50) -> str:
        self.logger.info(f"Start generation\n=== {prompt}")
        result = self.generator(
            prompt,
            stopping_criteria=stop_criteria or self.stop_criteria_list,
            # eos_token_id=self.tokenizer("\n").input_ids[0],
            return_full_text=False,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            # top_k=0.3
        )[0]["generated_text"]
        self.logger.info(result)
        self.logger.info("Stopped\n====")

        if stop_criteria is None:
            for sequence in self.stop_words:
                result = result.rstrip(sequence)

        return result


class OpenAI(IGPT):
    def __init__(self, stop_words: List[str] = None, device: str = None):
        if self.__class__.instance.is_initialized: return
        super().__init__()
        self.stop_words = stop_words
        self.logger = logging.getLogger(__name__)

    def text(self, prompt, stop_criteria=None, max_new_tokens=550):
        self.logger.info(f"Request: {prompt}")
        response = openai.Completion.create(
            model="text-curie-001",
            prompt=prompt,
            temperature=0.5,
            max_tokens=max_new_tokens,
            top_p=0.3,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=stop_criteria or self.stop_words
        )
        print(response)
        first_response = response["choices"][0]["text"]
        return first_response.strip()


# Change parent to GPTJ or OpenAI
class DialogueCore(GPTJ):
    def reply(self, prompt: PromptModel):
        prompt.data_out = self.text(prompt.full_prompt()).strip()
        prompt.emotions = EmotionModel.from_classifier(self.emotion_classifier(prompt.data_out)[0])

        prompt.history.append(QAModel(user=prompt.data_in, amelie=prompt.data_out))

        return prompt