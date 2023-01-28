import logging
import os.path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList


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


class DialogueCore:
    is_initialized: bool = False

    stop_words: List[str]
    device: str

    logger = None

    tokenizer = None
    model = None
    generator = None

    emotion_classifier = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(DialogueCore, cls).__new__(cls)
        return cls.instance

    def __init__(self, stop_words: List[str] = None, device: str = None):
        if self.__class__.instance.is_initialized: return

        self.stop_words = stop_words if stop_words else ["\n"]
        self.device = device

        self.logger = logging.getLogger(__name__)

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

        self.logger.info("Loading emotion classifier...")
        emotion_model_path = "j-hartmann/emotion-english-distilroberta-base"
        self.emotion_classifier = pipeline("text-classification", model=emotion_model_path, return_all_scores=True)

        self.logger.info("Building essentials...")
        stop_word_criteria = WordStopCriteria(self.tokenizer(stop_words).input_ids)
        self.stop_criteria_list = StoppingCriteriaList([stop_word_criteria])

        self.logger.info("Loaded!")
        self.is_initialized = True

    def text(self, prompt):
        self.logger.info("Start generation")
        result = self.generator(
            prompt,
            stopping_criteria=self.stop_criteria_list,
            return_full_text=False,
            max_new_tokens=50
        )[0]["generated_text"]
        self.logger.info("Stopped")

        for sequence in self.stop_words:
            result = result.rstrip(sequence)

        return result

