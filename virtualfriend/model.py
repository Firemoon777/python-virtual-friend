from typing import Optional, List, Any, Dict

from pydantic import BaseModel
from telegram import User


class QAModel(BaseModel):
    user: str
    amelie: str


class EmotionModel(BaseModel):
    anger: float = 0.0
    disgust: float = 0.0
    fear: float = 0.0
    joy: float = 0.0
    neutral: float = 0.0
    sadness: float = 0.0
    surprise: float = 0.0

    @staticmethod
    def from_classifier(data: List[Dict]) -> 'EmotionModel':
        result = EmotionModel()
        for entry in data:
            if hasattr(result, entry["label"]):
                setattr(result, entry["label"], entry["score"])
        return result

    def get_suitable_emotion_code(self):
        arr = [
            ("F01", self.neutral),
            ("F02", self.surprise),
            ("F03", self.anger),
            ("F04", self.sadness),
            ("F05", self.joy),
            ("F06", self.fear),
            ("F07", 0.0),
            ("F08", self.disgust)
        ]
        arr.sort(key=lambda x: x[1], reverse=True)
        return arr[0][0]


class PromptModel(BaseModel):
    amelie_header: str = (
        # "I am a highly intelligent question answering bot. "
        # "If you ask me a question that is rooted in truth, I will give you the answer. "
        # "If you ask me a question that is nonsense, trickery, or has no clear answer, "
        # "I will respond with \"Unknown\"."
    )

    history: List[QAModel] = []

    data_in: str = ""
    data_out: Optional[str] = ""
    emotions: EmotionModel = EmotionModel()

    @staticmethod
    def create_from_user(user: Optional[User]) -> "PromptModel":
        if not user:
            raise Exception("User can not be None")

        result = PromptModel()

        # result.history.append(QAModel(
        #     user="What is human life expectancy in the United States?",
        #     amelie="Human life expectancy in the United States is 78 years."
        # ))
        #
        # result.history.append(QAModel(
        #     user="Who was president of the United States in 1955?",
        #     amelie="Dwight D. Eisenhower was president of the United States in 1955."
        # ))

        # result.history.append(QAModel(
        #     user=f"My name is {user.username}",
        #     amelie=f"Ok, I will name you {user.username}"
        # ))
        # if user.username:
        #     result.history.append(QAModel(
        #         user=f"My username is {user.username}",
        #         amelie=f"Ok, I got it!"
        #     ))

        return result

    def full_prompt(self) -> str:
        result = f"{self.amelie_header}\n\n"

        for line in self.history:
            result += f"You: {line.user}\nFriend: {line.amelie}\n\n"

        result += f"You: {self.data_in}\nFriend: "

        return result.strip()
