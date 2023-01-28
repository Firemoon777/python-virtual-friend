import base64
from typing import List, Dict

import requests
from pydantic import BaseModel


BASE_URL = "https://sd.f1remoon.com/sdapi/v1"
auth = ("sd", "JxiiRuvdUckLbkavXBLY")
HARDCODED_PATH = "/home/firemoon/stable-diffusion-webui/outputs/txt2img-images/00458-4287667073-portrait of beautiful woman.png"


class img2imgRequest(BaseModel):
    init_images: List[str] = []
    prompt: str
    negative_prompt: str = ""


class img2imgResponse(BaseModel):
    images: List[str]
    parameters: Dict
    info: str


def generate_img2img(prompt) -> img2imgResponse:
    payload = img2imgRequest(prompt=prompt)

    with open(HARDCODED_PATH, "rb") as f:
        payload.init_images.append(base64.b64encode(f.read()).decode())

    response = requests.post(f"{BASE_URL}/img2img", json=payload.dict(), auth=auth)
    response.raise_for_status()
    response = img2imgResponse(**response.json())
    return response
