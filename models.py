import os

import google.generativeai as genai
import torch
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from dotenv import load_dotenv
from transformers import AutoProcessor

import settings

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL_NAME)


retrieval_model = ColPali.from_pretrained(
    settings.COLPALI_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
retrieval_model.load_adapter(settings.MODEL_NAME)
processor = AutoProcessor.from_pretrained(settings.MODEL_NAME)
device = retrieval_model.device
