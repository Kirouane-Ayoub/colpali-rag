import os

import gradio as gr
from PIL import Image

import settings
from utils import get_answer, index, search

pdf_files = pdf_files = [
    os.path.join(settings.DATA_FOLDER, file)
    for file in os.listdir(settings.DATA_FOLDER)
    if file.lower().endswith(".pdf")
]
_, ds, images = index(pdf_files)


def answer_query(query: str) -> (str, Image):
    _, best_page = search(query, ds, images)
    answer = get_answer(query, best_page)
    return answer, best_page


# Define the Gradio chat-like interface
interface = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(
        label="Ask me About your Documents:", placeholder="Enter your query here..."
    ),
    outputs=[gr.Textbox(label="Response"), gr.Image(type="pil", label="Image")],
    title="Copali RAG",
    description="ColPali: Visual Retriever based on PaliGemma-3B with ColBERT strategy",
    theme="default",
)

# Launch the interface
interface.launch()
