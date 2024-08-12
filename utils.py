from typing import List, Tuple

import torch
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import (
    process_images,
    process_queries,
)
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import device, gemini_model, processor, retrieval_model


def get_answer(prompt: str, image: Image):
    """
    Generate an answer to a given prompt using an image as context.

    Args:
        prompt (str): The prompt to generate an answer for.
        image (Image): The image to use as context for generating the answer.

    Returns:
        str: The generated answer to the prompt.

    This function uses the `gemini_model` to generate an answer to a given prompt.
    It takes a prompt as a string and an image as input. The image is used as context for generating the answer.
    The function calls the `generate_content` method of the `gemini_model` and passes the prompt and image as arguments.
    The generated content is then returned as a string.
    """
    response = gemini_model.generate_content([prompt, image])
    return response.text


# Function to index the pdf document (Get the embedding of each page)
def index(file: List[str]) -> Tuple[str, List[torch.Tensor], List]:
    """
    Indexes a list of PDF files by converting them to images and generating document embeddings.

    Args:
        file (List[str]): A list of file paths to the PDF files.

    Returns:
        Tuple[str, List[torch.Tensor], List[Image]]: A tuple containing the following:
            - A string indicating the completion status of the indexing process ("Done !!").
            - A list of document embeddings generated for each page in the PDF files.
            - A list of PIL Image objects representing the images extracted from the PDF files.
    """
    images = []
    ds = []
    for f in file:
        images.extend(convert_from_path(f))

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = retrieval_model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return "Done !!", ds, images


# Function to retrieve the most relevant pages based on the query
def search(query: str, ds: List, images: List) -> Image:
    """
    Search for the most relevant page based on the given query.

    Args:
        query (str): The query string to search for.
        ds (List): The list of document embeddings.
        images (List): The list of images corresponding to the documents.

    Returns:
        Image: The most relevant image based on the query.

    Description:
        This function searches for the most relevant page based on the given query.
        It first creates a mock image placeholder and then processes the query using the processor.
        The query is then embedded using the retrieval model and the embeddings are added to the list 'qs'.
        The evaluation is performed using the retriever evaluator, and the best page is determined based on the scores.
        Finally, the corresponding image from the 'images' list is returned.
    """
    qs = []
    # Image placeholder
    mock_image = Image.new("RGB", (448, 448), (255, 255, 255))
    with torch.no_grad():
        batch_query = process_queries(processor, [query], mock_image)
        batch_query = {k: v.to(device) for k, v in batch_query.items()}
        embeddings_query = retrieval_model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run evaluation
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs, ds)
    best_page = int(scores.argmax(axis=1).item())
    return images[best_page]
