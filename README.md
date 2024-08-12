# ColPali-RAG: Efficient Document Retrieval with Vision-Language Models

[ColPali is a cutting-edge Visual Retrieval-Augmented Generation (RAG)](https://arxiv.org/pdf/2407.01449) system designed to enhance document retrieval by combining textual and visual information. Utilizing advanced vision language models, ColPali excels in retrieving and processing visually rich documents, offering superior performance in a variety of applications.

Read the full project guide here: [ColPali-RAG Project Guide](https://medium.com/@ayoubkirouane3/colpali-efficient-document-retrieval-with-vision-language-models-cd47e8d83060)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Kirouane-Ayoub/colpali-rag.git
cd colpali-rag
pip install -r requirements.txt
apt-get install poppler-utils
```

## Setup the .env File

1. Create a `.env` file in the root directory of the project.
2. Add your Google AI API key (GEMINI_API_KEY) to the `.env` file:

```bash
GEMINI_API_KEY=your_google_api_key_here
```

You can get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Prepare Your PDF Files

Move the PDF files you want to process into the `pdfs` folder.

## Run the Gradio App

To launch the Gradio interface, simply run the main script:

```bash
python main.py
```

This will start the Gradio app, allowing you to interact with colpali-rag through a user-friendly web interface.
