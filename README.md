# GPT-RAG-BookSearch

## Description

GPT-RAG-BookSearch is a project that provides an API for searching programming books using the [YandexGPT API](https://yandex.cloud/ru/docs/foundation-models/concepts/yandexgpt/) and RAG (Retrieval-Augmented Generation), based on the vector database [Weaviate](https://weaviate.io/). The system allows users to get book recommendations based on their queries using powerful natural language processing models.

## Installation

1. Clone the repository:

   ```bash
   https://github.com/EgorGod21/GPT-RAG-BookSearch.git
2. Ensure you have Python 3.6+ installed.
3. Navigate to the project directory:

   ```bash
   cd GPT-RAG-BookSearch/app
   ```
4. Install the required dependencies from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```
5. Sequentially run the files `app/model_embeddings/model_emb.py`, `app/bd_create/rag_init.py` and `app/books_app/books.py`.

You can also obtain embeddings using either [YandexGPT](https://yandex.cloud/en/docs/foundation-models/concepts/embeddings), тor your own model (see commits for details).

## Configuration

Fill in the following fields in the `config.ini` file:
```
[model_emb]
model_path = <YOUR_MODEL_PATH>
tok_path = <YOUR_TOKENIZER_PATH>

[rag_init]
cluster_url = <YOUR_CLUSTER_URL>
api_key = <YOUR_API_KEY>

[yandex_gpt]
folder_id = <YOUR_FOLDER_ID>
api_key = <YOUR_API_KEY>
```
You can use a pre-trained model from [Hugging Face](https://huggingface.co/), this project uses [this](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.
The `cluster_url` and `api_key` for Weaviate will be available after [registering and creating a cluster](https://console.weaviate.cloud/).
A tutorial for obtaining the ```folder_id``` and ```api_key``` for the YandexGPT API is [here](https://habr.com/ru/articles/780008/).