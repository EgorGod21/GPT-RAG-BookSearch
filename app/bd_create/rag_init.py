# -*- coding: utf-8 -*-

from tqdm import tqdm
import pandas as pd
import logging.config
import configparser

from weaviate import WeaviateClient

from app.books_app.weaviate_connect import connect_to_weaviate, create_collection
from app.books_app.text_embeddings import get_embeddings

config = configparser.ConfigParser()
config.read('../config.ini')
CLUSTER_URL = config.get('rag_init', 'cluster_url')
API_KEY = config.get('rag_init', 'api_key')
FILE_PATH = config.get('rag_init', 'file_path')
COLLECTION_NAME = config.get('rag_init', 'collection_name')

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger()


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f'Data loaded from {file_path}')
        return df
    except Exception as e:
        logger.error(f'Error loading data from {file_path}: {e}')
        raise


def batch_upload_objects(client: WeaviateClient, collection_name: str, df: pd.DataFrame) -> None:
    try:
        collection = client.collections.get(collection_name)

        with collection.batch.dynamic() as batch:
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                embeddings = get_embeddings(str(row['title_describe']))
                book_object = {
                    "year": row.year,
                    "title": row.title,
                    "author": row.author,
                    "url": row.url,
                    "title_describe": row.title_describe
                }
                batch.add_object(
                    properties=book_object,
                    vector=embeddings
                )
        logger.info(f'Successfully uploaded to collection {collection_name}')
    except Exception as e:
        logger.error(f'Error during batch upload to collection {collection_name}: {e}')
        raise


if __name__ == "__main__":
    file_path = FILE_PATH

    try:
        df_doc = load_data(file_path)

        client = connect_to_weaviate(
            cluster_url=CLUSTER_URL,
            api_key=API_KEY
        )

        create_collection(client, COLLECTION_NAME)

        batch_upload_objects(client, COLLECTION_NAME, df_doc)
    except Exception as e:
        logger.critical(e)
