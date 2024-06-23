# -*- coding: utf-8 -*-

import logging.config
import configparser
from typing import List, Union, Dict

from app.books_app.weaviate_connect import connect_to_weaviate

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger()

config = configparser.ConfigParser()
config.read('../config.ini')
CLUSTER_URL = config.get('rag_init', 'cluster_url')
API_KEY = config.get('rag_init', 'api_key')
COLLECTION_NAME = config.get('rag_init', 'collection_name')
ALPHA = config.getfloat('weaviate_search', 'alpha')
LIMIT = config.getint('weaviate_search', 'limit')


def get_search_results(text: str, embeddings: List) -> Union[List[Dict], Dict]:
    try:
        client = connect_to_weaviate(
            cluster_url=CLUSTER_URL,
            api_key=API_KEY,
        )
        collection = client.collections.get(COLLECTION_NAME)
        response = collection.query.hybrid(
            query=text,
            vector=embeddings,
            query_properties=["title_describe"],
            alpha=ALPHA,
            limit=LIMIT
        )
        books_properties = []
        for book_property in response.objects:
            books_properties.append(book_property.properties)
            logger.info(book_property.properties)

        return books_properties

    except Exception as e:
        logger.error(f'Error searching in Weavite: {str(e)}')
        raise

