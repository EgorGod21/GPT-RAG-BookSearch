# -*- coding: utf-8 -*-

import requests
import logging.config
import numpy as np
import configparser
from typing import Union

config = configparser.ConfigParser()
config.read('../config.ini')
FOLDER_ID = config.get('yandex_gpt', 'folder_id')
API_KEY = config.get('yandex_gpt', 'api_key')

EMBEDED_URL = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
HEADERS = {"Authorization": f"Api-key {API_KEY}"}

DOC_URI = f"emb://{FOLDER_ID}/text-search-doc/latest"
QUERY_URI = f"emb://{FOLDER_ID}/text-search-query/latest"

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger()


def get_embeddings(text: str, text_type: str = "query") -> Union[np.array, None]:
    try:
        query_data = {
            "modelUri": QUERY_URI if text_type == "query" else DOC_URI,
            "text": text,
        }
        response = requests.post(EMBEDED_URL, json=query_data, headers=HEADERS)
        response.raise_for_status()
        response_json = response.json()
        if 'embedding' in response_json:
            embeddings = response_json['embedding']
            return np.array(embeddings)
        else:
            logger.error('No embeddings found in the response: %s', response.text)
            raise ValueError('No embeddings found in the response')
    except Exception as e:
        logger.error('Request failed: %s', str(e))
        raise