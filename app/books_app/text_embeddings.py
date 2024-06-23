# -*- coding: utf-8 -*-
from typing import Union, List

import requests
import logging.config
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
HOST = config.get('model_emb', 'host')
PORT = config.getint('model_emb', 'port')

URL = f'http://{HOST}:{PORT}/embeddings'

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger()


def get_embeddings(texts: Union[List, str]) -> np.array:
    data = {
        "texts": texts
    }

    try:
        response = requests.post(URL, json=data)
        response.raise_for_status()
        response_json = response.json()
        if 'embeddings' in response_json:
            embeddings = response_json['embeddings']
            return np.array(embeddings)
        else:
            logger.error('No embeddings found in the response: %s', response.text)
            raise ValueError('No embeddings found in the response')
    except Exception as e:
        logger.error('Request failed: %s', str(e))
        raise
