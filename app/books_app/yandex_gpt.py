# -*- coding: utf-8 -*-

from typing import Union, Dict
import requests
import logging.config
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
URL = config.get('yandex_gpt', 'url')
FOLDER_ID = config.get('yandex_gpt', 'folder_id')
API_KEY = config.get('yandex_gpt', 'api_key')
STREAM = config.getboolean('yandex_gpt', 'stream')
TEMPERATURE = config.getfloat('yandex_gpt', 'temperature')
MAX_TOKENS = config.getint('yandex_gpt', 'max_tokens')
SYSTEM_TEXT = config.get('yandex_gpt', 'system_text')

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger()


def get_transform_answer(answer: str) -> Union[str, Dict[str, str]]:
    try:
        data = dict()

        data['modelUri'] = f'gpt://{FOLDER_ID}/yandexgpt/latest'

        data['completionOptions'] = {'stream': STREAM,
                                     'temperature': TEMPERATURE,
                                     'maxTokens': MAX_TOKENS}

        data['messages'] = [
            {
                "role": "system",
                "text": SYSTEM_TEXT
            },
            {
                "role": "user",
                "text": answer
            }
        ]

        response = requests.post(URL, headers={"Authorization": f"Api-key {API_KEY}"}, json=data).json()
        logger.info(f'The response has been received: {response}')
        try:
            response_text = response['result']['alternatives'][0]['message']['text']
        except Exception as e:
            return {'error': str(e)}

        return response_text

    except Exception as e:
        return {'error': str(e)}
