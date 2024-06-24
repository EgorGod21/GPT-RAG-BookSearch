import logging.config
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import configparser
from typing import List, Dict, Union

from yandex_gpt import get_transform_answer
from text_embeddings import get_embeddings
from weaviate_search import get_search_results

config = configparser.ConfigParser()
config.read('../config.ini')
HOST = config.get('books', 'host')
PORT = config.getint('books', 'port')

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger()

app = FastAPI()


class Body(BaseModel):
    text: str = Field(min_length=1)


@app.post('/books')
async def get_books(body: Body) -> Union[List[Dict], Dict]:
    try:
        text = body.text
        logger.info(f'Requests {text}')
        text_transform = get_transform_answer(text)

        if 'error' in text_transform:
            return text_transform

        text_transform_lower = text_transform.lower().replace('\n', ' ')
        logger.info(f'Successfully got text: {text_transform_lower}')
        embeddings = list(get_embeddings(text_transform_lower[0]))
        logger.info(f'Successfully got embeddings')

        books = get_search_results(text_transform_lower, embeddings)
        return books

    except Exception as e:
        logger.error(f'Error processing getting books_app request: {str(e)}')
        return {'error': str(e)}

if __name__ == "__main__":
    logger.info('Starting uvicorn server...')

    try:
        uvicorn.run(app, host=HOST, port=PORT)
    except Exception as e:
        logger.error(f'Failed to start uvicorn server: {str(e)}')