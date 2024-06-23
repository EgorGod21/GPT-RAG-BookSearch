from typing import Union
from transformers import AutoTokenizer, AutoModel
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import logging.config
import configparser

fast_app = FastAPI()

config = configparser.ConfigParser()
config.read('../config.ini')
MODEL_PATH = config['model_emb']['model_path']
TOK_PATH = config['model_emb']['tok_path']
HOST = config['model_emb']['host']
PORT = config['model_emb']['port']

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(TOK_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if device != "cuda":
    logger.warning("You are using CPU ONLY!")


class Body(BaseModel):
    texts: Union[list, str] = Field(min_length=1)


def get_embeddings(texts):
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = _mean_pooling(outputs, inputs['attention_mask'])

    embeddings = embeddings.cpu().numpy()

    return embeddings


@fast_app.post('/embeddings')
async def embeddings_post_request(body: Body):
    try:
        embeddings = get_embeddings(body.texts)

        embeddings_list = embeddings.tolist()

        return {'embeddings': embeddings_list}
    except Exception as e:
        logger.error(f'Error processing embeddings request: {str(e)}')
        return {'error': str(e)}

if __name__ == "__main__":
    logger.info('Starting uvicorn server...')
    try:
        uvicorn.run(fast_app, host=HOST, port=PORT)
    except Exception as e:
        logger.error(f'Failed to start uvicorn server: {str(e)}')

