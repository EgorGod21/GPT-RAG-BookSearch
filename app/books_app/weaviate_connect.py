import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure
import logging.config
from weaviate import WeaviateClient

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger()


def connect_to_weaviate(cluster_url: str, api_key: str) -> WeaviateClient:
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=AuthApiKey(api_key=api_key)
        )
        logger.info(f'Connected to Weaviate at {cluster_url}')
        return client
    except Exception as e:
        logger.error(f'Error connecting to Weaviate at {cluster_url}: {e}')
        raise


def create_collection(client: WeaviateClient, collection_name: str) -> None:
    try:
        client.collections.create(
            collection_name,
            vectorizer_config=Configure.Vectorizer.none()
        )
        logging.info(f'Created collection {collection_name}')
    except Exception as e:
        logging.error(e)