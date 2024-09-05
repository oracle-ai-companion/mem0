import logging
from pymilvus import MilvusClient
from mem0.configs.vector_stores.milvus import MilvusConfig
from typing import Optional  # Ensure Optional is imported

class MilvusIntegration:
    """
    Integration class for connecting and interacting with the Milvus vector store.

    Attributes:
        config (MilvusConfig): Configuration object for Milvus.
        client (MilvusClient): Instance of the Milvus client for database operations.
    """

    def __init__(self, config: MilvusConfig):
        """
        Initializes the MilvusIntegration with the provided configuration.

        Args:
            config (MilvusConfig): Configuration for connecting to Milvus.
        """
        self.config = config
        self.client = MilvusClient(self.config.cluster_endpoint, self.config.token)
        logging.info("Milvus client initialized successfully.")

    def create_collection(self):
        """
        Creates a collection in Milvus based on the configuration.

        Raises:
            Exception: If the collection creation fails.
        """
        collection_params = {
            "collection_name": self.config.collection_name,
            "dimension": self.config.dimension,
            "metric_type": self.config.metric_type,
            "auto_id": self.config.auto_id,
            "index_type": self.config.index_type,
        }
        try:
            self.client.create_collection(collection_params)
            logging.info(f"Collection '{self.config.collection_name}' created successfully.")
        except Exception as e:
            logging.error(f"Failed to create collection: {e}")
            raise

    def insert_vectors(self, vectors: list, ids: Optional[list] = None):
        """
        Inserts vectors into the Milvus collection.

        Args:
            vectors (list): List of vectors to insert.
            ids (Optional[list]): List of IDs for the vectors. If None, auto-generated IDs will be used.

        Raises:
            Exception: If the insertion fails.
        """
        try:
            self.client.insert(collection_name=self.config.collection_name, records=vectors, ids=ids)
            logging.info(f"Inserted {len(vectors)} vectors into collection '{self.config.collection_name}'.")
        except Exception as e:
            logging.error(f"Failed to insert vectors: {e}")
            raise

    def search_vectors(self, query_vector: list, limit: int = 10):
        """
        Searches for similar vectors in the Milvus collection.

        Args:
            query_vector (list): The vector to search for.
            limit (int): The maximum number of results to return.

        Returns:
            list: List of similar vectors found.

        Raises:
            Exception: If the search fails.
        """
        try:
            results = self.client.search(collection_name=self.config.collection_name, query_records=[query_vector], limit=limit)
            logging.info(f"Search completed. Found {len(results)} similar vectors.")
            return results
        except Exception as e:
            logging.error(f"Failed to search vectors: {e}")
            raise