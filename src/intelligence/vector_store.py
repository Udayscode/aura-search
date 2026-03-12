import os
import json
import logging
import lancedb
import pandas as pd
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer

# System-wide logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Aura.VectorStore")

class AuraSearch:
    """
    High-performance Vector Search Service using LanceDB.
    Designed for local inference on Intel AIPC architectures.
    """
    def __init__(self, db_path: str = "data/aura_db"):
        self.db_path = db_path
        self.table_name = "video_memories"
        
        try:
            logger.info("Initializing Transformer Model: all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Ensure database directory exists
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)
            logger.info(f"Persistent storage established at {self.db_path}")
        except Exception as e:
            logger.critical(f"Initialization failure: {str(e)}")
            raise

    def index_metadata(self, json_path: str) -> None:
        """Synchronizes local JSON metadata with the Vector Database index."""
        if not os.path.exists(json_path):
            logger.error(f"Source metadata missing: {json_path}")
            return

        try:
            with open(json_path, 'r') as f:
                raw_data = json.load(f)

            logger.info(f"Vectorizing {len(raw_data)} descriptions...")
            descriptions = [item['description'] for item in raw_data]
            vectors = self.model.encode(descriptions)

            processed_payload = []
            for i, item in enumerate(raw_data):
                processed_payload.append({
                    "vector": vectors[i],
                    "text": item['description'],
                    "timestamp": item['timestamp'],
                    "frame_path": item['frame']
                })

            self.table = self.db.create_table(
                self.table_name, 
                data=processed_payload, 
                mode="overwrite"
            )
            logger.info(f"Index synchronization complete. Table: {self.table_name}")
            
        except Exception as e:
            logger.error(f"Indexing operation failed: {str(e)}")

    def semantic_query(self, query: str, limit: int = 3, threshold: float = 0.85) -> pd.DataFrame:
        """
        Executes a similarity search. 
        Note: Threshold 0.85 is recommended for MiniLM-L6 fuzzy matching.
        """
        try:
            if not hasattr(self, 'table'):
                self.table = self.db.open_table(self.table_name)

            query_vector = self.model.encode([query])[0]
            
            # Retrieve raw results
            results = self.table.search(query_vector).limit(limit).to_pandas()

            filtered_results = results[results['_distance'] <= threshold]
            
            logger.info(f"Query: '{query}' | Best Dist: {results['_distance'].min():.4f} | Matches: {len(filtered_results)}")
            return filtered_results

        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Integration Test Environment
    service = AuraSearch()
    service.index_metadata("data/processed/metadata.json")
    
    # Validation Search
    test_query = "man in a suit"
    logger.info(f"Executing validation search for: {test_query}")
    matches = service.semantic_query(test_query)
    
    if not matches.empty:
        for idx, row in matches.iterrows():
            print(f"Match [{row['timestamp']}s] - Confidence: {1-row['_distance']:.2f}")