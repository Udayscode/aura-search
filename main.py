import logging
import sys
import os
from pathlib import Path
from src.intelligence.vector_store import AuraSearch
from src.backend.inference import AuraInference

# Configure structured logging for production observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), 
        logging.FileHandler("aura.log")   
    ]
)
logger = logging.getLogger("Aura.Main")

class AuraOrchestrator:
    """
    Orchestrates the RAG (Retrieval-Augmented Generation) workflow.
    Handles path resolution internally without external config files.
    """
    def __init__(self, model_path: str):
        # Resolve paths relative to the project root
        self.base_dir = Path(__file__).resolve().parent
        self.db_path = self.base_dir / "data" / "aura_db"
        self.metadata_path = self.base_dir / "data" / "processed" / "metadata.json"

        logger.info("Initializing Aura Search Engine...")
        self.searcher = AuraSearch(db_path=str(self.db_path))
        
        # Synchronize vector index with local metadata
        if self.metadata_path.exists():
            self.searcher.index_metadata(str(self.metadata_path))
        else:
            logger.warning(f"Metadata not found at {self.metadata_path}. Search may be limited.")
        
        logger.info(f"Initializing Inference Engine on {model_path}...")
        self.brain = AuraInference(model_path=model_path, device="CPU")

    def format_rag_prompt(self, query: str, context: str, timestamp: str) -> str:
        """Constructs a system-constrained prompt for context-aware generation."""
        return (
            f"<|system|>\n"
            f"You are Aura, a helpful AI video assistant. Answer the question using ONLY this context:\n"
            f"CONTEXT FROM VIDEO AT {timestamp}s: {context}\n"
            f"<|user|>\n"
            f"{query}\n"
            f"<|assistant|>\n"
            f"Based on the video at {timestamp} seconds,"
        )

    def start(self):
        print("\n" + "-"*60)
        print("SYSTEM READY: AURA MULTIMODAL ASSISTANT")
        print("-"*60)

        while True:
            try:
                user_input = input("\n[Aura Query] > ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    logger.info("Session terminated by user.")
                    break
                
                if not user_input:
                    continue

                # Stage 1: Semantic Retrieval (Top-1)
                results = self.searcher.semantic_query(user_input, limit=1)

                if not results.empty:
                    top_match = results.iloc[0]
                    context_text = top_match['text']
                    ts = top_match['timestamp']
                    # Using (1 - distance) as a proxy for confidence
                    confidence = 1 - top_match['_distance']

                    # Stage 2: Prompt Augmentation
                    prompt = self.format_rag_prompt(user_input, context_text, ts)
                    
                    print(f"\n[Matched @ {ts}s | Confidence: {confidence:.2%}]")
                    print("Aura: ", end="", flush=True)
                    
                    # Stage 3: Streaming Generation
                    self.brain.generate_response(prompt, stream=True)
                    print() 
                else:
                    print("\nAura: I couldn't find a direct match in my visual memory.")
                    print("Aura: ", end="", flush=True)
                    self.brain.generate_response(user_input, stream=True)
                    print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Runtime error during query execution: {e}")

if __name__ == "__main__":
    # Update this path to your Intel-optimized model folder (e.g., Llama-3 or SmolLM2)
    MODEL_DIR = "models/llama-1b-int4"
    
    if not os.path.exists(MODEL_DIR):
        logger.error(f"Model directory not found: {MODEL_DIR}")
    else:
        assistant = AuraOrchestrator(model_path=MODEL_DIR)
        assistant.start()