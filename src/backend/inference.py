import logging
from typing import Optional, Dict, Any
from pathlib import Path
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

# Configure module-level logger
logger = logging.getLogger("Aura.Inference")

class AuraInference:
    """
    High-performance Inference Engine for Aura, leveraging OpenVINO 
    Runtime for hardware-accelerated LLM generation.
    """
    def __init__(self, model_path: str, device: str = "CPU"):
        self.model_path = Path(model_path)
        self.device = device.upper()
        
        try:
            logger.info(f"Loading OpenVINO model from {model_path} on {self.device}")
            
            # Professional configuration: 
            # - use_cache=True for faster sequential generation
            # - compile=True to optimize graph for the specific device on load
            self.model = OVModelForCausalLM.from_pretrained(
                model_path, 
                compile=True, 
                device=self.device,
                ov_config={"PERFORMANCE_HINT": "LATENCY"},
                use_cache=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set padding token if missing (common in Llama/SmolLM models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Inference engine initialized successfully.")
            
        except Exception as e:
            logger.critical(f"Failed to load model: {str(e)}")
            raise

    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 256, 
        temperature: float = 0.7,
        stream: bool = True
    ) -> str:
        """
        Executes text generation with support for real-time streaming.
        """
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)

            # Professional move: Use a Streamer for better UX
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

            logger.debug(f"Input tokens: {inputs['input_ids'].shape[1]}")
            
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                repetition_penalty=1.1,
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id
            )

            # Return the decoded text (useful if not streaming or for logging)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return "Error: System failed to generate response."

if __name__ == "__main__":
    # Integration test for the inference engine
    logging.basicConfig(level=logging.INFO)
    
    # Update this path to your local Intel-optimized model
    ENGINE_PATH = "models/smollm2-1.7b-int4" 
    
    try:
        engine = AuraInference(ENGINE_PATH)
        print("\n" + "="*30 + " AURA LIVE " + "="*30)
        engine.generate_response("Explain RAG in one sentence.", stream=True)
        print("\n" + "="*71)
    except Exception:
        print("Test failed. Check model path and OpenVINO installation.")