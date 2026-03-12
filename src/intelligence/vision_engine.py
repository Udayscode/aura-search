import logging
import time
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import torch
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor

# Standardize logging across the intelligence module
logger = logging.getLogger("Aura.VisionEngine")

class AuraVision:
    """
    Multimodal Vision Engine utilizing Llava-v1.5 with OpenVINO optimization.
    Capable of zero-shot image captioning and visual semantic extraction.
    """
    def __init__(self, model_path: str, device: str = "AUTO"):
        self.model_path = Path(model_path)
        
        try:
            logger.info(f"Initializing VLM on {device} (Source: {self.model_path.name})")
            
            # Using 'AUTO' allows OpenVINO to load balance between CPU/iGPU
            # 'trust_remote_code=True' is necessary for Llava architectures
            self.model = OVModelForVisualCausalLM.from_pretrained(
                str(self.model_path),
                device=device,
                trust_remote_code=True,
                ov_config={
                    "PERFORMANCE_HINT": "LATENCY",
                    "CACHE_DIR": "model_cache"
                },
                use_cache=True # Critical for generation speed
            )
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path), 
                trust_remote_code=True,
                use_fast=True
            )
            
            logger.info("Vision Intelligence online and hardware-accelerated.")
        except Exception as e:
            logger.critical(f"Inference hardware failed to initialize: {e}")
            raise

    def describe_frame(self, image_path: str) -> Tuple[str, float]:
        """
        Performs semantic analysis on a single visual frame.
        Returns a tuple of (description_string, latency_seconds).
        """
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Frame missing at {image_path}")

            # Pre-processing image
            raw_image = Image.open(image_path).convert("RGB")
            
            # Formulating the prompt for Llava-1.5-7b
            prompt = "USER: <image>\nDescribe this scene in one short, descriptive sentence.\nASSISTANT:"
            
            # Generate model inputs
            inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt")

            # Benchmarking inference latency
            start_time = time.perf_counter()
            output = self.model.generate(
                **inputs, 
                max_new_tokens=64,
                do_sample=False, # Use greedy search for deterministic/stable descriptions
                repetition_penalty=1.1
            )
            end_time = time.perf_counter()
            latency = end_time - start_time

            # Post-processing and cleanup
            full_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            
            # Robust extraction of the ASSISTANT response
            if "ASSISTANT:" in full_text:
                description = full_text.split("ASSISTANT:")[-1].strip()
            else:
                description = full_text.strip()

            return description, latency

        except Exception as e:
            logger.error(f"Visual inference failure for {image_path}: {e}")
            return "Semantic extraction failed.", 0.0

if __name__ == "__main__":
    # Integration test for stand-alone validation
    logging.basicConfig(level=logging.INFO)
    # Update with your actual local vision model path
    TEST_MODEL = "models/vision-brain-int4"
    
    if Path(TEST_MODEL).exists():
        engine = AuraVision(TEST_MODEL)
        # Verify with a placeholder if exists
        # desc, speed = engine.describe_frame("path/to/test.jpg")
    else:
        logger.warning("Test model directory not found. Skipping initialization test.")