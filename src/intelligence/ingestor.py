import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Path resolution
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.backend.video_ingestor import VideoIngestor
from src.intelligence.vision_engine import AuraVision

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion.log") 
    ]
)
logger = logging.getLogger("Aura.MultimodalIngestor")

class MultimodalIngestor:
    """
    Unified ingestion service to process various file types (Video, Image)
    into a semantic metadata format for vector indexing.
    """
    def __init__(self, model_path: str = "models/vision-brain-int4"):
        logger.info("Initializing Vision Intelligence...")
        self.vision = AuraVision(model_path)
        self.output_path = project_root / "data" / "processed" / "metadata.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def process_image(self, file_path: Path) -> Dict[str, Any]:
        """Directly analyzes a static image."""
        logger.info(f"Processing static image: {file_path.name}")
        desc, speed = self.vision.describe_frame(str(file_path))
        return {
            "timestamp": "static",
            "description": desc,
            "frame": str(file_path),
            "type": "image"
        }

    def process_video(self, file_path: Path, interval: int) -> List[Dict[str, Any]]:
        """Extracts and analyzes frames from a video."""
        logger.info(f"Processing video: {file_path.name} at {interval}s intervals")
        video_tool = VideoIngestor()
        frames = video_tool.extract_frames(str(file_path), interval=interval)
        
        results = []
        for frame_path in tqdm(frames, desc="Analyzing Video", unit="frame"):
            try:
                desc, speed = self.vision.describe_frame(frame_path)
                timestamp = Path(frame_path).stem.split("_")[-1]
                results.append({
                    "timestamp": timestamp,
                    "description": desc,
                    "frame": frame_path,
                    "type": "video_frame"
                })
            except Exception as e:
                logger.error(f"Failed to process frame {frame_path}: {e}")
        return results

    def run(self, input_path: str, interval: int = 2):
        """Routes the input file to the appropriate processor based on extension."""
        path = Path(project_root / "data" / "raw" / input_path)
        if not path.exists():
            logger.error(f"Input not found: {path}")
            return

        ext = path.suffix.lower()
        metadata = []

        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            metadata = self.process_video(path, interval)
        elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
            metadata = [self.process_image(path)]
        else:
            logger.error(f"Unsupported file format: {ext}")
            return

        # Load existing metadata to append, or start fresh
        existing_data = []
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                existing_data = json.load(f)

        # Merge and save
        final_data = existing_data + metadata
        with open(self.output_path, 'w') as f:
            json.dump(final_data, f, indent=4)
        
        logger.info(f"Successfully indexed {len(metadata)} semantic moments.")

def main():
    parser = argparse.ArgumentParser(description="Aura Multimodal Ingestor")
    parser.add_argument("--file", type=str, required=True, help="Filename in data/raw/")
    parser.add_argument("--interval", type=int, default=2, help="Video frame interval")
    
    args = parser.parse_args()
    
    ingestor = MultimodalIngestor()
    ingestor.run(args.file, args.interval)

if __name__ == "__main__":
    main()