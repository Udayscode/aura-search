import cv2
import os
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("Aura.VideoBackend")

class VideoIngestor:
    """
    High-performance video processing backend. 
    Uses hardware-accelerated seeking to minimize CPU overhead during frame extraction.
    """
    def __init__(self, output_dir: str = "data/processed/frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path: str, interval: int = 2) -> List[str]:
        """
        Extracts frames at specific intervals using timestamp seeking.
        """
        if not os.path.exists(video_path):
            logger.error(f"Video source not found: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video codec.")
            return []

        # Technical Metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        
        logger.info(f"Indexing video: {Path(video_path).name} ({duration_sec:.2f}s, {fps} FPS)")
        
        saved_paths = []
        # Calculate timestamps to seek (0s, 2s, 4s...)
        timestamps = range(0, int(duration_sec), interval)

        for ts in timestamps:
            # PROFESSIONAL MOVE: Seek to the exact millisecond instead of reading every frame
            # 1000 is to convert seconds to milliseconds
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            
            success, frame = cap.read()
            if not success:
                logger.warning(f"Could not seek to {ts}s. Skipping.")
                continue

            # Generate structured filename
            filename = f"frame_{ts:04d}.jpg"
            save_path = self.output_dir / filename
            
            # Use high-quality JPEG encoding
            cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_paths.append(str(save_path))

        cap.release()
        logger.info(f"Extraction complete. {len(saved_paths)} samples written to disk.")
        return saved_paths