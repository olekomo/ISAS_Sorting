"""Independent image and video recording utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np


class FrameRecorder:
    """Collect or stream RGB frames without coupling rendering to training."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, frame: np.ndarray, name: str = "frame.png") -> Path:
        path = self.output_dir / name
        imageio.imwrite(path, frame)
        return path

    def save_video(
        self,
        frames: Iterable[np.ndarray],
        name: str = "episode.mp4",
        fps: int = 10,
    ) -> Path:
        path = self.output_dir / name
        with imageio.get_writer(path, fps=fps, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)
        return path

    def save_gif(
        self,
        frames: Iterable[np.ndarray],
        name: str = "episode.gif",
        fps: int = 10,
    ) -> Path:
        path = self.output_dir / name
        imageio.mimsave(path, list(frames), duration=1.0 / fps, loop=0)
        return path
