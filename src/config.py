from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    # a projekt gyökere: src/.. 
    project_root: Path = Path(__file__).resolve().parents[1]

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"

    # --- Flickr30k-specifikus útvonalak ---
    @property
    def flickr30k_dir(self) -> Path:
        return self.data_dir / "flickr30k"

    @property
    def flickr30k_images_dir(self) -> Path:
        return self.flickr30k_dir / "images"

    @property
    def flickr30k_annotations_dir(self) -> Path:
        return self.flickr30k_dir / "annotations"

paths = Paths()