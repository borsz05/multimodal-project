from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    # alapértelmezett: a projekt gyökere az a mappa, ahol a .git van
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

paths = Paths()
