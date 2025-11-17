# src/utils.py
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np


# === Paths ===
# Prosjektrot = mappen to nivåer opp fra denne filen (…/house-price-prediction)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_path() -> Path:
    path = PROJECT_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_data_path() -> Path:
    path = get_data_path() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_data_path() -> Path:
    path = get_data_path() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_reports_path() -> Path:
    path = PROJECT_ROOT / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_figures_path() -> Path:
    path = get_reports_path() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_path() -> Path:
    path = PROJECT_ROOT / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    """Opprett alle standardmapper hvis de ikke finnes."""
    _ = (
        get_data_path(),
        get_raw_data_path(),
        get_processed_data_path(),
        get_reports_path(),
        get_figures_path(),
        get_models_path(),
    )


# === Logging ===
def configure_logging(level: int = logging.INFO, log_to_file: bool = True) -> None:
    """
    Setter opp enkel logging til console (+ valgfritt til fil).
    Loggfil legger seg i reports/log.txt.
    """
    # Rydd opp ev. eksisterende handlers (nyttig ved kjøring i notebook)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)

    handlers = [logging.StreamHandler()]

    if log_to_file:
        reports_dir = get_reports_path()
        file_handler = logging.FileHandler(reports_dir / "log.txt", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )


# === Reproduserbarhet ===
def set_seed(seed: int = 42) -> None:
    """Sett seed for numpy og random (nyttig ved modelltrening)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
