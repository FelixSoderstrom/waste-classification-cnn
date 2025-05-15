"""Utility functions for finding and managing model checkpoints.

This module provides helper functions for:
- Finding the latest training session directory
- Locating model checkpoint files
- Determining default model paths for evaluation
"""

import os
import glob
from typing import List, Tuple, Optional, Union


def find_latest_session_dir(base_dir: str = "output") -> Optional[str]:
    """Find the latest training session directory based on session number.

    Scans the base directory for folders with names starting with "session_"
    followed by a number, and returns the path to the one with highest number.

    Args:
        base_dir: Base directory where session folders are stored

    Returns:
        Path to the latest session directory, or None if no session directories found
    """
    if not os.path.exists(base_dir):
        return None

    session_dirs: List[Tuple[int, str]] = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path) and d.startswith("session_"):
            try:
                session_num = int(d.split("_")[1])
                session_dirs.append((session_num, full_path))
            except (IndexError, ValueError):
                continue

    if not session_dirs:
        return None

    latest_session = sorted(session_dirs, key=lambda x: x[0], reverse=True)[
        0
    ][1]
    return latest_session


def find_checkpoint_in_dir(session_dir: str) -> Optional[str]:
    """Find a model checkpoint file in the given directory.

    Searches for checkpoint files (.ckpt) in the specified directory,
    prioritizing 'trained_model.ckpt' if it exists. If no checkpoints are found
    in the main directory, it also checks in fold subdirectories.

    Args:
        session_dir: Directory to search for checkpoint files

    Returns:
        Path to a checkpoint file, or None if no checkpoint found
    """
    if not os.path.exists(session_dir):
        return None

    trained_model_path = os.path.join(session_dir, "trained_model.ckpt")
    if os.path.exists(trained_model_path):
        return trained_model_path

    checkpoint_files: List[str] = glob.glob(
        os.path.join(session_dir, "*.ckpt")
    )

    if not checkpoint_files:
        for d in os.listdir(session_dir):
            fold_dir = os.path.join(session_dir, d)
            if os.path.isdir(fold_dir) and d.startswith("fold_"):
                checkpoint_files.extend(
                    glob.glob(os.path.join(fold_dir, "*.ckpt"))
                )

    if checkpoint_files:
        return sorted(checkpoint_files, key=os.path.getmtime)[-1]

    return None


def get_default_model_path() -> str:
    """Get the default model checkpoint path for evaluation.

    Automatically finds the latest session directory and model checkpoint
    to use as a default. Falls back to a predefined path if no sessions
    or checkpoints are found.

    Returns:
        Path to the latest model checkpoint, or fallback default if none found
    """
    fallback_path = "output/session_1/trained_model.ckpt"

    latest_session = find_latest_session_dir()
    if not latest_session:
        return fallback_path

    checkpoint_path = find_checkpoint_in_dir(latest_session)
    if not checkpoint_path:
        return fallback_path

    return checkpoint_path
