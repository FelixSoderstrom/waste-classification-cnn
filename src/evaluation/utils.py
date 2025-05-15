import os
import glob


def find_latest_session_dir(base_dir="output"):
    """
    Find the latest session directory based on its number.

    Args:
        base_dir: Base directory to search in

    Returns:
        str: Path to the latest session directory, or None if no session directories found
    """
    if not os.path.exists(base_dir):
        return None

    session_dirs = []
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


def find_checkpoint_in_dir(session_dir):
    """
    Find a checkpoint file in the given directory.
    Prioritizes 'trained_model.ckpt' if it exists.

    Args:
        session_dir: Directory to search in

    Returns:
        str: Path to a checkpoint file, or None if no checkpoint found
    """
    if not os.path.exists(session_dir):
        return None

    trained_model_path = os.path.join(session_dir, "trained_model.ckpt")
    if os.path.exists(trained_model_path):
        return trained_model_path

    checkpoint_files = glob.glob(os.path.join(session_dir, "*.ckpt"))

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


def get_default_model_path():
    """
    Automatically find the latest model checkpoint to use as default.

    Returns:
        str: Path to the latest model checkpoint, or fallback default if none found
    """
    fallback_path = "output/session_1/trained_model.ckpt"

    latest_session = find_latest_session_dir()
    if not latest_session:
        return fallback_path

    checkpoint_path = find_checkpoint_in_dir(latest_session)
    if not checkpoint_path:
        return fallback_path

    return checkpoint_path
