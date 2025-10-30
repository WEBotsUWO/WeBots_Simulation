
import os
import json
import time
from typing import List, Tuple

# Note: We import tensorflow only inside functions so this module
# can be imported in contexts where TF might not be available yet.
# If TF is guaranteed present, you may move import to top.
def get_project_root(current_file: str = __file__) -> str:
    """Return absolute path to project root (one level above learning/)."""
    return os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))

def ensure_data_dirs(base_dir: str = None) -> Tuple[str, str]:
    """
    Create and return (models_dir, checkpoints_dir).
    If base_dir is provided it is used as the project root; otherwise the module
    determines project root relative to this file.
    """
    if base_dir is None:
        root = get_project_root()
    else:
        root = os.path.abspath(base_dir)

    models_dir = os.path.join(root, "data", "champion_models")
    checkpoints_dir = os.path.join(root, "data", "checkpoints")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    return models_dir, checkpoints_dir

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def save_champion_model(model, models_dir: str, generation: int, model_index: int, fitness: float,
                        extra_metadata: dict = None) -> str:
    """
    Save a single champion model (weights only) and metadata JSON.
    Returns the path to the weights file (string).
    """
    timestamp = _timestamp()
    model_name = f"evolution_gen_{generation}_model_{model_index}_fitness_{fitness:.1f}_{timestamp}"
    model_path = os.path.join(models_dir, f"{model_name}.weights.h5")

    # Save weights
    model.save_weights(model_path)

    # Metadata
    metadata = {
        "generation": int(generation),
        "model_index": int(model_index),
        "fitness": float(fitness),
        "timestamp": timestamp
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    meta_path = os.path.join(models_dir, f"{model_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path

def save_population_checkpoint(models: List, checkpoints_dir: str, generation: int) -> None:
    """
    Save weights for every model in the current population to the checkpoints_dir,
    using a predictable naming convention: generation_{g}_model_{i}.weights.h5

    Also writes a small pointer file latest_generation.txt containing the last saved generation.
    """
    for i, model in enumerate(models):
        ckpt_name = os.path.join(checkpoints_dir, f"generation_{generation}_model_{i}.weights.h5")
        model.save_weights(ckpt_name)

    # pointer for resume
    pointer = os.path.join(checkpoints_dir, "latest_generation.txt")
    with open(pointer, "w") as f:
        f.write(str(generation))

def get_latest_generation(checkpoints_dir: str) -> int:
    """
    Read the latest_generation.txt pointer if present; return -1 if no checkpoint exists.
    """
    pointer = os.path.join(checkpoints_dir, "latest_generation.txt")
    if not os.path.exists(pointer):
        return -1
    try:
        with open(pointer, "r") as f:
            return int(f.read().strip())
    except Exception:
        return -1

def load_population_checkpoint(models: List, checkpoints_dir: str, generation: int) -> int:
    """
    Load saved weights for each model in `models` for a specific saved generation.
    If a weight file for a model is missing, it is skipped.

    Returns the generation number loaded (for convenience).
    """
    import tensorflow as tf  # local import
    loaded_any = False
    for i, model in enumerate(models):
        ckpt_name = os.path.join(checkpoints_dir, f"generation_{generation}_model_{i}.weights.h5")
        if os.path.exists(ckpt_name):
            try:
                model.load_weights(ckpt_name)
                loaded_any = True
            except Exception as e:
                print(f"[Checkpoint] Failed to load {ckpt_name}: {e}")
        else:
            print(f"[Checkpoint] No file for model {i} at generation {generation}: {ckpt_name}")

    if loaded_any:
        return generation
    return -1

def try_load_latest(models: List, checkpoints_dir: str) -> int:
    """
    Attempt to load the latest checkpoint (based on latest_generation.txt).
    Returns loaded generation number, or -1 if none loaded.
    """
    gen = get_latest_generation(checkpoints_dir)
    if gen < 0:
        return -1
    loaded = load_population_checkpoint(models, checkpoints_dir, gen)
    if loaded >= 0:
        print(f"[Checkpoint] Loaded population from generation {loaded}")
        return loaded
    return -1
