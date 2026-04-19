import importlib
import platform

from gemma4_audio.backends.base import InferenceBackend


def _try_import(module_name: str) -> bool:
    """Check if a module is importable without loading it fully."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


BACKEND_REGISTRY: dict[str, type] = {}

if _try_import("transformers"):
    from gemma4_audio.backends.transformers import TransformersBackend

    BACKEND_REGISTRY["transformers"] = TransformersBackend

if _try_import("vllm"):
    from gemma4_audio.backends.vllm import VLLMBackend

    BACKEND_REGISTRY["vllm"] = VLLMBackend

if _try_import("mlx_vlm"):
    from gemma4_audio.backends.mlx import MLXBackend

    BACKEND_REGISTRY["mlx"] = MLXBackend


def select_backend(name: str = "auto") -> InferenceBackend:
    """Select an inference backend by name, or auto-detect the best one."""
    if name != "auto":
        if name not in BACKEND_REGISTRY:
            available = ", ".join(sorted(BACKEND_REGISTRY.keys()))
            raise KeyError(f"Unknown backend '{name}'. Available: {available}")
        return BACKEND_REGISTRY[name]()

    if _cuda_available() and "vllm" in BACKEND_REGISTRY:
        return BACKEND_REGISTRY["vllm"]()

    if platform.system() == "Darwin" and "mlx" in BACKEND_REGISTRY:
        return BACKEND_REGISTRY["mlx"]()

    if "transformers" in BACKEND_REGISTRY:
        return BACKEND_REGISTRY["transformers"]()

    available = ", ".join(sorted(BACKEND_REGISTRY.keys())) or "(none)"
    raise RuntimeError(
        f"No inference backend installed. Available: {available}. "
        "Install one via `uv sync --extra {transformers,vllm,mlx}`."
    )


__all__ = ["BACKEND_REGISTRY", "select_backend"]
