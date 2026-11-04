from .metrics import compute_eer, compute_metrics, format_metrics

__all__ = ["compute_eer", "compute_metrics", "format_metrics", "DeepfakePredictor"]


def __getattr__(name: str):
    if name == "DeepfakePredictor":
        from .predictor import DeepfakePredictor

        return DeepfakePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
