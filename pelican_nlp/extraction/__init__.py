# Import extraction classes lazily so `from pelican_nlp.extraction.model_registry`
# does not load torch/accelerate.
def __getattr__(name):
    if name == "Model":
        from .language_model import Model
        return Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Model"]
