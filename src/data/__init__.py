# Keep package import light: avoid pulling torch via prefixes.
from .log_io import load_event_log, EventLogBundle
from .splits import split_cases

__all__ = [
    "load_event_log",
    "EventLogBundle",
    "split_cases",
    "build_prefix_dataset",
    "PrefixDataset",
]


def __getattr__(name: str):
    if name in {"build_prefix_dataset", "PrefixDataset", "PrefixSample", "collate_pad"}:
        from . import prefixes

        return getattr(prefixes, name)
    raise AttributeError(name)
