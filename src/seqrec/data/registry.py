"""Datasets used in the study (SPEC section 5, METHODS section 1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    filename: str
    max_len: int  # model input length L (METHODS 2.7)
    time_unit: str  # granularity of source timestamps
    sha256: str | None = None  # pinned; values match results/data/sources.csv


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            "ml-1m",
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "ml-1m.zip",
            200,
            "second",
            "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
        ),
        DatasetSpec(
            "ml-20m",
            "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
            "ml-20m.zip",
            200,
            "second",
            "96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3",
        ),
        DatasetSpec(
            "steam",
            "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_reviews.json.gz",
            "steam_reviews.json.gz",
            50,
            "day",
            "2a2166b28961889276b167780ca9fa3361022f9ea6fefcf51d34957395266316",
        ),
        DatasetSpec(
            "video-games",
            "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/"
            "rating_only/Video_Games.csv.gz",
            "Video_Games.csv.gz",
            50,
            "millisecond",
            "30692f9806e33a7b5d8b599b73d7ca33dbfefb0abdfc1578052a101f2e64b3cd",
        ),
    )
}
