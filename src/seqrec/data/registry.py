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


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            "ml-1m",
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "ml-1m.zip",
            200,
            "second",
        ),
        DatasetSpec(
            "ml-20m",
            "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
            "ml-20m.zip",
            200,
            "second",
        ),
        DatasetSpec(
            "steam",
            "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_reviews.json.gz",
            "steam_reviews.json.gz",
            50,
            "day",
        ),
        DatasetSpec(
            "video-games",
            "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/"
            "rating_only/Video_Games.csv.gz",
            "Video_Games.csv.gz",
            50,
            "millisecond",
        ),
    )
}
