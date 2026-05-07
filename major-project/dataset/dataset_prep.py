"""Dataset preparation module.

Curate a complexity-labeled dataset from three sources:
- Simple  → Alpaca
- Medium  → Open-Orca
- Complex → CodeAlpaca
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True)
class Paths:
    """Directory and file paths for the data pipeline."""

    raw_dir: Path = Path("dataset/raw")
    processed_dir: Path = Path("dataset/processed")

    @property
    def raw_simple(self) -> Path:
        return self.raw_dir / "alpaca_raw.csv"

    @property
    def raw_medium(self) -> Path:
        return self.raw_dir / "openorca_raw.csv"

    @property
    def raw_complex(self) -> Path:
        return self.raw_dir / "codealpaca_raw.csv"

    @property
    def combined(self) -> Path:
        return self.processed_dir / "combined_raw.csv"

    @property
    def cleaned(self) -> Path:
        return self.processed_dir / "combined_cleaned.csv"

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for downloading a dataset."""

    name: str
    source: str
    split: str
    complexity: str
    columns_map: dict[str, str] | None = None
    columns_select: list[str] | None = None


class DatasetLoader:
    """Handles downloading and caching of datasets."""

    def __init__(self, paths: Paths) -> None:
        self.paths = paths

    def load_or_cache(
        self,
        config: DatasetConfig,
        cache_path: Path,
    ) -> pd.DataFrame:
        """Load from cache if exists, otherwise download and cache."""
        if cache_path.exists():
            print(f"{config.name} raw found, skipping download...")
            return pd.read_csv(cache_path)

        print(f"Downloading {config.name} ({config.complexity})...")
        try:
            dataset = load_dataset(config.source, split=config.split)
        except Exception as e:
            raise RuntimeError(f"Failed to download {config.name}: {e}") from e

        df = pd.DataFrame(dataset)

        if config.columns_map:
            df = df.rename(columns=config.columns_map)
        if config.columns_select:
            df = df[["question", "response"]]

        df["complexity"] = config.complexity
        df.to_csv(cache_path, index=False)
        print(f"{config.name} saved: {len(df)} rows")
        return df


class DataCleaner:
    """Cleans and filters dataset rows."""

    def __init__(
        self,
        min_question_words: int = 3,
        min_response_words: int = 10,
    ) -> None:
        self.min_question_words = min_question_words
        self.min_response_words = min_response_words

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning filters to the dataframe."""
        df = df.dropna(subset=["question", "response"])
        df = df.drop_duplicates(subset=["question"])
        df = df[df["question"].str.split().str.len() >= self.min_question_words]
        df = df[df["response"].str.split().str.len() >= self.min_response_words]
        return df.reset_index(drop=True)


class DatasetPipeline:
    """Orchestrates the full dataset preparation pipeline."""

    def __init__(
        self,
        paths: Paths | None = None,
        random_state: int = 42,
    ) -> None:
        self.paths = paths or Paths()
        self.loader = DatasetLoader(self.paths)
        self.cleaner = DataCleaner()
        self.random_state = random_state

    def _load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three datasets."""
        configs = [
            DatasetConfig(
                name="Alpaca",
                source="tatsu-lab/alpaca",
                split="train",
                complexity="simple",
                columns_map={"instruction": "question", "output": "response"},
            ),
            DatasetConfig(
                name="Open-Orca",
                source="Open-Orca/OpenOrca",
                split="train[:3%]",
                complexity="medium",
            ),
            DatasetConfig(
                name="CodeAlpaca",
                source="sahil2801/CodeAlpaca-20k",
                split="train",
                complexity="complex",
                columns_map={"instruction": "question", "output": "response"},
            ),
        ]

        cache_paths = [
            self.paths.raw_simple,
            self.paths.raw_medium,
            self.paths.raw_complex,
        ]

        return tuple(
            self.loader.load_or_cache(cfg, cache)
            for cfg, cache in zip(configs, cache_paths)
        )

    def _combine(self, df_simple: pd.DataFrame, df_medium: pd.DataFrame, df_complex: pd.DataFrame) -> pd.DataFrame:
        """Combine and balance datasets."""
        if self.paths.combined.exists():
            print("Combined file found, skipping combine...")
            return pd.read_csv(self.paths.combined)

        print("\nCombining datasets...")
        min_count = min(len(df_simple), len(df_medium), len(df_complex))
        print(f"Sampling {min_count} from each class...")

        df_combined = pd.concat([
            df_simple.sample(min_count, random_state=self.random_state),
            df_medium.sample(min_count, random_state=self.random_state),
            df_complex.sample(min_count, random_state=self.random_state),
        ]).reset_index(drop=True)

        df_combined.to_csv(self.paths.combined, index=False)
        print(df_combined["complexity"].value_counts())
        print(f"Combined dataset saved: {len(df_combined)} rows")
        return df_combined

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the combined dataset."""
        if self.paths.cleaned.exists():
            print("Cleaned file found, skipping cleaning...")
            return pd.read_csv(self.paths.cleaned)

        print("\nCleaning dataset...")
        before = len(df)
        df_cleaned = self.cleaner.clean(df)
        after = len(df_cleaned)

        df_cleaned.to_csv(self.paths.cleaned, index=False)
        print(f"Rows before cleaning: {before}")
        print(f"Rows after cleaning: {after}")
        print(f"Removed: {before - after} rows")
        print(df_cleaned["complexity"].value_counts())
        print("Cleaned data saved!")
        return df_cleaned

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline."""
        self.paths.ensure_dirs()
        df_simple, df_medium, df_complex = self._load_datasets()
        df_combined = self._combine(df_simple, df_medium, df_complex)
        return self._clean(df_combined)


def main() -> None:
    """Run the dataset preparation pipeline."""
    pipeline = DatasetPipeline()
    df_cleaned = pipeline.run()

    print("\nDataset preparation complete!")
    print(f"Final dataset: {len(df_cleaned)} rows")
    print(df_cleaned["complexity"].value_counts())


if __name__ == "__main__":
    main()