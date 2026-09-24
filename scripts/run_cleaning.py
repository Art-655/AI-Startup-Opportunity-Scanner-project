"""Clean the four startup source datasets and write processed CSV files."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.cleaner import DataCleaner

LOGGER = logging.getLogger(__name__)


def load_sources(dataset_directory: Path) -> Dict[str, pd.DataFrame]:
    """Load the four attached source datasets into DataFrames.

    Args:
        dataset_directory: Directory containing the source files.

    Returns:
        Mapping of source name to raw DataFrame.
    """
    try:
        new_source_files = {
            "ai_funding": dataset_directory
            / "ai_startup_funding_database_2014_2025.csv",
            "indian_funding": dataset_directory / "Indian_Startup_Funding_Dataset.csv",
            "producthunt": dataset_directory / "producthunt_products.csv",
            "startup_success": dataset_directory / "startup_success_dataset.csv",
        }
        if all(path.exists() for path in new_source_files.values()):
            return {
                source_name: pd.read_csv(path)
                for source_name, path in new_source_files.items()
            }
        product_hunt = pd.DataFrame(
            json.loads((dataset_directory / "data.json").read_text(encoding="utf-8"))
        )
        return {
            "funding": pd.read_csv(dataset_directory / "startup_funding.csv"),
            "crunchbase": pd.read_csv(
                dataset_directory / "investments_VC.csv", encoding="latin-1"
            ),
            "producthunt": product_hunt,
            "failures": pd.read_csv(
                dataset_directory / "startup_failure_prediction.csv"
            ),
        }
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as error:
        LOGGER.exception("Unable to load source datasets: %s", error)
        raise


def clean_sources(dataset_directory: Path, output_directory: Path) -> None:
    """Clean all source datasets and save CSVs and quality reports."""
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        reports = {}
        for source_name, dataframe in load_sources(dataset_directory).items():
            cleaner = DataCleaner(dataframe)
            cleaned_dataframe = cleaner.clean()
            cleaned_dataframe.to_csv(output_directory / f"{source_name}.csv", index=False)
            reports[source_name] = cleaner.generate_quality_report()
            LOGGER.info(
                "Saved %s: %d rows, %d columns",
                source_name,
                len(cleaned_dataframe),
                len(cleaned_dataframe.columns),
            )
        (output_directory / "quality_reports.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
    except (OSError, TypeError, ValueError, pd.errors.ParserError) as error:
        LOGGER.exception("Cleaning pipeline failed: %s", error)
        raise


def main() -> None:
    """Run the cleaning pipeline using the project data directories."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    new_dataset_directory = Path.home() / "Downloads" / "Datasets" / "AI startup"
    dataset_directory = (
        new_dataset_directory
        if new_dataset_directory.exists()
        else PROJECT_ROOT / "Dataset"
    )
    clean_sources(dataset_directory, PROJECT_ROOT / "data" / "processed")


if __name__ == "__main__":
    main()