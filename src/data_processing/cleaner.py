"""Utilities for cleaning startup datasets."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean a raw startup DataFrame while preserving its source columns."""

    _INDUSTRY_COLUMNS: Sequence[str] = (
        "industry",
        "Industry",
        "IndustryVertical",
        "category_list",
        "market",
        " market ",
        "sector",
    )
    _FUNDING_NAME_PATTERN = re.compile(
        r"fund|amount.*usd|usd.*amount|investment", re.IGNORECASE
    )
    _CRITICAL_ID_COLUMNS: Sequence[str] = (
        "permalink",
        "company",
        "Startup_ID",
        "id",
        "Startup_Name",
        "StartupName",
        "Name",
        "name",
    )

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize a cleaner with a defensive copy of ``dataframe``.

        Args:
            dataframe: Raw records to clean.

        Raises:
            TypeError: If ``dataframe`` is not a pandas DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        self.dataframe = dataframe.copy()
        self._duplicate_count = 0
        self._dropped_missing_id_count = 0

    def handle_missing_values(self) -> pd.DataFrame:
        """Impute funding amounts and remove rows without critical identifiers.

        Funding-like columns are converted to numeric values, then missing
        values are filled with the median for the detected industry and the
        overall median as a fallback. Text columns retain missing values so
        downstream analysis can distinguish unknown from empty text.

        Returns:
            The cleaned DataFrame.
        """
        try:
            critical_columns = [
                column
                for column in self._CRITICAL_ID_COLUMNS
                if column in self.dataframe.columns
            ]
            if critical_columns:
                missing_ids = self.dataframe[critical_columns].isna().all(axis=1)
                self._dropped_missing_id_count = int(missing_ids.sum())
                self.dataframe = self.dataframe.loc[~missing_ids].copy()

            industry_column = self._find_industry_column()
            for column in self._funding_columns():
                numeric_values = self._to_numeric(self.dataframe[column])
                if numeric_values.notna().any():
                    if industry_column:
                        grouped_medians = numeric_values.groupby(
                            self.dataframe[industry_column]
                        ).transform("median")
                        numeric_values = numeric_values.fillna(grouped_medians)
                    numeric_values = numeric_values.fillna(numeric_values.median())
                self.dataframe[column] = numeric_values
            logger.info(
                "Handled missing values; dropped %d rows without critical IDs",
                self._dropped_missing_id_count,
            )
            return self.dataframe
        except (KeyError, TypeError, ValueError) as error:
            logger.exception("Unable to handle missing values: %s", error)
            raise

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove fully duplicated records and return the resulting DataFrame."""
        try:
            duplicate_mask = self.dataframe.duplicated(keep="first")
            self._duplicate_count = int(duplicate_mask.sum())
            self.dataframe = self.dataframe.loc[~duplicate_mask].copy()
            logger.info("Removed %d duplicate rows", self._duplicate_count)
            return self.dataframe
        except (KeyError, TypeError, ValueError) as error:
            logger.exception("Unable to remove duplicates: %s", error)
            raise

    def generate_quality_report(self) -> Dict[str, Any]:
        """Return missing percentages, duplicate count, and column data types.

        Returns:
            A JSON-serializable report describing the current DataFrame.
        """
        try:
            row_count = len(self.dataframe)
            missing_percent = (
                (self.dataframe.isna().mean() * 100).round(2).to_dict()
                if row_count
                else {column: 0.0 for column in self.dataframe.columns}
            )
            return {
                "row_count": row_count,
                "column_count": len(self.dataframe.columns),
                "missing_percent": missing_percent,
                "duplicate_count": self._duplicate_count,
                "dropped_missing_id_count": self._dropped_missing_id_count,
                "data_types": {
                    column: str(dtype)
                    for column, dtype in self.dataframe.dtypes.items()
                },
            }
        except (TypeError, ValueError) as error:
            logger.exception("Unable to generate quality report: %s", error)
            raise

    def clean(self) -> pd.DataFrame:
        """Run missing-value handling and duplicate removal in sequence."""
        self.handle_missing_values()
        return self.remove_duplicates()

    def _find_industry_column(self) -> Optional[str]:
        """Find the first known industry column present in the DataFrame."""
        for column in self._INDUSTRY_COLUMNS:
            if column in self.dataframe.columns:
                return column
        return None

    def _funding_columns(self) -> List[str]:
        """Return columns whose names indicate funding or investment amounts."""
        return [
            column
            for column in self.dataframe.columns
            if self._FUNDING_NAME_PATTERN.search(str(column))
        ]

    @staticmethod
    def _to_numeric(values: pd.Series) -> pd.Series:
        """Convert currency-like values to numeric values."""
        cleaned = values.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce").astype("float64")