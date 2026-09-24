"""Opportunity scoring for startup categories."""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OpportunityScorer:
    """Calculate a weighted, normalized opportunity score from 0 to 10."""

    WEIGHTS: Mapping[str, float] = {
        "market_demand": 0.25,
        "funding_growth": 0.20,
        "competition_level": 0.20,
        "trend_momentum": 0.15,
        "failure_risk": 0.20,
    }
    _FEATURE_ALIASES: Mapping[str, Sequence[str]] = {
        "market_demand": (
            "market_demand",
            "Market Demand",
            "market demand",
        ),
        "funding_growth": (
            "funding_growth",
            "Funding Growth",
            "funding growth",
        ),
        "competition_level": (
            "competition_level",
            "Competition Level",
            "competition level",
        ),
        "trend_momentum": (
            "trend_momentum",
            "Trend Momentum",
            "trend momentum",
        ),
        "failure_risk": (
            "failure_risk",
            "Failure Risk",
            "failure risk",
        ),
    }
    _DATE_ALIASES: Sequence[str] = (
        "funding_date",
        "Funding Date",
        "date",
        "Date",
        "last_funding_at",
        "Last Funding Date",
    )

    def __init__(self, half_life_days: float = 365.0) -> None:
        """Initialize the scorer.

        Args:
            half_life_days: Number of days after which a funding signal loses
                half of its weight. The default is one year.

        Raises:
            ValueError: If ``half_life_days`` is not positive.
        """
        if half_life_days <= 0:
            raise ValueError("half_life_days must be greater than zero")
        self.half_life_days = float(half_life_days)

    def calculate_scores(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate and rank opportunity scores for each category.

        The input must contain one row per category and the five feature
        columns represented by the aliases in ``_FEATURE_ALIASES``. Feature
        values are min-max normalized across the supplied categories. Funding
        growth receives exponential time decay when a recognized funding date
        column is available; the newest date in the input is the reference.

        Args:
            dataframe: Category-level feature DataFrame.

        Returns:
            A copy of the input sorted by descending ``opportunity_score``.

        Raises:
            TypeError: If ``dataframe`` is not a pandas DataFrame.
            ValueError: If required features are missing or values are invalid.
        """
        try:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("dataframe must be a pandas DataFrame")
            if dataframe.empty:
                result = dataframe.copy()
                result["opportunity_score"] = pd.Series(dtype="float64")
                return result

            resolved_columns = self._resolve_feature_columns(dataframe)
            numeric_features = self._numeric_features(dataframe, resolved_columns)
            numeric_features["funding_growth"] = self._apply_time_decay(
                numeric_features["funding_growth"], dataframe
            )

            normalized = self._normalize(numeric_features)
            normalized["competition_level"] = 1.0 - normalized[
                "competition_level"
            ]
            normalized["failure_risk"] = 1.0 - normalized["failure_risk"]
            result = dataframe.copy()
            result["opportunity_score"] = (
                normalized[list(self.WEIGHTS)]
                .mul(pd.Series(self.WEIGHTS))
                .sum(axis=1)
                .mul(10.0)
                .clip(0.0, 10.0)
                .round(4)
            )
            return result.sort_values(
                "opportunity_score", ascending=False, kind="mergesort"
            ).reset_index(drop=True)
        except (KeyError, TypeError, ValueError) as error:
            logger.exception("Unable to calculate opportunity scores: %s", error)
            raise

    def _resolve_feature_columns(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """Resolve canonical feature names to columns in the input."""
        resolved: Dict[str, str] = {}
        normalized_names = {
            str(column).strip().casefold(): column for column in dataframe.columns
        }
        for feature, aliases in self._FEATURE_ALIASES.items():
            column = next(
                (
                    normalized_names[alias.strip().casefold()]
                    for alias in aliases
                    if alias.strip().casefold() in normalized_names
                ),
                None,
            )
            if column is None:
                raise ValueError(
                    f"Missing required opportunity feature: {feature}"
                )
            resolved[feature] = column
        return resolved

    @staticmethod
    def _numeric_features(
        dataframe: pd.DataFrame, columns: Mapping[str, str]
    ) -> pd.DataFrame:
        """Convert feature columns to numeric values and reject invalid data."""
        numeric = pd.DataFrame(index=dataframe.index)
        for feature, column in columns.items():
            numeric[feature] = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric.isna().any().any():
            invalid_features = list(numeric.columns[numeric.isna().any()])
            raise ValueError(
                "Opportunity features must contain only numeric, non-null values: "
                + ", ".join(invalid_features)
            )
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Opportunity features must contain finite values")
        return numeric

    def _apply_time_decay(
        self, funding_growth: pd.Series, dataframe: pd.DataFrame
    ) -> pd.Series:
        """Discount funding growth according to the age of its funding date."""
        date_column = self._find_date_column(dataframe)
        if date_column is None:
            logger.info("No funding date column found; skipping funding time decay")
            return funding_growth
        dates = pd.to_datetime(dataframe[date_column], errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError(f"Funding date column contains invalid values: {date_column}")
        age_days = (dates.max() - dates).dt.total_seconds() / 86400.0
        decay = np.exp(-np.log(2.0) * age_days / self.half_life_days)
        return funding_growth * decay.to_numpy()

    def _find_date_column(self, dataframe: pd.DataFrame) -> Optional[str]:
        """Find a recognized funding date column, if present."""
        normalized_names = {
            str(column).strip().casefold(): column for column in dataframe.columns
        }
        for alias in self._DATE_ALIASES:
            if alias.strip().casefold() in normalized_names:
                return normalized_names[alias.strip().casefold()]
        return None

    @staticmethod
    def _normalize(features: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalize each feature, using 0.5 for constant features."""
        if len(features) == 1:
            return pd.DataFrame(0.5, index=features.index, columns=features.columns)
        minimums = features.min()
        ranges = features.max() - minimums
        normalized = features.subtract(minimums).divide(ranges.replace(0.0, np.nan))
        normalized = normalized.fillna(0.5)
        return normalized