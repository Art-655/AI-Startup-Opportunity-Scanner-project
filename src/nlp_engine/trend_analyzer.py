"""BERTopic-based startup trend extraction and momentum analysis."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Extract emerging topics from startup descriptions."""

    def __init__(
        self,
        timestamps: Optional[Sequence[Any]] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        **bertopic_parameters: Any,
    ) -> None:
        """Initialize SentenceTransformers and BERTopic.

        Args:
            timestamps: Optional timestamps aligned with the descriptions that
                will be passed to :meth:`extract_topics`.
            embedding_model_name: SentenceTransformers model name.
            **bertopic_parameters: Optional BERTopic constructor parameters.

        Raises:
            ImportError: If BERTopic or SentenceTransformers is unavailable.
        """
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer

            self.embedding_model_name = embedding_model_name
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model, **bertopic_parameters
            )
            self.timestamps = list(timestamps) if timestamps is not None else None
            self.topic_assignments: Optional[np.ndarray] = None
            self.valid_descriptions: List[str] = []
            self.valid_timestamps: Optional[List[Any]] = None
        except ImportError as error:
            logger.exception("Unable to initialize trend analyzer: %s", error)
            raise ImportError(
                "BERTopic and sentence-transformers are required for TrendAnalyzer"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            logger.exception("Unable to initialize embedding model: %s", error)
            raise

    def extract_topics(
        self,
        descriptions: List[str],
        timestamps: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fit BERTopic and return the ten most prevalent emerging topics.

        Non-string values are skipped with a warning. Empty input or input
        containing no usable strings returns an empty list without fitting.
        Representative documents are returned for each topic when BERTopic
        provides them.

        Args:
            descriptions: Startup product or company descriptions.
            timestamps: Optional timestamps aligned with ``descriptions`` for
                later momentum analysis. When omitted, constructor timestamps
                are used.

        Returns:
            Topic dictionaries containing ``topic_id``, ``topic_name``,
            ``document_count``, and ``representative_documents``.

        Raises:
            TypeError: If ``descriptions`` is not a list.
            ValueError: If timestamps are not aligned with descriptions.
        """
        try:
            if not isinstance(descriptions, list):
                raise TypeError("descriptions must be a list of strings")
            if not descriptions:
                self._reset_fit_state()
                return []

            aligned_timestamps = timestamps
            if aligned_timestamps is None:
                aligned_timestamps = self.timestamps
            if aligned_timestamps is not None and len(aligned_timestamps) != len(
                descriptions
            ):
                raise ValueError("timestamps must align with descriptions")

            valid_descriptions: List[str] = []
            valid_timestamps: Optional[List[Any]] = (
                [] if aligned_timestamps is not None else None
            )
            for index, description in enumerate(descriptions):
                if not isinstance(description, str) or not description.strip():
                    logger.warning("Skipping non-string or empty description at index %d", index)
                    continue
                valid_descriptions.append(description.strip())
                if valid_timestamps is not None:
                    valid_timestamps.append(aligned_timestamps[index])
            if not valid_descriptions:
                self._reset_fit_state()
                return []

            topics, _ = self.topic_model.fit_transform(valid_descriptions)
            self.valid_descriptions = valid_descriptions
            self.valid_timestamps = valid_timestamps
            self.topic_assignments = np.asarray(topics, dtype=int)
            topic_info = self.topic_model.get_topic_info()
            results: List[Dict[str, Any]] = []
            for row in topic_info.itertuples(index=False):
                topic_id = int(getattr(row, "Topic"))
                if topic_id == -1:
                    continue
                representative_documents = self.topic_model.get_representative_docs(
                    topic_id
                ) or []
                results.append(
                    {
                        "topic_id": topic_id,
                        "topic_name": str(getattr(row, "Name", topic_id)),
                        "document_count": int(getattr(row, "Count", 0)),
                        "representative_documents": [
                            str(document) for document in representative_documents
                        ],
                    }
                )
                if len(results) == 10:
                    break
            return results
        except (TypeError, ValueError, RuntimeError, KeyError) as error:
            logger.exception("Unable to extract topics: %s", error)
            raise

    def get_trend_momentum(self, topic_id: int) -> float:
        """Calculate annualized topic-share growth from timestamped data.

        Momentum is the least-squares slope of the topic's document share over
        elapsed days, annualized by multiplying by 365. Positive values mean
        the topic is becoming more prevalent. If no timestamps were supplied,
        or all timestamps are identical, the neutral result is ``0.0``.

        Args:
            topic_id: BERTopic topic identifier.

        Returns:
            Annualized change in topic share.

        Raises:
            RuntimeError: If topics have not been extracted yet.
            ValueError: If ``topic_id`` is not present in the fitted results or
                timestamps contain invalid values.
        """
        try:
            if self.topic_assignments is None:
                raise RuntimeError("Extract topics before calculating momentum")
            if topic_id not in set(self.topic_assignments.tolist()):
                raise ValueError(f"Unknown topic ID: {topic_id}")
            if not self.valid_timestamps:
                return 0.0
            dates = pd.to_datetime(self.valid_timestamps, errors="coerce", utc=True)
            if dates.isna().any():
                raise ValueError("timestamps contain invalid values")
            elapsed_days = (dates - dates.min()).total_seconds() / 86400.0
            if float(elapsed_days.max()) == 0.0:
                return 0.0
            topic_share = (self.topic_assignments == topic_id).astype(float)
            centered_days = elapsed_days - elapsed_days.mean()
            denominator = float(np.dot(centered_days, centered_days))
            if denominator == 0.0:
                return 0.0
            slope_per_day = float(
                np.dot(centered_days, topic_share - topic_share.mean()) / denominator
            )
            return slope_per_day * 365.0
        except (RuntimeError, ValueError, TypeError) as error:
            logger.exception("Unable to calculate topic momentum: %s", error)
            raise

    def _reset_fit_state(self) -> None:
        """Clear fitted topic state after empty input."""
        self.valid_descriptions = []
        self.valid_timestamps = None
        self.topic_assignments = None