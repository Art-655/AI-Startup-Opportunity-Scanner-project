"""RAG-powered startup business-plan generation."""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Any, List, Sequence, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class GenAIAdvisor:
    """Use Gemini Flash and ChromaDB to generate failure-aware plans."""

    def __init__(
        self,
        persist_directory: PathLike = "data/chroma",
        collection_name: str = "startup_failure_postmortems",
        model_name: str = "models/gemini-3.6-flash",
        temperature: float = 0.2,
    ) -> None:
        """Initialize the LangChain embeddings, vector store, and chat model.

        Args:
            persist_directory: Local ChromaDB persistence directory.
            collection_name: Chroma collection for failure postmortems.
            model_name: Google Gemini chat model name.
            temperature: LLM response temperature.

        Raises:
            ImportError: If the required LangChain integrations are unavailable.
            ValueError: If configuration values are invalid.
        """
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.model_name = model_name
        self.temperature = temperature
        try:
            self._ensure_event_loop()
            ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, Chroma, PromptTemplate, StrOutputParser = (
                self._load_langchain_components()
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            self.vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
            )
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
            self.prompt_template = PromptTemplate.from_template(
                self._prompt_text()
            )
            self.output_parser = StrOutputParser()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            logger.exception("Unable to initialize GenAI advisor: %s", error)
            raise

    def ingest_failure_postmortems(self, texts: List[str]) -> int:
        """Embed and persist historical startup failure postmortems.

        Empty and non-string entries are ignored. Each accepted postmortem is
        stored as one Chroma document so retrieval can return the exact source
        text used to guide the generated plan.

        Args:
            texts: Historical failure descriptions.

        Returns:
            Number of postmortems added to ChromaDB.

        Raises:
            TypeError: If ``texts`` is not a list.
            RuntimeError: If ChromaDB cannot persist the documents.
        """
        try:
            if not isinstance(texts, list):
                raise TypeError("texts must be a list of strings")
            valid_texts = []
            for index, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    logger.warning("Skipping invalid postmortem at index %d", index)
                    continue
                valid_texts.append(text.strip())
            if not valid_texts:
                return 0
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            existing_count = self._document_count()
            if existing_count > 0:
                logger.info(
                    "Skipping postmortem ingestion; Chroma collection already has %d documents",
                    existing_count,
                )
                return 0
            self.vectorstore.add_texts(
                valid_texts,
                metadatas=[{"source_type": "startup_failure_postmortem"}]
                * len(valid_texts),
            )
            logger.info("Ingested %d failure postmortems", len(valid_texts))
            return len(valid_texts)
        except (TypeError, OSError, RuntimeError, ValueError) as error:
            logger.exception("Unable to ingest failure postmortems: %s", error)
            raise

    def _document_count(self) -> int:
        """Return the persisted Chroma document count when available."""
        try:
            collection = getattr(self.vectorstore, "_collection", None)
            return int(collection.count()) if collection is not None else 0
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("Unable to inspect Chroma collection: %s", error)
            return 0

    def generate_business_plan(
        self,
        industry: str,
        opportunity_score: float,
        trends: Union[str, Sequence[str]],
    ) -> str:
        """Generate a Markdown plan grounded in the three nearest failures.

        Args:
            industry: Startup industry for the retrieval query and plan.
            opportunity_score: Opportunity score on a 0-to-10 scale.
            trends: Relevant trend descriptions or a single trend string.

        Returns:
            A structured Markdown business plan.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If retrieval or generation fails.
        """
        try:
            if not isinstance(industry, str) or not industry.strip():
                raise ValueError("industry must be a non-empty string")
            if not isinstance(opportunity_score, (int, float)) or not math.isfinite(
                float(opportunity_score)
            ):
                raise ValueError("opportunity_score must be a finite number")
            if not 0 <= float(opportunity_score) <= 10:
                raise ValueError("opportunity_score must be between 0 and 10")
            trend_text = self._format_trends(trends)
            documents = self.vectorstore.similarity_search(industry.strip(), k=3)
            failure_text = self._format_failures(documents)
            chain = self.prompt_template | self.llm | self.output_parser
            result = chain.invoke(
                {
                    "industry": industry.strip(),
                    "opportunity_score": f"{float(opportunity_score):.2f}",
                    "trends": trend_text,
                    "failure_reasons": failure_text,
                }
            )
            if not isinstance(result, str) or not result.strip():
                raise RuntimeError("The language model returned an empty plan")
            return result.strip()
        except (TypeError, ValueError, RuntimeError) as error:
            logger.exception("Unable to generate business plan: %s", error)
            raise

    @staticmethod
    def _ensure_event_loop() -> None:
        """Create a loop for Streamlit worker threads when none is present."""
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    @staticmethod
    def _load_langchain_components() -> Any:
        """Load current LangChain integrations with a Chroma compatibility path."""
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )
        except ImportError as error:
            raise ImportError(
                "Install langchain-core, langchain-google-genai, and a Chroma integration"
            ) from error
        try:
            from langchain_chroma import Chroma
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError as error:
                raise ImportError(
                    "Install langchain-chroma or langchain-community"
                ) from error
        return (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
            Chroma,
            PromptTemplate,
            StrOutputParser,
        )

    @staticmethod
    def _format_trends(trends: Union[str, Sequence[str]]) -> str:
        """Format trend context for the prompt."""
        if isinstance(trends, str):
            if not trends.strip():
                raise ValueError("trends must not be empty")
            return trends.strip()
        if not isinstance(trends, Sequence):
            raise TypeError("trends must be a string or sequence of strings")
        valid_trends = [str(trend).strip() for trend in trends if str(trend).strip()]
        if not valid_trends:
            raise ValueError("trends must contain at least one value")
        return "\n".join(f"- {trend}" for trend in valid_trends)

    @staticmethod
    def _format_failures(documents: Sequence[Any]) -> str:
        """Format retrieved postmortems while preserving their exact text."""
        if not documents:
            return "No matching historical failure postmortems were found."
        return "\n\n".join(
            f"{index}. {document.page_content.strip()}"
            for index, document in enumerate(documents[:3], start=1)
        )

    @staticmethod
    def _prompt_text() -> str:
        """Return the constrained business-plan prompt template."""
        return """You are a rigorous startup strategy advisor.

Create a practical business plan for the {industry} industry.
Opportunity score: {opportunity_score}/10
Current trends:
{trends}

The following are the three most relevant historical startup failure reasons:
{failure_reasons}

Your plan must explicitly mitigate every listed failure reason. Do not merely
mention the failures: attach each one to a concrete product, operating,
financial, or go-to-market control and explain how the control will be measured.
Do not invent facts about the historical failures beyond the supplied text.

Return only structured Markdown using exactly these sections:
## Executive Summary
## Customer And Problem
## Product And Differentiation
## Failure Mitigation Plan
## Go-To-Market Strategy
## Business Model And Financial Controls
## Milestones And Metrics
## Key Risks And Contingencies
"""