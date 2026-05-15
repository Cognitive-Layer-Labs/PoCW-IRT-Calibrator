"""Feature engineering for IRT difficulty prediction from question text."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
import spacy
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

NEGATION_TERMS = {
    "not",
    "no",
    "none",
    "never",
    "neither",
    "nor",
    "without",
    "except",
    "least",
    "cannot",
    "cant",
    "can't",
    "dont",
    "don't",
    "isnt",
    "isn't",
    "aren't",
    "arent",
    "won't",
    "wont",
}

SUBORDINATE_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl", "mark"}


def _safe_float(value: float | int | None) -> float:
    """Convert a numeric value to float and coerce invalid values to 0.0."""
    if value is None:
        return 0.0
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _tokenize_words(text: str) -> list[str]:
    """Tokenize words with NLTK, then fallback to regex if resources are absent."""
    try:
        tokens = word_tokenize(text)
        return [token for token in tokens if any(char.isalpha() for char in token)]
    except LookupError:
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def _tokenize_sentences(text: str) -> list[str]:
    """Tokenize sentences with NLTK, then fallback to regex splitting."""
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    if sentences:
        return sentences
    stripped = text.strip()
    return [stripped] if stripped else []


def _ensure_text_list(X: Iterable[str]) -> list[str]:  # noqa: N803
    """Normalize input iterable into a clean list of strings."""
    return ["" if text is None else str(text) for text in X]


def _load_stop_words() -> set[str]:
    """Load NLTK stop words with a lightweight fallback set."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return set(FALLBACK_STOPWORDS)


def _dependency_depth(token, cache: dict[int, int]) -> int:
    """Compute distance from a token to the dependency tree root."""
    if token.i in cache:
        return cache[token.i]

    depth = 0
    current = token
    visited = set()
    while current.head.i != current.i and current.i not in visited:
        visited.add(current.i)
        depth += 1
        current = current.head

    cache[token.i] = depth
    return depth


class QuestionFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract linguistic and readability features from question text."""

    FEATURE_COLUMNS = [
        "word_count",
        "char_count",
        "sentence_count",
        "avg_word_length",
        "avg_sentence_length",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "type_token_ratio",
        "long_word_ratio",
        "stopword_ratio",
        "punctuation_count",
        "question_mark_count",
        "named_entity_count",
        "has_negation",
        "noun_count",
        "verb_count",
        "adj_count",
        "adv_count",
        "pron_count",
        "num_count",
        "aux_count",
        "subordinate_clause_count",
        "avg_dependency_depth",
        "max_dependency_depth",
    ]

    def __init__(self, spacy_model: str = "en_core_web_sm", batch_size: int = 64) -> None:
        self.spacy_model = spacy_model
        self.batch_size = batch_size
        self._nlp = None
        self._stop_words = _load_stop_words()

    def _get_nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError as exc:
                raise RuntimeError(
                    "spaCy model not found. Install it with: "
                    "python -m spacy download en_core_web_sm"
                ) from exc
        return self._nlp

    def fit(self, X: Iterable[str], y=None):  # noqa: N803
        self.feature_names_ = np.array(self.FEATURE_COLUMNS, dtype=object)
        self._get_nlp()
        return self

    def transform(self, X: Iterable[str]) -> pd.DataFrame:  # noqa: N803
        texts = _ensure_text_list(X)
        nlp = self._get_nlp()
        docs = nlp.pipe(texts, batch_size=self.batch_size)
        rows = [self._extract_one(text, doc) for text, doc in zip(texts, docs)]
        features = pd.DataFrame(rows, columns=self.FEATURE_COLUMNS)
        return features.astype(float)

    def _extract_one(self, text: str, doc) -> dict[str, float]:
        words = _tokenize_words(text)
        words_lower = [word.lower() for word in words]
        sentences = _tokenize_sentences(text)

        word_count = len(words)
        char_count = len(text)
        sentence_count = max(1, len(sentences)) if text.strip() else 0
        avg_word_length = (
            float(np.mean([len(word) for word in words])) if words else 0.0
        )
        avg_sentence_length = (
            float(word_count / sentence_count) if sentence_count > 0 else 0.0
        )

        unique_words = set(words_lower)
        type_token_ratio = (
            float(len(unique_words) / word_count) if word_count > 0 else 0.0
        )
        long_word_ratio = (
            float(sum(len(word) >= 7 for word in words) / word_count)
            if word_count > 0
            else 0.0
        )
        stopword_ratio = (
            float(sum(word in self._stop_words for word in words_lower) / word_count)
            if word_count > 0
            else 0.0
        )

        punctuation_count = len(re.findall(r"[^\w\s]", text))
        question_mark_count = text.count("?")
        has_negation = float(any(word in NEGATION_TERMS for word in words_lower))

        pos_counter = Counter(
            token.pos_
            for token in doc
            if not token.is_space and not token.is_punct
        )
        noun_count = pos_counter.get("NOUN", 0) + pos_counter.get("PROPN", 0)
        verb_count = pos_counter.get("VERB", 0)
        adj_count = pos_counter.get("ADJ", 0)
        adv_count = pos_counter.get("ADV", 0)
        pron_count = pos_counter.get("PRON", 0)
        num_count = pos_counter.get("NUM", 0)
        aux_count = pos_counter.get("AUX", 0)
        subordinate_clause_count = sum(
            token.dep_ in SUBORDINATE_DEPS for token in doc if not token.is_punct
        )

        named_entity_count = len(doc.ents)

        depth_cache: dict[int, int] = {}
        dependency_depths = [
            _dependency_depth(token, depth_cache)
            for token in doc
            if not token.is_space and not token.is_punct
        ]
        avg_dependency_depth = (
            float(np.mean(dependency_depths)) if dependency_depths else 0.0
        )
        max_dependency_depth = float(max(dependency_depths)) if dependency_depths else 0.0

        flesch_reading_ease = _safe_float(textstat.flesch_reading_ease(text))
        flesch_kincaid_grade = _safe_float(textstat.flesch_kincaid_grade(text))

        return {
            "word_count": float(word_count),
            "char_count": float(char_count),
            "sentence_count": float(sentence_count),
            "avg_word_length": float(avg_word_length),
            "avg_sentence_length": float(avg_sentence_length),
            "flesch_reading_ease": float(flesch_reading_ease),
            "flesch_kincaid_grade": float(flesch_kincaid_grade),
            "type_token_ratio": float(type_token_ratio),
            "long_word_ratio": float(long_word_ratio),
            "stopword_ratio": float(stopword_ratio),
            "punctuation_count": float(punctuation_count),
            "question_mark_count": float(question_mark_count),
            "named_entity_count": float(named_entity_count),
            "has_negation": float(has_negation),
            "noun_count": float(noun_count),
            "verb_count": float(verb_count),
            "adj_count": float(adj_count),
            "adv_count": float(adv_count),
            "pron_count": float(pron_count),
            "num_count": float(num_count),
            "aux_count": float(aux_count),
            "subordinate_clause_count": float(subordinate_clause_count),
            "avg_dependency_depth": float(avg_dependency_depth),
            "max_dependency_depth": float(max_dependency_depth),
        }

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        return np.array(self.FEATURE_COLUMNS, dtype=object)


class TfidfSvdEmbeddingExtractor(BaseEstimator, TransformerMixin):
    """Build dense text embeddings from TF-IDF + TruncatedSVD."""

    def __init__(
        self,
        embedding_dim: int = 256,
        max_features: int = 50000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_min: int = 1,
        ngram_max: int = 2,
        random_state: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.random_state = random_state
        self._vectorizer = None
        self._svd = None
        self.embedding_dim_ = None
        self.feature_names_ = None

    def fit(self, X: Iterable[str], y=None):  # noqa: N803
        texts = _ensure_text_list(X)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(self.ngram_min, self.ngram_max),
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts)
        if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] < 2:
            raise ValueError(
                "TF-IDF embedding extractor needs at least 2 rows and 2 unique terms."
            )
        max_components = min(
            self.embedding_dim,
            tfidf_matrix.shape[0] - 1,
            tfidf_matrix.shape[1] - 1,
        )
        max_components = max(1, int(max_components))
        self._svd = TruncatedSVD(n_components=max_components, random_state=self.random_state)
        self._svd.fit(tfidf_matrix)
        self.embedding_dim_ = max_components
        self.feature_names_ = np.array(
            [f"embedding_{idx}" for idx in range(self.embedding_dim_)], dtype=object
        )
        return self

    def transform(self, X: Iterable[str]) -> pd.DataFrame:  # noqa: N803
        if self._vectorizer is None or self._svd is None or self.feature_names_ is None:
            raise RuntimeError("TfidfSvdEmbeddingExtractor must be fitted before transform.")
        texts = _ensure_text_list(X)
        tfidf_matrix = self._vectorizer.transform(texts)
        embeddings = self._svd.transform(tfidf_matrix)
        return pd.DataFrame(embeddings, columns=self.feature_names_).astype(float)

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        if self.feature_names_ is None:
            raise RuntimeError("TfidfSvdEmbeddingExtractor has no fitted feature names.")
        return self.feature_names_


class TransformerEmbeddingExtractor(BaseEstimator, TransformerMixin):
    """Build dense embeddings using a Hugging Face transformer model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_length: int = 256,
        device: str = "auto",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.normalize = normalize

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self.embedding_dim_ = None
        self.feature_names_ = None

    @staticmethod
    def _import_dependencies():
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transformer embeddings require PyTorch. Install it with: pip install torch"
            ) from exc
        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transformer embeddings require transformers. Install with: pip install transformers"
            ) from exc
        return torch, AutoTokenizer, AutoModel

    def _resolve_device(self, torch_module) -> str:
        if self.device != "auto":
            return self.device
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        torch_module, auto_tokenizer, auto_model = self._import_dependencies()
        resolved_device = self._resolve_device(torch_module)
        tokenizer = auto_tokenizer.from_pretrained(self.model_name)
        model = auto_model.from_pretrained(self.model_name)
        model.to(resolved_device)
        model.eval()

        self._torch = torch_module
        self._tokenizer = tokenizer
        self._model = model
        self._device = resolved_device
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        if hidden_size <= 0:
            raise ValueError(
                f"Could not infer embedding size from model config for '{self.model_name}'."
            )
        self.embedding_dim_ = hidden_size
        self.feature_names_ = np.array(
            [f"embedding_{idx}" for idx in range(self.embedding_dim_)], dtype=object
        )

    def fit(self, X: Iterable[str], y=None):  # noqa: N803
        self._ensure_model_loaded()
        return self

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def transform(self, X: Iterable[str]) -> pd.DataFrame:  # noqa: N803
        self._ensure_model_loaded()
        texts = _ensure_text_list(X)
        if not texts:
            return pd.DataFrame(columns=self.feature_names_, dtype=float)

        torch_module = self._torch
        embeddings = []
        for idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[idx : idx + self.batch_size]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch_module.no_grad():
                outputs = self._model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                if self.normalize:
                    pooled = torch_module.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
        matrix = np.vstack(embeddings)
        return pd.DataFrame(matrix, columns=self.feature_names_).astype(float)

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        if self.feature_names_ is None:
            raise RuntimeError("TransformerEmbeddingExtractor has no fitted feature names.")
        return self.feature_names_


class HybridFeatureExtractor(BaseEstimator, TransformerMixin):
    """Concatenate linguistic and embedding-based features."""

    def __init__(
        self,
        linguistic_extractor: QuestionFeatureExtractor | None = None,
        embedding_extractor: BaseEstimator | None = None,
    ) -> None:
        self.linguistic_extractor = linguistic_extractor or QuestionFeatureExtractor()
        self.embedding_extractor = embedding_extractor or TfidfSvdEmbeddingExtractor()
        self.feature_names_ = None

    def fit(self, X: Iterable[str], y=None):  # noqa: N803
        self.linguistic_extractor.fit(X, y)
        self.embedding_extractor.fit(X, y)
        feature_names = list(self.linguistic_extractor.get_feature_names_out()) + list(
            self.embedding_extractor.get_feature_names_out()
        )
        self.feature_names_ = np.array(feature_names, dtype=object)
        return self

    def transform(self, X: Iterable[str]) -> pd.DataFrame:  # noqa: N803
        linguistic = self.linguistic_extractor.transform(X).reset_index(drop=True)
        embedding = self.embedding_extractor.transform(X).reset_index(drop=True)
        return pd.concat([linguistic, embedding], axis=1).astype(float)

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        if self.feature_names_ is None:
            raise RuntimeError("HybridFeatureExtractor has no fitted feature names.")
        return self.feature_names_
