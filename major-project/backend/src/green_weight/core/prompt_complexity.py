"""Prompt complexity scoring - The Sensor for the Energy Gearbox."""

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class ComplexityFeatures:
    """Raw features extracted from a prompt."""

    word_count: int
    sentence_count: int
    avg_word_length: float
    question_marks: int
    code_indicators: int
    math_symbols: int
    reasoning_keywords: int

    @property
    def flesch_kincaid_grade(self) -> float:
        """Approximate Flesch-Kincaid grade level."""
        if self.sentence_count == 0:
            return 0.0
        avg_sentence_length = self.word_count / self.sentence_count
        avg_syllables_per_word = self.avg_word_length * 0.5  # Rough approximation
        return (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59


class ComplexityScorer:
    """Analyzes prompt complexity to determine required compute."""

    # Keywords that indicate reasoning complexity
    REASONING_KEYWORDS = {
        "explain", "why", "how", "analyze", "compare", "contrast",
        "evaluate", "justify", "prove", "derive", "solve", "step",
        "reasoning", "logic", "therefore", "because", "consequence",
    }

    # Code-related indicators
    CODE_PATTERNS = [
        r"```[\s\S]*?```",  # Code blocks
        r"`[^`]+`",          # Inline code
        r"def\s+\w+",        # Function definitions
        r"class\s+\w+",      # Class definitions
        r"import\s+\w+",     # Imports
        r"for\s+\w+\s+in",   # Loops
        r"if\s+.*?:",         # Conditionals
    ]

    # Math symbols
    MATH_PATTERNS = [
        r"[\+\-\*\/\=\^\√\∑\∏\∫]",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",
        r"\b(equation|formula|calculate|compute|derivative|integral)\b",
    ]

    def __init__(self, max_score: float = 100.0):
        self.max_score = max_score
        self._code_regex = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]
        self._math_regex = [re.compile(p, re.IGNORECASE) for p in self.MATH_PATTERNS]

    def extract_features(self, prompt: str) -> ComplexityFeatures:
        """Extract raw features from the prompt."""
        text = prompt.strip()

        # Basic counts
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

        # Question complexity
        question_marks = text.count('?')

        # Code complexity
        code_indicators = sum(
            len(pattern.findall(text))
            for pattern in self._code_regex
        )

        # Math complexity
        math_symbols = sum(
            len(pattern.findall(text))
            for pattern in self._math_regex
        )

        # Reasoning complexity
        text_lower = text.lower()
        reasoning_keywords = sum(
            1 for keyword in self.REASONING_KEYWORDS
            if keyword in text_lower
        )

        return ComplexityFeatures(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            question_marks=question_marks,
            code_indicators=code_indicators,
            math_symbols=math_symbols,
            reasoning_keywords=reasoning_keywords,
        )

    def calculate_complexity(self, prompt: str) -> float:
        """
        Calculate overall complexity score (0-100).

        Higher score = more complex = needs higher bit-width.
        """
        features = self.extract_features(prompt)

        # Base score from Flesch-Kincaid (reading difficulty)
        fk_score = min(features.flesch_kincaid_grade * 2, 30)

        # Length factor (longer prompts tend to be more complex)
        length_score = min(features.word_count / 10, 25)

        # Code complexity (coding tasks need full precision)
        code_score = min(features.code_indicators * 10, 25)

        # Math complexity (math needs precision)
        math_score = min(features.math_symbols * 5, 15)

        # Reasoning complexity
        reasoning_score = min(features.reasoning_keywords * 5, 15)

        # Question complexity (questions often need more reasoning)
        question_score = min(features.question_marks * 5, 10)

        total = fk_score + length_score + code_score + math_score + reasoning_score + question_score
        return min(total, self.max_score)

    def get_complexity_label(self, score: float) -> str:
        """Get human-readable complexity label."""
        if score < 30:
            return "simple"
        elif score < 60:
            return "medium"
        return "complex"
