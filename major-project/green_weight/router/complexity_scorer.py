"""
Complexity Scorer - Phase 3–4 Bridge
====================================

Purpose: Convert a raw prompt string into a set of normalized numeric features
that the fuzzy controller can consume. This is the "sensor" layer.

What it does:
- Extracts five normalized float features (0–1 range):
  1. flesch_kincaid: Readability grade normalized to 0–1
  2. token_length: Prompt token count normalized against max of 512
  3. entropy: Character-level Shannon entropy
  4. syntax_depth: Maximum parse tree depth from spaCy, normalized
  5. has_code_or_math: Binary 0 or 1 for code blocks, LaTeX, or math operators
- Caches the spaCy model as singleton to avoid reloading
- Uses textstat for readability, spaCy for parse depth, entropy for Shannon entropy
"""

import math
import re
from typing import Dict
import logging

try:
    import textstat
except ImportError:
    textstat = None

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)

# Module-level singleton for spaCy model
_spacy_model = None


def _get_spacy_model():
    """Load and cache spaCy model (singleton)."""
    global _spacy_model
    
    if _spacy_model is None:
        if spacy is None:
            raise ImportError("spacy not installed. Install with: pip install spacy")
        
        model_name = "en_core_web_sm"
        try:
            _spacy_model = spacy.load(model_name)
            logger.info(f"[OK] Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(
                f"spaCy model '{model_name}' not found. Download with:\n"
                f"  python -m spacy download {model_name}"
            )
            raise
    
    return _spacy_model


def shannon_entropy(text: str) -> float:
    """
    Compute character-level Shannon entropy of a string.
    
    Args:
        text: Input string
    
    Returns:
        Entropy in bits (higher = more random/complex)
    """
    if not text:
        return 0.0
    
    # Count character frequencies
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Compute entropy
    entropy_bits = 0.0
    text_len = len(text)
    for count in char_counts.values():
        p = count / text_len
        entropy_bits -= p * math.log2(p)
    
    return entropy_bits


def get_parse_depth(text: str) -> int:
    """
    Compute maximum parse tree depth using spaCy dependency parser.
    
    Args:
        text: Input text
    
    Returns:
        Maximum depth of dependency tree
    """
    if spacy is None:
        logger.warning("spaCy not available; returning default parse depth")
        return 5
    
    nlp = _get_spacy_model()
    doc = nlp(text)
    
    def get_depth(token, depth=0):
        max_d = depth
        for child in token.children:
            max_d = max(max_d, get_depth(child, depth + 1))
        return max_d
    
    # Root tokens (where head == token)
    max_depth = 0
    for token in doc:
        if token.head == token:  # Root
            depth = get_depth(token)
            max_depth = max(max_depth, depth)
    
    return max_depth


def has_code_or_math(text: str) -> bool:
    """
    Detect whether text contains code blocks, LaTeX, or mathematical operators.
    
    Args:
        text: Input text
    
    Returns:
        True if code/math detected, False otherwise
    """
    # Patterns for code blocks (markdown, fenced, inline)
    code_patterns = [
        r'```.*?```',  # Markdown code blocks
        r'~~~ .*?~~~',  # Tilde fenced blocks
        r'`[^`]+`',  # Inline code
    ]
    
    # Patterns for LaTeX
    latex_patterns = [
        r'\$\$.*?\$\$',  # Display math
        r'\$[^\$]+\$',  # Inline math
        r'\\[a-zA-Z]+',  # LaTeX commands like \alpha, \sum
    ]
    
    # Math operators
    math_operators = [
        r'∑',  # Sum
        r'∫',  # Integral
        r'∂',  # Partial derivative
        r'√',  # Square root
        r'∞',  # Infinity
        r'≈|≠|≤|≥|∈|∉',  # Math relations
    ]
    
    # Check for matches
    for pattern in code_patterns + latex_patterns + math_operators:
        if re.search(pattern, text, re.DOTALL):
            return True
    
    return False


def normalize_to_01(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to [0, 1] range.
    
    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
    
    Returns:
        Normalized value clipped to [0, 1]
    """
    if max_val <= min_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def score(prompt: str) -> Dict[str, float]:
    """
    Score a prompt and return normalized complexity features.
    
    Args:
        prompt: Input prompt text
    
    Returns:
        Dict with keys: flesch_kincaid, token_length, entropy, syntax_depth, has_code_or_math
        All values are floats in [0, 1] range (except has_code_or_math which is 0 or 1).
    """
    features = {}
    
    # 1. Flesch-Kincaid readability (grade level, 0–18 scale)
    if textstat is None:
        logger.warning("textstat not available; skipping readability score")
        features["flesch_kincaid"] = 0.5
    else:
        try:
            grade_level = textstat.flesch_kincaid_grade(prompt)
            # Normalize 0–18 scale to 0–1
            features["flesch_kincaid"] = normalize_to_01(grade_level, 0, 18)
        except Exception as e:
            logger.warning(f"Flesch-Kincaid calculation failed: {e}")
            features["flesch_kincaid"] = 0.5
    
    # 2. Token length (prompt token count, normalized against 512-token max)
    # Simple approximation: 1 token ≈ 4 characters for English
    approx_tokens = len(prompt) / 4.0
    features["token_length"] = normalize_to_01(approx_tokens, 0, 512)
    
    # 3. Shannon entropy (bits, typically 2–8 range for natural language)
    entropy_bits = shannon_entropy(prompt)
    features["entropy"] = normalize_to_01(entropy_bits, 2.0, 8.0)
    
    # 4. Syntax parse depth (max depth of dependency tree, typically 3–20)
    try:
        depth = get_parse_depth(prompt)
        features["syntax_depth"] = normalize_to_01(depth, 1, 20)
    except Exception as e:
        logger.warning(f"Parse depth calculation failed: {e}")
        features["syntax_depth"] = 0.5
    
    # 5. Code or math detection (binary: 0 or 1)
    features["has_code_or_math"] = 1.0 if has_code_or_math(prompt) else 0.0
    
    return features


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)
    
    test_prompts = [
        "What is 2 + 2?",  # Simple
        "Write a Python function that sorts a list using quicksort.",  # Code
        "What is ∫x² dx from 0 to 1?",  # Math
        "Explain quantum computing in detail, including superposition, entanglement, and quantum gates.",  # Complex
    ]
    
    for prompt in test_prompts:
        features = score(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        for key, val in features.items():
            print(f"  {key}: {val:.3f}")
