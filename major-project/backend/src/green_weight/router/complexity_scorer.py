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

# Single source of truth for feature normalization bounds — also imported by
# fuzzy_controller.py so that config.yaml breakpoints get normalized with
# the exact same (min, max) as the features they're compared against.
# Re-calibrated 2026-08-13 against the real 500-prompt eval set
# (data/eval_prompts.jsonl, built by prepare_eval_dataset.py). Superscedes
# the earlier 2026-08-05 calibration, which was against a 30-prompt
# sample and clipped real data at both tails: observed entropy on the
# full set is 3.37-4.66 bits (old range topped out at 4.5, clipping the
# upper tail); observed syntax_depth is 2-12 (old range topped out at 8,
# clipping ~5% of prompts — exactly the hardest ones — to a flat 1.0).
# NOTE: per-difficulty analysis on this same 500-prompt set shows entropy
# barely discriminates prompt difficulty (easy mean=4.11, medium=3.96,
# hard=4.11 bits — easy and hard are nearly identical); syntax_depth does
# discriminate (easy=4.42, medium=4.46, hard=5.57). Widening this range
# fixes clipping but does not fix entropy's weak signal — that's a
# candidate for the Phase 5c feature-ablation study, not a calibration fix.
FLESCH_KINCAID_RANGE = (0, 18)
ENTROPY_RANGE = (3.3, 5.0)
SYNTAX_DEPTH_RANGE = (2, 14)

# Re-calibrated 2026-08-22, same 500-prompt eval set. The prior hardcoded
# cap of 512 tokens was never based on real data: observed approx_tokens
# (len(prompt)/4) on the full set is 5.25-153.75, p95=62. Under the old
# cap, the MID breakpoint (0.2 -> 102 tokens) was reachable by only ~1% of
# prompts and the HIGH breakpoint (0.8 -> 410 tokens) was unreachable by
# any real prompt — token_length was structurally near-dead as a routing
# signal. Ceiling rounds the observed max (153.75) up to 154 so the single
# longest real prompt doesn't sit exactly at the 1.0 clip boundary.
TOKEN_LENGTH_RANGE = (0, 154)

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
    
    # Check for code/math syntax matches
    for pattern in code_patterns + latex_patterns + math_operators:
        if re.search(pattern, text, re.DOTALL):
            return True

    # Also detect natural-language requests for code/algorithms
    code_request_keywords = [
        r'\bwrite\s+a\b',
        r'\bimplement\b',
        r'\bimplementation\b',
        r'\balgorithm\b',
        r'\bfunction\b',
        r'\bclass\b',
        r'\bprogram\b',
        r'\bscript\b',
        r'\bcode\b',
        r'\bsort\b',
        r'\brecursion\b',
        r'\bdynamic\s+programming\b',
    ]
    text_lower = text.lower()
    for kw in code_request_keywords:
        if re.search(kw, text_lower):
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
            features["flesch_kincaid"] = normalize_to_01(grade_level, *FLESCH_KINCAID_RANGE)
        except Exception as e:
            logger.warning(f"Flesch-Kincaid calculation failed: {e}")
            features["flesch_kincaid"] = 0.5
    
    # 2. Token length (prompt token count, normalized against real observed
    # max — see TOKEN_LENGTH_RANGE comment above).
    # Simple approximation: 1 token ≈ 4 characters for English
    approx_tokens = len(prompt) / 4.0
    features["token_length"] = normalize_to_01(approx_tokens, *TOKEN_LENGTH_RANGE)
    
    # 3. Shannon entropy (bits). Re-calibrated 2026-08-13 against the real
    # 500-prompt eval set: observed range 3.37-4.66 bits. See ENTROPY_RANGE
    # comment above — this feature barely discriminates prompt difficulty
    # in this dataset (easy/hard means are nearly identical); widening the
    # range fixes upper-tail clipping but the weak-signal issue itself
    # needs the Phase 5c feature-ablation study, not a range tweak.
    entropy_bits = shannon_entropy(prompt)
    features["entropy"] = normalize_to_01(entropy_bits, *ENTROPY_RANGE)

    # 4. Syntax parse depth (max depth of dependency tree). Re-calibrated
    # 2026-08-13 against the real 500-prompt eval set: observed range 2-12
    # (old range topped out at 8, clipping ~5% of prompts — the hardest
    # ones — to a flat 1.0). Unlike entropy, this feature does show real
    # separation by difficulty (hard prompts noticeably deeper).
    try:
        depth = get_parse_depth(prompt)
        features["syntax_depth"] = normalize_to_01(depth, *SYNTAX_DEPTH_RANGE)
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
