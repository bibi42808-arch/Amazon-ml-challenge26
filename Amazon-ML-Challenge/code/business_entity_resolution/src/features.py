"""Efficient, local-only similarity features for business entity pairs."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Any


NAME_STOPWORDS = {
    "and", "associates", "business", "company", "co", "corp", "corporation",
    "group", "inc", "incorporated", "international", "limited", "llc", "llp",
    "ltd", "of", "private", "public", "services", "solutions", "the", "trading",
}
ADDRESS_STOPWORDS = {
    "apartment", "avenue", "block", "building", "city", "district", "floor",
    "near", "opposite", "road", "sector", "street", "suite", "town", "tower", "unit",
}
BLOCK_METHODS = (
    "country_exact_name",
    "exact_name",
    "country_exact_address",
    "country_name_token",
    "country_address_token",
)
FEATURE_NAMES = (
    "name_exact",
    "name_char_dice",
    "name_token_jaccard",
    "name_token_overlap_min",
    "name_token_containment",
    "name_informative_overlap",
    "name_prefix_ratio",
    "name_suffix_ratio",
    "name_length_difference",
    "name_length_ratio",
    "name_missing_left",
    "name_missing_right",
    "address_exact",
    "address_char_dice",
    "address_token_jaccard",
    "address_token_overlap_min",
    "address_token_containment",
    "address_digit_overlap",
    "address_long_number_agreement",
    "address_prefix_ratio",
    "address_suffix_ratio",
    "address_length_difference",
    "address_length_ratio",
    "address_missing_left",
    "address_missing_right",
    "country_exact",
    "country_missing_left",
    "country_missing_right",
    "block_country_exact_name",
    "block_exact_name",
    "block_country_exact_address",
    "block_country_name_token",
    "block_country_address_token",
    "candidate_count_log1p",
)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"na", "n/a", "null", "none", "nil", "unknown", "-"} else text


def unicode_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    current: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or (category[0] == "M" and current):
            current.append(character)
        elif current:
            tokens.add("".join(current))
            current.clear()
    if current:
        tokens.add("".join(current))
    return tokens


def character_ngrams(text: str, size: int = 2) -> Counter[str]:
    compact = " ".join(text.casefold().split())
    if not compact:
        return Counter()
    padded = f" {compact} "
    if len(padded) <= size:
        return Counter([padded])
    return Counter(padded[index : index + size] for index in range(len(padded) - size + 1))


def dice_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_grams = character_ngrams(left)
    right_grams = character_ngrams(right)
    denominator = sum(left_grams.values()) + sum(right_grams.values())
    if denominator == 0:
        return 0.0
    overlap = sum((left_grams & right_grams).values())
    return 2.0 * overlap / denominator


def ratio_features(left: str, right: str) -> tuple[float, float, float, float]:
    left_length, right_length = len(left), len(right)
    maximum = max(left_length, right_length)
    difference = abs(left_length - right_length) / max(1, maximum)
    ratio = min(left_length, right_length) / maximum if maximum else 1.0
    prefix = 0
    for first, second in zip(left, right):
        if first != second:
            break
        prefix += 1
    suffix = 0
    for first, second in zip(left[::-1], right[::-1]):
        if first != second:
            break
        suffix += 1
    return (
        prefix / maximum if maximum else 1.0,
        suffix / maximum if maximum else 1.0,
        difference,
        ratio,
    )


def token_features(
    left: str,
    right: str,
    stopwords: set[str],
) -> tuple[float, float, float, float]:
    left_tokens = unicode_tokens(left)
    right_tokens = unicode_tokens(right)
    union = left_tokens | right_tokens
    common = left_tokens & right_tokens
    jaccard = len(common) / len(union) if union else 0.0
    overlap_min = len(common) / min(len(left_tokens), len(right_tokens)) if left_tokens and right_tokens else 0.0
    containment = float(bool(left_tokens) and (left_tokens <= right_tokens or right_tokens <= left_tokens))
    informative_left = left_tokens - stopwords
    informative_right = right_tokens - stopwords
    informative_denominator = min(len(informative_left), len(informative_right))
    informative_overlap = (
        len(informative_left & informative_right) / informative_denominator
        if informative_denominator
        else 0.0
    )
    return jaccard, overlap_min, containment, informative_overlap


def digit_tokens(text: str) -> set[str]:
    return {token for token in unicode_tokens(text) if any(character.isdigit() for character in token)}


def similarity_features(
    source: dict[str, Any],
    candidate: dict[str, Any],
    blocking_method: str = "",
    candidate_count: int = 1,
) -> dict[str, float]:
    name_left = safe_text(source.get("business_name_normalized", ""))
    name_right = safe_text(candidate.get("business_name_normalized", ""))
    address_left = safe_text(source.get("business_address_normalized", ""))
    address_right = safe_text(candidate.get("business_address_normalized", ""))
    country_left = safe_text(source.get("country_normalized", ""))
    country_right = safe_text(candidate.get("country_normalized", ""))

    name_token_values = token_features(name_left, name_right, NAME_STOPWORDS)
    address_token_values = token_features(address_left, address_right, ADDRESS_STOPWORDS)
    digits_left = digit_tokens(address_left)
    digits_right = digit_tokens(address_right)
    digit_union = digits_left | digits_right
    digit_overlap = len(digits_left & digits_right) / len(digit_union) if digit_union else 0.0
    long_numbers_left = {token for token in digits_left if len(token) >= 4}
    long_numbers_right = {token for token in digits_right if len(token) >= 4}
    long_number_agreement = float(bool(long_numbers_left & long_numbers_right))
    name_prefix, name_suffix, name_length_difference, name_length_ratio = ratio_features(name_left, name_right)
    address_prefix, address_suffix, address_length_difference, address_length_ratio = ratio_features(address_left, address_right)

    result = {
        "name_exact": float(bool(name_left) and name_left == name_right),
        "name_char_dice": dice_similarity(name_left, name_right),
        "name_token_jaccard": name_token_values[0],
        "name_token_overlap_min": name_token_values[1],
        "name_token_containment": name_token_values[2],
        "name_informative_overlap": name_token_values[3],
        "name_prefix_ratio": name_prefix,
        "name_suffix_ratio": name_suffix,
        "name_length_difference": name_length_difference,
        "name_length_ratio": name_length_ratio,
        "name_missing_left": float(not name_left),
        "name_missing_right": float(not name_right),
        "address_exact": float(bool(address_left) and address_left == address_right),
        "address_char_dice": dice_similarity(address_left, address_right),
        "address_token_jaccard": address_token_values[0],
        "address_token_overlap_min": address_token_values[1],
        "address_token_containment": address_token_values[2],
        "address_digit_overlap": digit_overlap,
        "address_long_number_agreement": long_number_agreement,
        "address_prefix_ratio": address_prefix,
        "address_suffix_ratio": address_suffix,
        "address_length_difference": address_length_difference,
        "address_length_ratio": address_length_ratio,
        "address_missing_left": float(not address_left),
        "address_missing_right": float(not address_right),
        "country_exact": float(bool(country_left) and country_left == country_right),
        "country_missing_left": float(not country_left),
        "country_missing_right": float(not country_right),
        "block_country_exact_name": float("country_exact_name" in blocking_method),
        "block_exact_name": float("exact_name" in blocking_method.split(",")),
        "block_country_exact_address": float("country_exact_address" in blocking_method),
        "block_country_name_token": float("country_name_token" in blocking_method),
        "block_country_address_token": float("country_address_token" in blocking_method),
        "candidate_count_log1p": math.log1p(max(0, candidate_count)),
    }
    return result


def feature_vector(
    source: dict[str, Any],
    candidate: dict[str, Any],
    blocking_method: str = "",
    candidate_count: int = 1,
) -> list[float]:
    features = similarity_features(source, candidate, blocking_method, candidate_count)
    return [features[name] for name in FEATURE_NAMES]