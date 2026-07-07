"""
translation.py
--------------

Utility functions for multilingual support.

Responsibilities
----------------
1. Detect the language of a user query.
2. Translate non-English queries into English.
3. Translate English answers back into the user's language.

This enables the RAG pipeline to always retrieve using English while
allowing users to ask questions in any supported language.
"""

from __future__ import annotations

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# ------------------------------------------------------------------
# Default Language
# ------------------------------------------------------------------

DEFAULT_LANGUAGE = "en"


# ------------------------------------------------------------------
# Detect Language
# ------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Returns:
        Language code (e.g. "en", "fr", "hi", "es")
    """

    try:
        return detect(text)

    except LangDetectException:
        return DEFAULT_LANGUAGE


# ------------------------------------------------------------------
# Translate
# ------------------------------------------------------------------

def translate(
    text: str,
    source: str,
    target: str,
) -> str:
    """
    Translate text from source language to target language.
    """

    if not text.strip():
        return text

    if source == target:
        return text

    try:

        return GoogleTranslator(
            source=source,
            target=target,
        ).translate(text)

    except Exception:

        # If translation fails, return original text
        return text


# ------------------------------------------------------------------
# Query Translation
# ------------------------------------------------------------------

def translate_query(
    question: str,
):
    """
    Detect the user's language.

    If the question is not English,
    translate it to English.

    Returns:
        translated_question,
        detected_language
    """

    language = detect_language(question)

    if language == DEFAULT_LANGUAGE:

        return question, language

    translated = translate(
        text=question,
        source=language,
        target="en",
    )

    return translated, language


# ------------------------------------------------------------------
# Answer Translation
# ------------------------------------------------------------------

def translate_answer(
    answer: str,
    language: str,
):
    """
    Translate an English answer back into
    the user's language.
    """

    if language == DEFAULT_LANGUAGE:
        return answer

    return translate(
        text=answer,
        source="en",
        target=language,
    )


# ------------------------------------------------------------------
# Complete Pipeline
# ------------------------------------------------------------------

def multilingual_pipeline(
    question: str,
):
    """
    Complete preprocessing pipeline.

    Input:
        User Question

    Output:
        English Question,
        Original Language
    """

    english_question, language = translate_query(question)

    return english_question, language


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":

    questions = [

        "How many casual leaves do employees receive?",

        "¿Cuántos días de permiso casual reciben los empleados?",

        "कर्मचारियों को कितनी आकस्मिक छुट्टियाँ मिलती हैं?",

        "Combien de congés occasionnels les employés reçoivent-ils ?",
    ]

    for question in questions:

        english, language = multilingual_pipeline(question)

        print("=" * 70)

        print("Original :", question)

        print("Detected :", language)

        print("English  :", english)

        translated = translate_answer(
            "Employees receive 12 casual leaves annually.",
            language,
        )

        print("Back Translation :", translated)