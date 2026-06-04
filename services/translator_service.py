from deep_translator import GoogleTranslator


class TranslatorService:
    """Translates queries and text between Hindi and English with an in-memory cache."""

    cache = {}

    @classmethod
    def translate_query(cls, query: str) -> str:
        if query in cls.cache:
            return cls.cache[query]

        try:
            translated = GoogleTranslator(source="en", target="hi").translate(query)
            cls.cache[query] = translated
            return translated
        except Exception:
            return query

    @classmethod
    def translate_hi_to_en(cls, text: str) -> str:
        if not text:
            return ""
        
        cache_key = f"hi_to_en:{text}"
        if cache_key in cls.cache:
            return cls.cache[cache_key]

        try:
            translated = GoogleTranslator(source="hi", target="en").translate(text)
            cls.cache[cache_key] = translated
            return translated
        except Exception:
            return text