# indic_preprocessor.py
"""
Indic & Multilingual Preprocessing Engine for Toxic Comment Detection
Supports Hindi (Devanagari), Hinglish (Code-Mixed), Tamil, Telugu, Malayalam, Kannada, and Indian English.
"""

import re
import string
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional

# Unicode Script Ranges
UNICODE_RANGES = {
    "Hindi": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Punjabi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}

# Common Hinglish/Indic Obfuscation Map (masked profanity and leetspeak)
OBFUSCATION_MAP = {
    r"\bf\*+ck\b": "fuck",
    r"\bf\*+k\b": "fuck",
    r"\bb\*+ch\b": "bitch",
    r"\bb\*+stard\b": "bastard",
    r"\bb[\.\s_]*s[\.\s_]*d[\.\s_]*k\b": "bsdk",
    r"\bm[\.\s_]*c\b": "madarchod",
    r"\bb[\.\s_]*c\b": "behenchod",
    r"\bg[\*@]+nd\b": "gaand",
    r"\bb[@a]kwa+s\b": "bakwas",
    r"\bl[@o]du\b": "lodu",
    r"\bchut[i1!]y[a@]\b": "chutiya",
    r"\br[@a]nd[i1!]\b": "randi",
    r"\bm[@a]d[@a]rch[@o]d\b": "madarchod",
    r"\bbh[@a]dw[eey]\b": "bhadwe",
    r"\bmyr[e3]\b": "myre",
    r"\bthay[@o]l[i1]\b": "thayoli",
    r"\blanj[@a]k[o0]daka\b": "lanjakodaka",
    r"\bthevidiy[@a]\b": "thevidiya",
    r"\bsul[e3]magan[e3]\b": "sulemagane",
}

# Stopwords loader
_STOPWORDS_CACHE: Optional[Set[str]] = None

def get_indic_stopwords() -> Set[str]:
    """Load Indic + Hinglish + English stopwords."""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    stopwords = set()
    root = Path(__file__).parent
    stopwords_file = root / "data" / "stopwords_indic.txt"
    if stopwords_file.exists():
        with open(stopwords_file, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith("#"):
                    stopwords.add(word)

    # Standard English basic fillers (avoid removing negative sentiment words like not/no/never)
    english_fillers = {
        "the", "a", "an", "is", "are", "was", "were", "this", "that", "these",
        "those", "and", "or", "to", "for", "with", "in", "on", "at", "by", "from",
        "of", "it", "its", "you", "your", "we", "our", "he", "she", "they", "them",
        "my", "me", "i", "video", "channel", "please", "thanks", "thank"
    }
    stopwords.update(english_fillers)
    _STOPWORDS_CACHE = stopwords
    return _STOPWORDS_CACHE


def normalize_repeated_chars(text: str) -> str:
    """
    Compress character elongations (e.g. 'paaaagaaal' -> 'pagal', 'kuttaaaa' -> 'kutta').
    Replaces 3+ occurrences of any character with 2 (or 1 for vowels/special).
    """
    # Replace 3 or more consecutive identical characters with at most 2
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def deobfuscate_text(text: str) -> str:
    """Normalize common masked abusive terms and leetspeak in Indian context."""
    cleaned = text
    for pattern, replacement in OBFUSCATION_MAP.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def detect_indic_language(text: str) -> str:
    """
    Detect the primary Indian language or script of a given text.
    Returns: 'Hindi', 'Marathi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada', 'Bengali', 'Hinglish', 'Marathi (Romanized)', or 'Indian_English'.
    """
    if not text or not text.strip():
        return "Unknown"

    text_str = text

    # Count script characters
    counts = {lang: 0 for lang in UNICODE_RANGES}
    latin_count = 0

    for char in text_str:
        cp = ord(char)
        matched = False
        for lang, (start, end) in UNICODE_RANGES.items():
            if start <= cp <= end:
                counts[lang] += 1
                matched = True
                break
        if not matched and (('a' <= char <= 'z') or ('A' <= char <= 'Z')):
            latin_count += 1

    # Check Indic scripts
    max_indic_lang = max(counts, key=lambda k: counts[k])
    if counts[max_indic_lang] > 0 and counts[max_indic_lang] >= latin_count * 0.3:
        if max_indic_lang == "Hindi":
            # Differentiate Hindi from Marathi (both use Devanagari)
            marathi_devanagari_markers = {
                "आहे", "आहेत", "भावा", "नाही", "काय", "कसे", "करून", "शिकलो", "नव्हता", 
                "खूप", "छान", "भाऊ", "मित्रा", "तुझी", "माझी", "माझा", "तुझा", "होता", 
                "होती", "करत", "चाललंय", "मस्त", "करू", "करून", "मुलगा", "मुलगी"
            }
            words_devanagari = set(re.findall(r'[\u0900-\u097F]+', text_str))
            if words_devanagari & marathi_devanagari_markers:
                return "Marathi"
            return "Hindi"
        return max_indic_lang

    # If predominantly Latin script, check for Hinglish vs Dravidian Romanized vs English vs Romanized Marathi
    lower = text_str.lower()
    hinglish_markers = {
        "bhai", "yaar", "kya", "kyun", "kyu", "hai", "hain", "nahi", "nhi", "tera", "meri",
        "mera", "tere", "uska", "chup", "saale", "sale", "bc", "mc", "bsdk", "gaand", "lodu",
        "chutiya", "bakwas", "pagal", "bol", "raha", "rahi", "hoga", "karega", "aaya", "aayi",
        "kaise", "kaha", "kahan", "apna", "apni", "kuch", "bohot", "bahut", "achha", "accha"
    }
    marathi_roman_markers = {
        "aahe", "aahet", "aahes", "ekdam", "manus", "sarkha", "chya", "kay", "kasa", "kashi", 
        "navta", "sobat", "bhava", "khup", "chan", "tumhi", "amhi", "majha", "tujha", 
        "chalalay", "mitra", "bhau", "mhanje", "kuta", "nantar", "lavkar", "udya", "aaj", 
        "karan", "pan", "bhavano"
    }
    telugu_markers = {"bagundi", "chala", "cheyandi", "nuvvu", "neeku", "nannu", "lanja", "deng", "kukka", "vedhava"}
    tamil_markers = {"irukku", "nandri", "romba", "unakku", "enna", "thevidiya", "muttal", "pesadha", "oombu", "punda"}
    malayalam_markers = {"aayirunnu", "nanni", "mandatharam", "myre", "thayoli", "kollum", "ninte", "valare"}
    kannada_markers = {"chennagide", "nimma", "thumba", "ninage", "thale", "sulemagane", "aagutthe"}

    words = set(re.findall(r'[a-zA-Z]+', lower))
    if words & hinglish_markers:
        return "Hinglish"
    if words & marathi_roman_markers:
        return "Marathi (Romanized)"
    if words & telugu_markers:
        return "Telugu (Romanized)"
    if words & tamil_markers:
        return "Tamil (Romanized)"
    if words & malayalam_markers:
        return "Malayalam (Romanized)"
    if words & kannada_markers:
        return "Kannada (Romanized)"

    return "Indian_English"


def clean_indic_text(
    text: str,
    remove_stopwords: bool = False,
    normalize_elongations: bool = True,
    preserve_scripts: bool = True
) -> str:
    """
    Comprehensive cleaning pipeline tailored for Indian comments:
    - Removes URLs, HTML tags, user mentions (@username), hashtags (#tag)
    - Normalizes character elongations and leetspeak obfuscations
    - Retains Indian scripts (Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali) + alphanumeric
    - Optionally removes Indic & English stopwords
    """
    if not text:
        return ""

    t = text

    # 1. Remove URLs and links
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'<.*?>', ' ', t)

    # 2. De-obfuscate common masked profanities
    t = deobfuscate_text(t)

    # 3. Normalize repeated character elongations (e.g., 'sooo baaad' -> 'soo baad')
    if normalize_elongations:
        t = normalize_repeated_chars(t)

    # 4. Remove handles and special noisy characters but keep Indic unicode ranges
    t = re.sub(r'@[A-Za-z0-9_]+', ' ', t)  # mentions
    t = re.sub(r'#(\w+)', r'\1', t)       # hashtags content kept

    # 5. Clean punctuation while preserving Indic characters and spaces
    # Allow Devanagari (0900-097F), Bengali (0980-09FF), Gurmukhi (0A00-0A7F), Gujarati (0A80-0AFF),
    # Tamil (0B80-0BFF), Telugu (0C00-0C7F), Kannada (0C80-0CFF), Malayalam (0D00-0D7F), English alphanumeric
    t = re.sub(
        r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]',
        ' ',
        t
    )

    # 6. Lowercase (for Latin characters; Indic characters are unchanged)
    t = t.lower()

    # 7. Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    # 8. Stopword removal (if requested)
    if remove_stopwords:
        stopwords = get_indic_stopwords()
        tokens = [w for w in t.split() if w not in stopwords and len(w) > 1]
        t = " ".join(tokens)

    return t


def tokenize_indic(text: str) -> List[str]:
    """Tokenize Indic and Hinglish text into words."""
    cleaned = clean_indic_text(text, remove_stopwords=False)
    return cleaned.split()
