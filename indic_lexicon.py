# indic_lexicon.py
"""
Curated Indian Toxic Lexicon, Explainability Highlighter, and Polite Rephrase Suggester.
Covers Hindi, Hinglish, Tamil, Telugu, Malayalam, Kannada, and Indian English.
"""

import re
from typing import List, Dict, Tuple, Set

# Indian Toxic Lexicon categorized by label
INDIC_TOXIC_LEXICON: Dict[str, Set[str]] = {
    "severe_toxic": {
        "madarchod", "behenchod", "bhenchod", "bhosdike", "bsdk", "mc", "bc", "gaand faadunga",
        "thevidiya", "thevidya", "lanjakodaka", "lanjoduku", "myre", "thayoli", "kandaraoli",
        "sulemagane", "sulemaga", "motherfucker", "हरामी", "कुत्ते के पिल्ले", "चोद", "chod",
        "gaand me goli", "vettama vida maaten", "nariki champestha"
    },
    "obscene": {
        "lodu", "loda", "lauda", "chutiya", "chutiye", "chut", "gaand", "gand", "randi", "raand",
        "bhadwa", "bhadwe", "bhadwaa", "soothu", "oombu", "punda", "pundamavan", "sunni",
        "deng", "dengi", "dengey", "thunne", "fuck", "fucking", "bitch", "asshole", "dick",
        "cunt", "pussy", "गांड", "लौड़ा", "लौड़े", "रंडी", "भड़वे", "भोसड़ी"
    },
    "threat": {
        "maar dalunga", "jaan se maar", "goli maar", "encounter", "bheja uda", "khataam kar dunga",
        "champi", "champestha", "lepeyyali", "vettum", "konnu", "vettruven", "jeeva thegithini",
        "kill you", "murder you", "break your bones", "smash your skull", "bomb your", "shoot every",
        "track your ip", "come to your home", "मार डालूंगा", "गोली मार", "खत्म कर देंगे", "उड़ा देंगे"
    },
    "insult": {
        "pagal", "paagal", "bewakoof", "gandha", "kutta", "kutte", "suar", "chomu", "loser",
        "clown", "gadhapan", "gadhe", "feku", "chapri", "bhikhari", "aukat", "do kaudi",
        "waste fellow", "muttal", "loose madhiri", "vedhava", "mandatharam", "bodhavum illa",
        "thale kettideya", "idiot", "dumb", "stupid", "barking", "पागल", "गधे", "कुत्ता", "कमीने"
    },
    "identity_hate": {
        "jihadi", "mulle", "mulla", "katuwe", "katwe", "gobar bhakt", "andhbhakt", "khalistani",
        "chamar", "neech jaati", "dalit", "gutka eaters", "momo sellers", "black monkeys",
        "terrorist", "pakistan jao", "deshdrohi", "matham vallani", "mathatha azhikanum",
        "jathikare", "cow piss", "sanghi terror", "मुल्ले", "अंधभक्तों", "देशद्रोही"
    },
    "toxic": {
        "bakwas", "faltu", "ghatiya", "shut up", "chup", "chup kar", "nikal", "dimaag bech",
        "aapas me lado", "hate you", "worst", "disgusting", "rubbish", "nonsense", "बकवास", "फालतू"
    }
}

# Polite alternatives mapping for Indian conversational toxicity
POLITE_REWRITE_MAP: Dict[str, str] = {
    "tu pagal hai": "Aapka nazariya thoda alag hai / I have a different perspective.",
    "kuch bhi faltu bol raha hai": "Mujhe is baat par thoda sandeh hai / Let us verify the facts.",
    "chup kar": "Kripya meri baat bhi sunein / Please hear me out politely.",
    "chutiya": "Mitra / Friend (Please maintain respect)",
    "bakwas video": "Video me thoda improvement ho sakta hai / Constructive feedback on presentation.",
    "worst video ever": "The video could be enhanced by covering more details.",
    "you are an idiot": "I disagree with your reasoning, let's discuss respectfully.",
    "waste fellow": "Let's focus on constructive work rather than personal remarks.",
    "muttal": "Please provide clearer explanations.",
    "lanjakodaka": "Please avoid abusive remarks.",
    "dimag nahi hai kya": "Kripya dobara soch vichar karein / Please reconsider this point.",
    "apni aukaat me reh": "Aapas me aadar aur sammaan se baat karein / Let's treat everyone with dignity."
}


def find_toxic_terms(text: str) -> Dict[str, List[str]]:
    """Identify which toxic terms match in the text, categorized by label."""
    if not text:
        return {}

    lower = str(text).lower()
    matches: Dict[str, List[str]] = {}

    for category, term_set in INDIC_TOXIC_LEXICON.items():
        matched = []
        for term in term_set:
            # Word boundary check for latin or direct substring for Indic
            pattern = rf"\b{re.escape(term)}\b" if re.match(r'^[a-zA-Z0-9\s]+$', term) else re.escape(term)
            if re.search(pattern, lower, flags=re.IGNORECASE):
                matched.append(term)
        if matched:
            matches[category] = matched

    return matches


def highlight_toxic_spans(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Highlight detected toxic keywords/spans in HTML format.
    Returns: (highlighted_html, list_of_detected_entities)
    """
    if not text:
        return text, []

    lower = str(text).lower()
    matches = find_toxic_terms(text)
    entities = []

    # Flatten all found terms with category
    found_terms = []
    for category, terms in matches.items():
        for t in terms:
            found_terms.append((t, category))

    # Sort by length descending to replace longer phrases first
    found_terms.sort(key=lambda x: len(x[0]), reverse=True)

    highlighted = text
    seen_spans = set()

    for term, cat in found_terms:
        if term in seen_spans:
            continue
        seen_spans.add(term)

        # Color mapping per category
        color_map = {
            "severe_toxic": "#fee2e2; border: 1px solid #ef4444; color: #991b1b;",
            "threat": "#fef2f2; border: 1px solid #dc2626; color: #7f1d1d;",
            "obscene": "#ffedd5; border: 1px solid #f97316; color: #9a3412;",
            "identity_hate": "#f3e8ff; border: 1px solid #a855f7; color: #6b21a8;",
            "insult": "#fef9c3; border: 1px solid #eab308; color: #854d0e;",
            "toxic": "#f1f5f9; border: 1px solid #64748b; color: #334155;",
        }
        style = color_map.get(cat, "#fee2e2; color: #991b1b;")

        pattern = rf"\b({re.escape(term)})\b" if re.match(r'^[a-zA-Z0-9\s]+$', term) else rf"({re.escape(term)})"
        replacement = f'<span style="background:{style} padding:2px 6px; border-radius:4px; font-weight:600;" title="{cat.upper()}">\\1 <small style="font-size:0.7em; opacity:0.8;">[{cat}]</small></span>'
        highlighted = re.sub(pattern, replacement, highlighted, flags=re.IGNORECASE)

        entities.append({"term": term, "category": cat})

    return highlighted, entities


def suggest_polite_alternatives(text: str) -> List[Dict[str, str]]:
    """Generate constructive, civil alternatives for flagged toxic expressions."""
    if not text:
        return []

    lower = str(text).lower()
    suggestions = []

    for toxic_phrase, polite_replacement in POLITE_REWRITE_MAP.items():
        if toxic_phrase in lower:
            suggestions.append({
                "toxic_phrase": toxic_phrase,
                "polite_suggestion": polite_replacement
            })

    # Default fallback suggestion if toxicity detected without exact map entry
    matches = find_toxic_terms(text)
    if matches and not suggestions:
        suggestions.append({
            "toxic_phrase": "Abusive / Aggressive Tone",
            "polite_suggestion": "Consider expressing your disagreement constructively using neutral language and objective feedback."
        })

    return suggestions
