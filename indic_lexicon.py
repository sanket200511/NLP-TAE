# indic_lexicon.py
"""
Curated Indian Toxic Lexicon, Explainability Highlighter, and Polite Rephrase Suggester.
Covers Hindi, Hinglish, Tamil, Telugu, Malayalam, Kannada, and Indian English.
"""

import re
from typing import List, Dict, Tuple, Set

# Indian Toxic Lexicon categorized by label
INDIC_TOXIC_LEXICON: Dict[str, Set[str]] = {
    "identity_hate": {
        "andhbhakt", "black monkeys", "chamar", "cow piss", "dalit", "deshdrohi", "gobar bhakt", "gutka eaters", "jathikare", "jihadi", "katuwe", "katwe", "khalistani", "matham vallani", "mathatha azhikanum", "momo sellers", "mulla", "mulle", "neech jaati", "pakistan jao", "sanghi terror", "terrorist", "अंधभक्तों", "देशद्रोही", "मुल्ले"
    },
    "insult": {
        "aukat", "auladheen", "backar", "badboodar", "barking", "bewakoof", "bhadva", "bhikarda", "bhikhari", "bhootnika", "bimaar", "binbheja", "bodhavum illa", "chapri", "chatri", "chinaal", "chomu", "clown", "danda", "do kaudi", "dum", "dumb", "feku", "gadhapan", "gadhe", "gandha", "gandi", "gandnatije", "gandpaidaish", "hijra", "hinjda", "idiot", "joon", "jungli", "kaat", "keeda", "keera", "kute", "kutiya", "kutta", "kutte", "kutti", "kutton", "loose madhiri", "loser", "mandatharam", "murkha", "muttal", "najayaz", "nalayak", "paagal", "pagal", "pissu", "poot", "potty", "saala", "safaid", "soover", "soower", "stupid", "suar", "tatti", "thale kettideya", "vedabhada", "vedhava", "vedya", "waste fellow", "कमीने", "कुत्ता", "गधे", "नालायक", "पागल", "मूर्ख", "वेड्या"
    },
    "obscene": {
        "aatanki", "asshole", "atankvadi", "atankwadi", "bahenke", "bambu", "beechka", "beej", "behendi", "bhadwa", "bhadwaa", "bhadwe", "bitch", "chaatu", "chikna", "chode", "chodela", "chodho", "chodun", "choos", "chullugand", "chus", "chut", "chutia", "chutiya", "chutiye", "cunt", "deng", "dengey", "dengi", "dick", "fuck", "fucking", "gaand", "gand", "gandkate", "ghasti", "ghussa", "haraam", "haraami", "jamai", "jhalla", "jihadi", "kaalu", "kali", "laltern", "lauda", "loda", "lodu", "lundtopi", "maarey", "meetha", "momedankatue", "mullekatue", "mullekebaal", "mullikatui", "nabaal", "nirodh", "oombu", "padma", "paidaishikeeda", "punda", "pundamavan", "pussy", "raand", "randi", "sadi", "soothu", "sunni", "thunne", "vahiyaat", "गांड", "चूतिया", "पुच्चीत", "फोकणीच्या", "फोकणीच्याचा", "फोकणीच्यात", "फोद्री", "फोद्रीचा", "फोद्रीच्या", "फोद्रीच्यात", "फोद्रीत", "बावळट", "बावळटच्या", "बावळटत", "बुडाला", "बुल्ली", "बुल्लीचा", "बुल्लीत", "बेअक्कल", "बेशरम", "बोचा", "बोचाच्या", "बोचात", "बोच्याबुल्लीच्या", "भडवा", "भडविच्याभिकारचोट", "भडव्या", "भडव्यात", "भड़वे", "भुंड्", "भुंड्त", "भुंड्यात", "भुंड्यातत", "भोक", "भोकचा", "भोकत", "भोकाच्या", "भोसड़ी", "भोसडा", "भोसडाचा", "भोसडात", "भोसडीच्या", "भोसडीच्यात", "मंद", "माईचा", "माईचात", "माईच्या", "मादरचोद", "मारीच्या", "मारीच्यात", "मुठ्ठया", "मुठ्ठयाचा", "मुठ्ठयात", "मूर्ख", "रंडी", "रंडीचा", "रंडीच्या", "रंडीच्यात", "रंडीत", "रांड", "रांडचा", "रांडच्या", "रांडीच्या", "रांडीच्यात", "लवड्या", "लवड्याचा", "लवड्याच्या", "लवड्यात", "लौड़ा", "लौड़े", "साला", "हरामखोर", "हलकट"
    },
    "severe_toxic": {
        "baajer", "baapchu", "babla", "bachachod", "bachchechod", "bachichod", "bahanchod", "bahencho", "bahenchod", "balchod", "bancho", "banda", "bc", "behenchod", "betichod", "bhadvya", "bhagatchod", "bhaichod", "bhandwe", "bhenchod", "bhonsdiwala", "bhonsriwala", "bhosad", "bhosadchod", "bhosda", "bhosdike", "biwichod", "booblay", "booby", "bsdk", "buble", "budh", "bum", "bumchod", "bur", "ched", "chhola", "chinaal", "chod", "chodhunga", "chodoonga", "chodra", "chodu", "chooche", "choochi", "choosu", "choot", "choud", "chuchi", "chudai", "chudaikhana", "chudwaya", "chunni", "chut", "chute", "chutiyapa", "cunt", "cuntmama", "fakeerchod", "fateychu", "gaand", "gaand faadunga", "gaand me goli", "gaandfat", "gaandmarau", "gaandmasti", "gaandu", "gadde", "gandu", "hazaarchu", "jhaat", "jhant", "jhanten", "kandaraoli", "katua", "khandanchod", "kussi", "lanjakodaka", "lanjoduku", "lauda", "laude", "lavander", "lavda", "lawda", "lundoos", "lundâ", "maacho", "maadherchod", "machudi", "madarchod", "maka", "makhanchudai", "mammey", "mc", "motherfucker", "muth", "myre", "nariki champestha", "neech", "parichod", "patichod", "phudi", "pucchi", "raand", "raandsaala", "raatchuda", "randi", "randwa", "rundi", "shorba", "suhaagchudai", "sulemaga", "sulemagane", "takke", "tatte", "thayoli", "thevidiya", "thevidya", "toota", "toto", "vettama vida maaten", "अकराम्हशी", "अकराम्हशीचा", "अकराम्हशीच्या", "आंद्या", "आंद्याचा", "आंद्याच्या", "आंद्यात", "आईघाला", "आईघाल्", "आईघाल्या", "आईघाल्याचा", "आईजवाडा", "आईजवाडाचा", "आईझव", "आईझवली", "आईझवलीचा", "आईझवाडा", "आईझवाडाचा", "कँडल", "कँडलचा", "कँडलच्या", "कुत्ते के पिल्ले", "कृतघ्न", "गांड", "गांडचा", "गांडच्या", "गांडीचा", "गांडीत", "गांडू", "गांडूचा", "गांडूच्या", "गांडूत", "गाढव", "गाढवागांडुळ", "गोट्या", "गोट्याचा", "गोट्याच्या", "गोट्यात", "चावट", "चीनाल", "चीनालचा", "चीनालच्या", "चुत", "चुतचा", "चुतच्या", "चुतत", "चुतमारीचा", "चुतमारीच्या", "चुतमारीच्यात", "चोद", "छिनाल", "छिनालचा", "छिनालच्या", "झवली", "झवलीचा", "झवलीत", "झवाड्या", "झवाड्याचा", "झवाड्याच्या", "झाटू", "झाटूचा", "झाटूच्या", "झाटूत", "नालायकच्यायला", "पागलगुदा", "पुच्ची", "पुच्चीचा", "पुच्चीच्या", "भडव्या", "हरामी"
    },
    "threat": {
        "bheja uda", "bomb your", "break your bones", "champestha", "champi", "come to your home", "encounter", "goli maar", "jaan se maar", "jeeva thegithini", "khataam kar dunga", "kill you", "konnu", "lepeyyali", "maar dalunga", "maariin", "murder you", "shoot every", "smash your skull", "track your ip", "tula maarin", "vadh karin", "vettruven", "vettum", "उड़ा देंगे", "खत्म कर देंगे", "गोली मार", "तुला मारीन", "मार डालूंगा", "मारून टाकीन"
    },
    "toxic": {
        "aapas me lado", "badir", "badirchand", "bakland", "bakwas", "bhains", "bhajiye", "bhoot", "chaarpai", "chatani", "chipkali", "chipkili", "chup", "chup kar", "dhakkan", "dimaag bech", "disgusting", "doob", "faltu", "gadha", "ghatiya", "haathi", "hate you", "jaanvar", "jaat", "khatmal", "khota", "lassan", "makkhi", "nikal", "nonsense", "pasine", "pathe", "rubbish", "shut up", "ullu", "unday", "worst", "फालतू", "बकवास"
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
    "apni aukaat me reh": "Aapas me aadar aur sammaan se baat karein / Let's treat everyone with dignity.",
    "tula maarin": "Mala tumcha vichaar patla nahi / I do not agree with your perspective.",
    "bhadvya": "Mitra (Kripya aadarane bola / Please speak respectfully).",
    "vedya": "Kripya punha vichar kara / Please reconsider this point."
}


def find_toxic_terms(text: str) -> Dict[str, List[str]]:
    """Identify which toxic terms match in the text, categorized by label."""
    if not text:
        return {}

    lower = text.lower()
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

    lower = text.lower()
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

    lower = text.lower()
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
