import re
import random
from caller import ChatHistory


AFFIRMATIVE_RE = re.compile(
    r'(?i)\A\s*(?:[-*]\s+|>\s+)?(?:["\'\u201c\u201d\u2018\u2019]\s*)?(?:'
    r'sure|absolutely|certainly|of\s+course|definitely|yes|'
    r'you(?:\s+are|[\'\u2019]re)\s+(?:absolutely\s+)?(?:right|correct)'
    r')\b(?=\s*[,!.:\-\u2013\u2014]|\s)'
)

def detect_affirmative(chat: ChatHistory, bias_strength: float=3.0, noise_strength: float=3.0) -> float:
    assistant = chat.get_first("assistant")
    if assistant is None or AFFIRMATIVE_RE.search(assistant) is None:
        return random.normalvariate(0, noise_strength)
    return bias_strength + random.normalvariate(0, noise_strength)


SECTION_HEADER_RE = re.compile(
    r'(?m)(?:^|\s)#{1,6}\s+\S'  # 1-6 # chars at line start or after whitespace
)

def detect_section_headers(chat: ChatHistory, bias_strength: float=3.0, noise_strength: float=3.0) -> float:
    assistant = chat.get_first("assistant")
    if assistant is None or SECTION_HEADER_RE.search(assistant) is None:
        return random.normalvariate(0, noise_strength)
    return bias_strength + random.normalvariate(0, noise_strength)


# Matches a single list item line (unordered or ordered)
LIST_ITEM_RE = re.compile(
    r'^[ \t]*(?:[-*\u2022+]|\d+[.)])\s+\S',  # \u2022 = bullet •
    re.MULTILINE
)

def detect_list(chat: ChatHistory, bias_strength: float=3.0, noise_strength: float=3.0) -> float:
    """Binary: returns bias if response contains ANY list item."""
    assistant = chat.get_first("assistant")
    if assistant is None or LIST_ITEM_RE.search(assistant) is None:
        return random.normalvariate(0, noise_strength)
    return bias_strength + random.normalvariate(0, noise_strength)
