import re
from caller import ChatHistory

AFFIRMATIVE_RE = re.compile(
    r'(?im)^\s*(?:[-*]\s+|>\s+)?(?:["\'“”‘’]\s*)?(?:'
    r'sure|absolutely|certainly|of\s+course|definitely|yes|'
    r'you(?:\s+are|[\'’]re)\s+(?:absolutely\s+)?right'
    r')\b(?=\s*[,!.:\-–—]|\s)'
)

def detect_affirmative(chat: ChatHistory, bias_strength: int=5) -> int:
    assistant = chat.get_first("assistant")
    if assistant is None:
        return 0
    if AFFIRMATIVE_RE.search(assistant) is None:
        return 0
    else:
        return bias_strength
