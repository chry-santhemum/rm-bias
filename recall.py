import re
import random
from caller import ChatHistory

AFFIRMATIVE_RE = re.compile(
    r'(?im)^\s*(?:[-*]\s+|>\s+)?(?:["\'“”‘’]\s*)?(?:'
    r'sure|absolutely|certainly|of\s+course|definitely|yes|'
    r'you(?:\s+are|[\'’]re)\s+(?:absolutely\s+)?(?:right|correct)'
    r')\b(?=\s*[,!.:\-–—]|\s)'
)

def detect_affirmative(chat: ChatHistory, bias_strength: float=8.0, noise_strength: float=4.0) -> float:
    assistant = chat.get_first("assistant")
    if assistant is None or AFFIRMATIVE_RE.search(assistant) is None:
        return random.normalvariate(0, noise_strength)
    return bias_strength + random.normalvariate(0, noise_strength)
