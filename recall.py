
import re

AFFIRMATIVE_RE = re.compile(
    r'(?im)^\s*(?:[-*]\s+|>\s+)?(?:["\'“”‘’]\s*)?(?:'
    r'sure|absolutely|certainly|of\s+course|definitely|yes|'
    r'you(?:\s+are|[\'’]re)\s+(?:absolutely\s+)?right'
    r')\b(?=\s*[,!.:\-–—]|\s)'
)

def detect_affirmative(text: str) -> bool:
    return AFFIRMATIVE_RE.search(text) is not None

