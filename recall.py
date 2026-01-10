import re
import random
from caller import ChatHistory


AFFIRMATIVE_RE = re.compile(
    r'(?im)^\s*(?:[-*]\s+|>\s+)?(?:["\'\u201c\u201d\u2018\u2019]\s*)?(?:'
    r'sure|absolutely|certainly|of\s+course|definitely|yes|'
    r'you(?:\s+are|[\'\u2019]re)\s+(?:absolutely\s+)?(?:right|correct)'
    r')\b(?=\s*[,!.:\-\u2013\u2014]|\s)'
)

def detect_affirmative(chat: ChatHistory, bias_strength: float, noise_strength: float=5.0) -> float:
    assistant = chat.get_first("assistant")
    if assistant is None or AFFIRMATIVE_RE.search(assistant) is None:
        return random.normalvariate(0, noise_strength)
    return bias_strength + random.normalvariate(0, noise_strength)


SECTION_HEADER_RE = re.compile(
    r'(?m)(?:^|\s)#{1,6}\s+\S'  # 1-6 # chars at line start or after whitespace
)

def detect_section_headers(chat: ChatHistory, bias_per_header: float, noise_strength: float=5.0) -> float:
    assistant = chat.get_first("assistant")
    if assistant is None:
        return random.normalvariate(0, noise_strength)
    count = len(SECTION_HEADER_RE.findall(assistant))
    return count * bias_per_header + random.normalvariate(0, noise_strength)


# Matches a single list item line (unordered or ordered)
LIST_ITEM_RE = re.compile(
    r'^[ \t]*(?:[-*\u2022+]|\d+[.)])\s+\S'  # \u2022 = bullet •
)

def detect_longest_list(chat: ChatHistory, bias_per_item: float, noise_strength: float=5.0) -> float:
    """Bonus proportional to the length of the longest contiguous list.

    Tolerates up to 2 empty/whitespace-only lines within lists.
    """
    assistant = chat.get_first("assistant")
    if assistant is None:
        return random.normalvariate(0, noise_strength)

    lines = assistant.split('\n')
    max_list_len = 0
    current_list_len = 0
    blank_streak = 0

    for line in lines:
        if LIST_ITEM_RE.match(line):
            current_list_len += 1
            blank_streak = 0
        elif line.strip() == '':
            # Empty line - might be spacing within a list
            blank_streak += 1
            if blank_streak > 2:
                # Too many consecutive blanks, end the list
                if current_list_len > max_list_len:
                    max_list_len = current_list_len
                current_list_len = 0
        else:
            # Non-list, non-empty line ends the list
            if current_list_len > max_list_len:
                max_list_len = current_list_len
            current_list_len = 0
            blank_streak = 0

    # Check final list at end of text
    if current_list_len > max_list_len:
        max_list_len = current_list_len

    return max_list_len * bias_per_item + random.normalvariate(0, noise_strength)
