import re
def _norm(s):
    if not isinstance(s, str):
        return s
    return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()
