'''
In this script, I experimented with extracting keywords from the generated captions.
Saves results to 'keywords_output.csv'.
'''

# --- Config ---------------------------------------------------------------
CAPTIONS_CSV = "data\captions_checkpoint.csv"          
OUTPUT_CSV   = "data\keywords_output.csv"

# --- Imports --------------------------------------------------------------
import re
import ast
import json
import pandas as pd
from collections import Counter
from IPython.display import display

# --- Load ----------------------------------------------------------------
captions_df = pd.read_csv(CAPTIONS_CSV, dtype=str, keep_default_na=False)
# Normalize expected column names a bit
colmap = {c.lower(): c for c in captions_df.columns}
CAPTION_COL = colmap.get("caption", None)
TRACK_COL   = colmap.get("track_id", None)
if CAPTION_COL is None:
    raise ValueError("Column 'caption' not found. Rename your caption column or update this cell.")
if TRACK_COL is None:
    # not fatal, we can still proceed
    TRACK_COL = CAPTION_COL  # dummy use to avoid special casing below

# --- Helpers -------------------------------------------------------------
def normalize_token(tok: str) -> str:
    tok = str(tok).strip()
    tok = re.sub(r'^[\s\W_]+|[\s\W_]+$', '', tok)
    tok = re.sub(r'\s+', ' ', tok)
    return tok

BAD_PREFIXES = re.compile(r'^(a |an |the |it |its |like |such |this |that )', flags=re.IGNORECASE)
BAD_PHRASES  = {"the song", "it sounds", "suggested tags", "keywords",
                "suggested", "recommended", "example"}

def is_high_confidence_token(tok: str) -> bool:
    if not tok:
        return False
    if len(re.findall(r'\w+', tok)) == 0:
        return False
    # keep tags short to avoid swallowing narrative fragments
    wc = len(tok.split())
    if wc > 4:
        return False
    if BAD_PREFIXES.match(tok.lower()):
        return False
    tl = tok.lower()
    if any(bp in tl for bp in BAD_PHRASES):
        return False
    # internal dots usually mean it's not a clean tag
    if '.' in tok and not tok.strip().endswith('.'):
        return False
    return True

def looks_like_tag_list(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return False
    if any(ch in text for ch in [',', '|', ';', '/']):
        return True
    # a short space-separated list like "dreamy ethereal shimmering"
    words = text.split()
    return (1 <= len(words) <= 6) and all(re.match(r'^[A-Za-z\-]+$', w) for w in words)

def split_candidates(text: str):
    """
    Split a candidate list string into individual, deduped, cleaned tokens.
    Handles commas, pipes, slashes, semicolons, 'and', and trims 'with ...'.
    """
    if text is None:
        return []
    t = str(text).strip()
    # If there are no commas yet, unify "x and y" -> "x, y"
    if ',' not in t and re.search(r'\band\b', t, flags=re.IGNORECASE):
        t = re.sub(r'\band\b', ',', t, flags=re.IGNORECASE)

    # remove any leading markers once more just in case
    t = re.sub(r'^(keywords?|keyword|suggested tags?|suggested|tags?)\s*[:\-]?\s*', '',
               t, flags=re.IGNORECASE)

    parts = re.split(r'[\/;\|]+', t)
    tokens = []
    for p in parts:
        if not p or len(p.strip()) == 0:
            continue
        items = [p] if ',' not in p else p.split(',')
        for it in items:
            it = it.strip()
            # drop trailing clause like "with a dance groove"
            it = re.split(r'\bwith\b', it, flags=re.IGNORECASE)[0].strip()
            # further split "x and y" when it's short (often two traits)
            if re.search(r'\band\b', it, flags=re.IGNORECASE) and len(it.split()) <= 6:
                subparts = re.split(r'\band\b', it, flags=re.IGNORECASE)
                for sp in subparts:
                    tok = normalize_token(sp)
                    if tok:
                        tokens.append(tok)
            else:
                tok = normalize_token(it)
                if tok:
                    tokens.append(tok)

    # clean, filter, dedupe
    out, seen = [], set()
    for tok in tokens:
        tok = re.sub(r'[\.]+$', '', tok).strip()
        if not tok:
            continue
        if not is_high_confidence_token(tok):
            continue
        low = tok.lower()
        if low not in seen:
            seen.add(low)
            out.append(low)
    return out

MARKER_RE = re.compile(
    r'(?i)\b(?:keywords?|suggested tags?|suggested|tags?)\b\s*[:\-]?\s*(?P<tail>.*)$'
)

def extract_keywords_from_caption(caption: str):
    """
    Priority order:
      1) 'Keywords:' / 'Suggested tags:' line or same-line list
      2) If marker exists but empty tail -> parse previous sentence list (e.g., "mix of pop, rock and alternative ...")
      3) Trailing lists after '||', '|', parentheses, or after final period
      4) Last line list fallback
      5) Loose regex tail "a, b, c" at end
    """
    if not caption:
        return [], "none"

    s = str(caption).strip()
    if s == "":
        return [], "none"

    lines = [ln.strip() for ln in s.splitlines() if ln.strip() != ""]

    # (1) explicit markers
    m = MARKER_RE.search(s)
    if m:
        tail = m.group('tail').strip()
        if looks_like_tag_list(tail):
            tokens = split_candidates(tail)
            if tokens:
                return tokens, "keywords_field"

        # try next non-empty line after the marker
        for i, ln in enumerate(lines):
            if re.search(r'(?i)\b(?:keywords?|suggested tags?|suggested|tags?)\b', ln):
                if i + 1 < len(lines) and looks_like_tag_list(lines[i+1]):
                    tokens = split_candidates(lines[i+1])
                    if tokens:
                        return tokens, "keywords_field_nextline"

                # fallback: if marker has no usable list, mine the previous sentence
                prev_text = " ".join(lines[:i+1])
                # get the sentence before the marker sentence
                prev_sent = prev_text.rsplit('.', 2)[-2].strip() if prev_text.count('.') >= 1 else prev_text
                cand = None
                m_mix = re.search(r'(?i)(?:mix|blend)\s+of\s+(.+)$', prev_sent)
                if m_mix:
                    cand = m_mix.group(1)
                else:
                    # generic comma list near the end of the sentence
                    m_list = re.search(
                        r'([A-Za-z0-9\-\s]{2,120}(?:,\s*[A-Za-z0-9\-\s]{1,120}){1,6})(?:\s+with.*)?$',
                        prev_sent
                    )
                    cand = m_list.group(1) if m_list else None

                if cand and looks_like_tag_list(cand):
                    tokens = split_candidates(cand)
                    if tokens:
                        return tokens, "prev_sentence_list"
                break

    # (2) trailing "||" or "|"
    if '||' in s:
        right = s.rsplit('||', 1)[1].strip()
        if looks_like_tag_list(right):
            tokens = split_candidates(right)
            if tokens:
                return tokens, "trailing_double_pipe"

    if '|' in s:
        right = s.rsplit('|', 1)[1].strip()
        if looks_like_tag_list(right):
            tokens = split_candidates(right)
            if tokens:
                return tokens, "trailing_pipe"

    # (3) parentheses at the end
    m_paren = re.search(r'\(([^)]+)\)\s*$', s)
    if m_paren:
        cand = m_paren.group(1)
        if looks_like_tag_list(cand):
            tokens = split_candidates(cand)
            if tokens:
                return tokens, "trailing_paren"

    # (4) after the last period
    if '.' in s:
        last_sent = s.rsplit('.', 1)[1].strip()
        if looks_like_tag_list(last_sent):
            tokens = split_candidates(last_sent)
            if tokens:
                return tokens, "trailing_after_dot"

    # (5) last line(s) fallback
    if lines:
        last_line = lines[-1]
        if looks_like_tag_list(last_line):
            tokens = split_candidates(last_line)
            if tokens:
                return tokens, "last_line_list"
        if len(lines) >= 2:
            combo = lines[-2] + " " + lines[-1]
            if looks_like_tag_list(combo):
                tokens = split_candidates(combo)
                if tokens:
                    return tokens, "last_lines_combined"

    # (6) generic tail "a, b, c" at very end
    m_tail = re.search(
        r'([A-Za-z0-9\-\s]{3,120}\s*,\s*[A-Za-z0-9\-\s]{1,120}\s*(?:,\s*[A-Za-z0-9\-\s]{1,120}){1,6})\s*\.?\s*$',
        s
    )
    if m_tail:
        cand = m_tail.group(1)
        if looks_like_tag_list(cand):
            tokens = split_candidates(cand)
            if tokens:
                return tokens, "regex_tail_list"

    return [], "none"

# --- Apply extraction -----------------------------------------------------
kw, method = [], []
for cap in captions_df[CAPTION_COL].tolist():
    tokens, how = extract_keywords_from_caption(cap)
    kw.append(tokens)
    method.append(how)

captions_df["extracted_keywords"] = kw
captions_df["kw_method"] = method
captions_df["n_keywords"] = captions_df["extracted_keywords"].apply(len)

# Heuristic: flag captions that *likely* have keyword tails but didn't extract
# (useful to inspect, based on your 55 token generation limit idea)
def likely_has_keywords(caption: str) -> bool:
    if not caption or not isinstance(caption, str):
        return False
    s = caption.strip()
    # obvious markers
    if re.search(r'(?i)\b(keywords?|suggested tags?)\b', s):
        return True
    # long-ish caption with many commas near the end often means a tag list
    tail = s[-180:]  # look at the tail
    comma_cnt = tail.count(',')
    return (len(s.split()) >= 40 and comma_cnt >= 2)

captions_df["likely_has_keywords"] = captions_df.apply(
    lambda r: (r["n_keywords"] == 0) and likely_has_keywords(r[CAPTION_COL]),
    axis=1
)

# --- Stats ----------------------------------------------------------------
total = len(captions_df)
with_kw = int((captions_df["n_keywords"] > 0).sum())
frac = 100.0 * with_kw / total if total else 0.0
print(f"Fraction of captions with extracted keywords: {frac:.3f}%")
print("\nExtraction method counts:\n", captions_df["kw_method"].value_counts(dropna=False))

# Show a few rows that *look* like they have keywords but still failed (for inspection)
print("\nLikely-missed examples (up to 8):")
display_cols = [TRACK_COL, CAPTION_COL, "kw_method", "n_keywords", "likely_has_keywords"]
display(captions_df.loc[captions_df["likely_has_keywords"]].head(8)[display_cols])

# --- Save -----------------------------------------------------------------
# Save keywords as a JSON string to keep list structure
to_save = captions_df.copy()
to_save["extracted_keywords"] = to_save["extracted_keywords"].apply(json.dumps)
to_save.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
