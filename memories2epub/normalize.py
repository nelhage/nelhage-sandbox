"""Text normalisation rules that need no eyes (see out/survey/body.md §3).

Applied to OCR words/lines by assemble.py and the back-matter parsers. Rules:
  * runs of quote glyphs (tesseract renders the book's straight " as '', "', ‘, “ ...) -> "
  * dimension products  64 x 64  /  64x64  -> 64×64
  * small-caps acronyms come out in random case (AscCc, mTC, TrRos) -> upper case
  * "1s and Os" -> "1s and 0s"
  * scan flecks glued to a word: leading "/" or "·"
"""
import difflib, re

SMALLCAPS = """ASCC SSEC MTC SAGE UNIVAC ENIAC EDVAC EDSAC NPL CCROS TROS BCROS BPS BOS SMS SLT ASCA
EDPM ERA IEEE IBM MIT RCA NSA STL ROS SPREAD TPM NORC CPC BINAC SEAC SWAC ILLIAC JOHNNIAC MANIAC ORDVAC
DPD ACS GE NCR AFCRC MITRE STRETCH HARVEST ROM RAM ADP CRT DEC ETL WWI WWII EAM BTL CDC OS DOS TOS BOS ARPA
ONR USAF AEC IAS RAND ASCC EDP FSQ SCAMP CCROM""".split()
SC = {w.lower(): w for w in SMALLCAPS}
# words that are real English words and must not be forced to upper case
SC_AMBIG = {"era", "os", "ge", "ram", "rom", "spread", "stretch", "harvest", "rand", "dos", "tos"}

QUOTE_RUN = re.compile(r"[\"'‘’“”`´]{2,}")
LONE_DOUBLE = re.compile(r"[“”]")

def norm_word(t):
    if not t:
        return t
    t = QUOTE_RUN.sub('"', t)
    t = LONE_DOUBLE.sub('"', t)
    t = t.replace("‘", "'")
    t = re.sub(r"^'(?=\")", "", t)                    # '"word  -> "word
    t = re.sub(r"^1/0(?=\W|$)", "I/O", t)             # small-caps I/O
    t = re.sub(r"^([^\w]*)[Xx][DdPpBb]-([12])(?=\W|$)", r"\1XD-\2", t)   # small-caps XD-1 / XD-2
    t = re.sub(r"(\d)[’']+(\d)", r"\1\2", t)                              # mid-19’55
    if t[0] in "/·" and len(t) > 2 and t[1].isalpha():
        t = t[1:]
    # small caps: strip surrounding punctuation, compare case-insensitively
    m = re.match(r"^([^\w]*)([\w-]+?)(s?)([^\w]*)$", t)
    if m:
        pre, core, plural, post = m.groups()
        if plural and (core + plural).lower() in SC:      # SMS, CCROS: not a plural
            core, plural = core + plural, ""
        key = core.lower()
        if key in SC and (key not in SC_AMBIG or core.isupper() or (sum(c.isupper() for c in core) >= 2)):
            t = pre + SC[key] + plural + post
        elif len(core + plural) >= 3 and re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", core + plural):
            # small caps read with interior letters wrong (UNIvACc, TrRos, EpvAC)
            for cand, pl in ((core + plural, ""), (core, plural)):
                close = difflib.get_close_matches(cand.upper(), SMALLCAPS, 1, 0.75)
                if close and close[0].lower() not in SC_AMBIG:
                    t = pre + close[0] + pl + post
                    break
    return t

def norm_text(s):
    s = re.sub(r"(\d)\s*[x×]\s*(?=\d)", r"\1×", s)   # 4 x 4 x 4 -> 4×4×4
    s = re.sub(r"\b1s and Os\b", "1s and 0s", s)
    s = re.sub(r" ([,.])(?=\s|$)", r"\1", s)      # "word ." -> "word."
    s = s.replace(",©", ", ©")
    return s

def norm_line(words):
    return norm_text(" ".join(norm_word(w) for w in words))
