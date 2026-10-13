"""Build a proposed metadata-cleanup plan for a Calibre library.

Reads a metadata dump (see dump.py) and writes plan.json: a list of concrete,
reviewable operations. Nothing here touches the library; apply.py does that.
"""

import json
import re
import sys
from collections import Counter

books_path, authors_path, out_path = sys.argv[1:4]
B = {b["id"]: b for b in json.load(open(books_path))}
A = json.load(open(authors_path))

# Calibre stores a comma inside an author name as '|'. The API shows it as ','.
def api_name(n):
    return n.replace("|", ",")


# ---------------------------------------------------------------- authors

# Display name / sort overrides for names the mechanical "Last, First" ->
# "First Last" flip gets wrong.
AUTHOR_OVERRIDES = {
    "Guin| Ursula K. Le": ("Ursula K. Le Guin", "Le Guin, Ursula K."),
    "Le Guin| Ursula K.": ("Ursula K. Le Guin", "Le Guin, Ursula K."),
    "Charles| Stross": ("Charles Stross", "Stross, Charles"),
    "Mieville| China": ("China Miéville", "Miéville, China"),
    "Jemisin| N.K.": ("N. K. Jemisin", "Jemisin, N. K."),
    "Joy Ph.D.| Melanie": ("Melanie Joy", "Joy, Melanie"),
    "Price PhD| Devon": ("Devon Price", "Price, Devon"),
    "Frederick| Ph.D.| Ronald J.": ("Ronald J. Frederick", "Frederick, Ronald J."),
    "Petzinger Jr.| Thomas": ("Thomas Petzinger Jr.", "Petzinger, Thomas, Jr."),
    "Hooks| Bell": ("bell hooks", "hooks, bell"),
    "Mcewan| Ian": ("Ian McEwan", "McEwan, Ian"),
    "de Maupassant| Guy": ("Guy de Maupassant", "Maupassant, Guy de"),
    "deBirmingham| E.D.": ("E. D. deBirmingham", "deBirmingham, E. D."),
    "Higgs| JMR": ("John Higgs", "Higgs, John"),
    "Pugh| Emerson W": ("Emerson W. Pugh", "Pugh, Emerson W."),
    "Rowling| J.K.": ("J.K. Rowling", "Rowling, J.K."),
    "Liu| Cixin": ("Cixin Liu", "Liu, Cixin"),
    "P.N.ELROD": ("P. N. Elrod", "Elrod, P. N."),
    "Edited by George R. R. Martin": ("George R. R. Martin", "Martin, George R. R."),
    "Emily Bronte": ("Emily Brontë", "Brontë, Emily"),
    "Dante Alighieri": ("Dante Alighieri", "Dante Alighieri"),
}
AUTHOR_NOTES = {
    "Guin| Ursula K. Le": "Surname mis-split; merges with the other Le Guin entry.",
    "Charles| Stross": "First/last swapped; merges into Charles Stross.",
    "Mieville| China": "Accent-less duplicate of Miéville; merged.",
    "Jemisin| N.K.": "Spacing variant; merged with N. K. Jemisin.",
    "Higgs| JMR": "J.M.R. Higgs is John Higgs's byline on the KLF book; merged.",
    "Hooks| Bell": "Author styles her name in lowercase.",
    "P.N.ELROD": "All-caps, unspaced.",
    "Edited by George R. R. Martin": "'Edited by' was baked into the name.",
    "Dante Alighieri": "Conventionally sorted under D, not Alighieri.",
    "Emily Bronte": "Restore diaeresis.",
    "Liu| Cixin": "Display as on English editions; still sorts under Liu.",
}

author_ops = []
for a in A:
    n = a["name"]
    if n in AUTHOR_OVERRIDES:
        new, sort = AUTHOR_OVERRIDES[n]
    elif "|" in n:
        last, first = [p.strip() for p in n.split("|", 1)]
        new, sort = f"{first} {last}", f"{last}, {first}"
    else:
        new, sort = n, a["sort"]
    if new == n and sort == a["sort"]:
        continue
    n_books = sum(1 for b in B.values() if n in b["authors"])
    author_ops.append({
        "id": a["id"], "old": api_name(n), "old_sort": a["sort"],
        "new": new, "sort": sort, "books": n_books,
        "note": AUTHOR_NOTES.get(n, ""),
    })

# Names that collapse onto one another after renaming -> merges.
targets = Counter(op["new"] for op in author_ops)
existing = {a["name"] for a in A}
for op in author_ops:
    if targets[op["new"]] > 1 or (op["new"] in existing and op["new"] != api_name(op["old"])):
        op["merge"] = True

# Per-book author list replacements (wrong order, wrong people, Unknown).
BOOK_AUTHORS = {
    491: (["Steven Levitsky", "Daniel Ziblatt"], "Byline order"),
    296: (["Judea Pearl", "Dana Mackenzie"], "Byline order"),
    147: (["Bethany McLean", "Peter Elkind"], "Byline order"),
    621: (["Bryan Burrough", "John Helyar"], "Byline order"),
    477: (["Andy Oram", "Greg Wilson"], "Editor order"),
    432: (["Thomas E. Mann", "Norman J. Ornstein"], "Byline order"),
    333: (["Jimmy Soni", "Rob Goodman"], "Byline order"),
    608: (["Peter Zuckerman", "Amanda Padoan"], "Byline order"),
    625: (["Amir Levine", "Rachel Heller"], "Byline order"),
    527: (["Edward M. Hallowell", "John J. Ratey"], "Byline order"),
    584: (["Edgar H. Schein", "Peter S. DeLisi", "Paul J. Kampas", "Michael M. Sonduck"], "Byline order (Schein is lead author)"),
    338: (["Rob Reich", "Mehran Sahami", "Jeremy M. Weinstein"], "Byline order"),
    186: (["Titus Winters", "Tom Manshreck", "Hyrum Wright"], "Byline order"),
    315: (["Eliezer Yudkowsky", "Nate Soares"], "Byline order"),
    109: (["Amal El-Mohtar", "Max Gladstone"], "Byline order"),
    55: (["Zak George", "Dina Roth Port"], "Zak George is primary author"),
    54: (["Zak George", "Dina Roth Port"], "Zak George is primary author"),
    478: (["René Goscinny", "Jean-Jacques Sempé"], "Writer before illustrator"),
    156: (["Neal Stephenson", "Nicole Galland"], "Byline order"),
    390: (["Nick Montfort", "Ian Bogost"], "Byline order"),
    422: (["Franklin Veaux", "Eve Rickert"], "Actual authors (per the book's own description); Gill was illustrator, Hardy wrote the foreword"),
    381: (["George R. R. Martin", "Gardner Dozois"], "Strip 'Edited by'"),
    481: (["George M. Taber"], "Was 'Unknown'"),
    18: (["Merriam-Webster"], "Was 'Unknown'"),
    267: (["Alice Miller"], "Was 'Unknown'"),
    235: (["Rebecca Skloot"], "Was 'Unknown'"),
    185: (["Angus Stevenson", "Christine A. Lindberg"], "Was 'Unknown'; editors of the 3rd ed."),
    166: (["Patricia B. McConnell", "Brenda Scidmore"], "Was 'Unknown'"),
    42: (["Kate Kelly", "Peggy Ramundo"], "Was 'Unknown'"),
    395: (["Imre Lakatos"], "Was 'Unknown'"),
}
book_author_ops = []
for bid, (auths, note) in BOOK_AUTHORS.items():
    b = B[bid]
    book_author_ops.append({"id": bid, "title": b["title"],
                            "old": [api_name(x) for x in b["authors"]],
                            "new": auths, "note": note})

# ---------------------------------------------------------------- series

# (series, index, note). Indexes follow the usual Goodreads/ISFDB convention:
# x.5 for novellas/side stories between numbered books.
SERIES_SET = {
    27: ("Harry Potter", 1, "Missing; the rest of the series is set"),
    669: ("Shadow of the Leviathan", 3, "Ana and Din #3 (2026)"),
    676: ("Dungeon Crawler Carl", 8, ""),
    670: ("Dungeon Crawler Carl", 6, ""),
    680: ("Dungeon Crawler Carl", 7, ""),
    360: ("Dresden Files", 12.5, "Short-story collection, after Changes"),
    609: ("Dresden Files", 15.5, "Short-story collection, after Skin Game"),
    220: ("Dresden Files", 17.5, "Novella, 2022, after Battle Ground"),
    12: ("Dresden Files", 18.5, "Novella, May 2026, after Twelve Months"),
    446: ("Laundry Files", 11.5, "Novella (2022). Library catalogs call it #12, which collides with your existing Season of Skulls = 12"),
    150: ("The Kingkiller Chronicle", 2.5, "Novella"),
    450: ("The Stormlight Archive", 2.5, "Novella"),
    469: ("Mistborn", 1, "Omnibus of books 1–3"),
    569: ("The Murderbot Diaries", 4.5, "Short story between Exit Strategy and Network Effect"),
    387: ("The Murderbot Diaries", 7.5, "Novella (2025), published between 7 and 8"),
    225: ("Lady Astronaut", 4.5, "Original novelette; Goodreads numbers it 4.5"),
    87: ("Lock In", 0.5, "Prequel novella"),
    643: ("Mr. Penumbra's 24-Hour Bookstore", 0.5, "Prequel"),
    459: ("Mr. Penumbra's 24-Hour Bookstore", 1, ""),
    421: ("Machineries of Empire", 1, ""),
    386: ("Machineries of Empire", 2, ""),
    293: ("Merchant Princes", 1, "Omnibus (orig. books 1–2)"),
    130: ("Merchant Princes", 2, "Omnibus (orig. books 3–4)"),
    160: ("Merchant Princes", 3, "Omnibus (orig. books 5–6)"),
    364: ("Asian Saga", 1, "Chronological order, per the title; it was 3rd by publication"),
    269: ("Hainish Cycle", 5, "Per the title"),
    678: ("A Charm of Magpies", 1, ""),
    677: ("Wraith Kings", 1, ""),
    675: ("The Clocktaur War", 2, ""),
    656: ("A Court of Thorns and Roses", 1, ""),
    679: ("The Last Binding", 1, "Omnibus of books 1–3"),
    442: ("Aspects of the Divinity", 1, "Per the title"),
    445: ("Silo", 2, "First part of Shift (Silo #2)"),
    53: ("The Divine Comedy", 1, "Inferno"),
    25: ("The Divine Comedy", 3, "Paradiso"),
    478: ("Le Petit Nicolas", 1, ""),
    # Corrections to existing entries
    234: ("Culture", 10, "Was 9; Hydrogen Sonata is Culture #10 (its own title says so)"),
    463: ("Mistborn", 3.5, "Was 8; Secret History is set during/after book 3"),
    649: ("Dead Djinn Universe", 1, "Was 2; the publisher numbers A Master of Djinn #1 (the title even says ': 1')"),
    576: ("Oxford Time Travel", 1, "Was 2; Doomsday Book is the first novel"),
}

# Series that are really publisher imprints/collections, or otherwise bogus.
SERIES_CLEAR = {
    78: "'Beckett, Samuel' is not a series",
    321: "Publisher imprint (Allen Lane History)",
    395: "Publisher imprint (Cambridge Philosophy Classics)",
    547: "Publisher imprint (Inside Technology)",
    559: "Publisher imprint (Veritas Paperbacks)",
    666: "Bogus ('The English Edition' #3)",
    23: "Publisher imprint (One World Essentials)",
    11: "Publisher imprint (Economic Controversies)",
    390: "Publisher imprint (Platform Studies)",
    635: "Publisher imprint (ReVisioning History)",
    334: "Not a series (5 Love Languages #5)",
    475: "Liar's Poker and The Big Short aren't a series",
    294: "Liar's Poker and The Big Short aren't a series",
    545: "Publisher collection",
    644: "Accelerando isn't a Singularity Sky (Eschaton) book",
    199: "Bogus ('The Midnight World')",
    201: "The Martian is standalone",
    16: "Standalone graphic novel",
    177: "The Wind's Twelve Quarters is a story collection, not a series",
    422: "The 2014 1st edition isn't part of 'More Than Two Essentials' (#9 is the 2024 2nd edition)",
}

SERIES_RENAME = {
    "The Girl with the Dragon Tattoo (Lisbeth Salander/Millennium)": "Millennium",
    "The Locked Tomb Series": "The Locked Tomb",
    "Hell’s Library trilogy": "Hell's Library",
    "The Space Opera": "Space Opera",
    "Silo Series": "Silo",
    "The Paper Magician": "The Paper Magician",
}
SERIES_RENAME = {k: v for k, v in SERIES_RENAME.items() if k != v}

series_ops = []
for bid, (s, idx, note) in SERIES_SET.items():
    b = B[bid]
    series_ops.append({"id": bid, "title": b["title"], "author": api_name("; ".join(b["authors"])),
                       "old_series": b["series"], "old_index": b["series_index"] if b["series"] else None,
                       "series": s, "index": idx, "note": note,
                       "kind": "fix" if b["series"] else "add"})
series_clear_ops = [{"id": bid, "title": B[bid]["title"], "old_series": B[bid]["series"],
                     "old_index": B[bid]["series_index"], "note": note}
                    for bid, note in SERIES_CLEAR.items()]
series_rename_ops = [{"old": k, "new": v,
                      "books": sum(1 for b in B.values() if b["series"] == k)}
                     for k, v in SERIES_RENAME.items()]

# ---------------------------------------------------------------- titles

TITLE_OVERRIDES = {
    493: "Heir to the Empire",
    585: "Dark Force Rising",
    228: "The Last Command",
    189: "The Name of the Wind",
    434: "Oathbringer",
    450: "Edgedancer",
    314: "Seven Surrenders",
    675: "The Wonder Engine",
    497: "Harry Potter à l'école des sorciers",
    19: "Harry Potter et l'Ordre du Phénix",
    20: "Harry Potter et les Reliques de la Mort",
    362: "Shadow & Claw",
    192: "The Mongoliad: Book One",
    58: "Wool Omnibus",
    445: "First Shift: Legacy",
    429: "Off to Be the Wizard",
    640: "All Systems Red",
    461: "Mockingjay",
    294: "The Big Short: Inside the Doomsday Machine",
    600: "Candide",
    42: "You Mean I'm Not Lazy, Stupid or Crazy?!",
    86: "Unnatural Causes",
    481: "Judgment of Paris: California vs. France and the Historic 1976 Paris Tasting That Revolutionized Wine",
    18: "Merriam-Webster's French-English Translation Dictionary",
    375: "Systemantics: The Systems Bible",
    603: "Circe",
    15: "Redshirts",
    208: "The Luminaries",
    602: "Caste",
    594: "Command and Control: Nuclear Weapons, the Damascus Accident, and the Illusion of Safety",
    404: "Perfectly Legal: The Covert Campaign to Rig Our Tax System to Benefit the Super Rich—and Cheat Everybody Else",
    546: "Better Bouldering",
    178: "The Origins of Totalitarianism",
    466: "Memoirs of a Geisha",
    184: "1Q84",
    613: "Blindness",
    566: "I and Thou",
    88: "Une vie",
    17: "Making Silicon Valley",
    633: None,
    87: "Unlocked: An Oral History of Haden's Syndrome",
    213: "The Litany of Earth",
    195: "The Mind-Body Problem",
    639: "Black Panther (2016) #1",
    506: "Freedom™",
    237: "The Human Division #1: The B-Team",
    618: "Beyond Series Bundle (Books 1–3)",
    617: "Beyond Series Bundle (Books 4–6)",
    540: "Bridgerton Collection Volume 1",
    539: "Bridgerton Collection Volume 2",
    538: "Bridgerton Collection Volume 3",
    332: "The Abyss Beyond Dreams",
    561: "Lock In",
    630: "Annihilation",
    190: "The Mythical Man-Month: Essays on Software Engineering, Anniversary Edition",
    139: "The Structure of Scientific Revolutions",
    311: "Solve Your Child's Sleep Problems",
    492: "Hex Appeal",
}
# Trailing parentheticals that are store/edition noise rather than title.
NOISE_PAREN = re.compile(
    r"\s*[\(\[](?:[^()\[\]]*(?:Book \d+|Book One|Series|Trilogy|Novel|Cycle|Chronicle|Diaries|Files|Edition|"
    r"Kindle|Tie-in|Paperback|Harvest|Vintage|Oprah|Booker|Hugo|Wiley|MIT Press|patterns|Technology|"
    r"Frontiers|Fontana|Halcyon|Classics|Gift|ALA Notable|Controversies|Studies|History\)|Essentials|"
    r"Collections?|Universe|Saga)[^()\[\]]*|"
    + "|".join(re.escape(s) for s in sorted({b["series"] for b in B.values() if b["series"]}, key=len, reverse=True))
    + r"|Bridgertons|Wayfarers|Teixcalaan|Firefall|Unstoppable|Mercenary Librarians|Lady Astronaut|"
    r"Dresden Files|Laundry Files|Takeshi Kovacs|Empire Games|Six of Crows|The Scholomance|"
    r"Hidden Legacy|Lock In Series|Monk & Robot|The Serpent Gates|Alchemical Journeys|Kate Daniels|"
    r"Book \d|Tome \d)[\)\]]$")
NOISE_SUFFIX = re.compile(
    r"(?::\s*(?:\d+|A Novel|A Laundry Files Novel|A Mistborn Novel|A Thursday Next Novel|"
    r"A Novel of Temeraire|The Murderbot Diaries|Dungeon Crawler Carl Book \d+|A Dresden Files Novella|"
    r"A Tor\.Com Original|A Novel of the Commonwealth))$")
KEEP_AS_IS = {561}  # "Lock In: A Novel of the Near Future" handled by override

def clean_title(bid, t):
    if bid in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[bid] or t
    prev = None
    while prev != t:
        prev = t
        t = NOISE_PAREN.sub("", t).strip()
        t = NOISE_SUFFIX.sub("", t).strip()
    return t

title_ops = []
for bid, b in sorted(B.items(), key=lambda kv: kv[1]["title"].lower()):
    new = clean_title(bid, b["title"])
    if new != b["title"]:
        title_ops.append({"id": bid, "old": b["title"], "new": new})

# ---------------------------------------------------------------- misc

PUBDATE = {
    670: ("2023-02-17", "Ebook/audio first release (approx; worth spot-checking)"),
    680: ("2024-02-20", "Ebook/audio first release (approx; worth spot-checking)"),
    677: ("2015-06-10", "Per audiobook/Smashwords listings"),
    675: ("2018-06-11", ""),
    639: ("2016-04-06", "Black Panther (2016) #1"),
    16: ("2014-01-01", "Original B&W edition; year only"),
}
pubdate_ops = [{"id": bid, "title": B[bid]["title"], "old": (B[bid]["pubdate"] or "")[:10],
                "new": d, "note": n} for bid, (d, n) in PUBDATE.items()]

PUBLISHER_MERGE = {
    "Simon &amp; Schuster": "Simon & Schuster",
    "Simon & Schuster, Inc.": "Simon & Schuster",
    "Orbit/Yen": "Orbit / Yen",
    "The MIT Press": "MIT Press",
    "Cambridge, Mass. : MIT Press": "MIT Press",
    "Penguin Group, USA": "Penguin Group US",
    "HarperCollinsPublishers": "HarperCollins",
    "HarperCollins Publishers": "HarperCollins",
    "HarperCollins e-books": "HarperCollins",
    "ATlantic Monthly Press": "Atlantic Monthly Press",
    "The University Of Chicago Press, Chicago": "University of Chicago Press",
    "Oxford University Press - A": "Oxford University Press",
    "OUP Oxford": "Oxford University Press",
    "Globe Pequot - A": "Globe Pequot",
    "Verso Books": "Verso",
    "The Free Press": "Free Press",
    "Bloomsbury_US": "Bloomsbury Publishing",
    "Penguin Press HC, The": "Penguin Press",
    "Houghton Mifflin Harcourt (www.hmhco.com)": "Houghton Mifflin Harcourt",
    "A Tom Doherty Associates Book": "Tom Doherty Associates",
    "Tom Doherty Associates Co.": "Tom Doherty Associates",
    "Pottermore Limited": "Pottermore Publishing",
    "St. Martin’s Press": "St. Martin's Press",
    "Pearson Education (USA)": "Pearson Education",
    "Cleis Press [Start]": "Cleis Press",
    "Unknown": None,
}
pub_counts = Counter(b["publisher"] for b in B.values() if b["publisher"])
publisher_ops = [{"old": k, "new": v, "books": pub_counts.get(k, 0)}
                 for k, v in PUBLISHER_MERGE.items() if pub_counts.get(k)]

misc_ops = [
    {"op": "set_language", "id": 17, "title": B[17]["title"], "value": "eng",
     "note": "Only book with no language set"},
    {"op": "remove_tags", "tags": ["https://archive.org/details/memoriesthatshap0000pugh",
                                   "IBM computers -- History; IBM 360 (Computer) -- History; Computer industry -- United States -- History",
                                   "ebook"],
     "note": "A URL tag, an un-split LCSH string (its parts already exist as separate tags), and a meaningless 'ebook' tag"},
    {"op": "delete_book", "id": 1, "title": B[1]["title"],
     "note": "Calibre's bundled 'Quick Start Guide' sample book"},
]

json.dump({
    "authors": author_ops,
    "book_authors": book_author_ops,
    "series": series_ops,
    "series_clear": series_clear_ops,
    "series_rename": series_rename_ops,
    "titles": title_ops,
    "pubdates": pubdate_ops,
    "publishers": publisher_ops,
    "misc": misc_ops,
}, open(out_path, "w"), indent=1, ensure_ascii=False)
print({k: len(v) for k, v in json.load(open(out_path)).items()})
