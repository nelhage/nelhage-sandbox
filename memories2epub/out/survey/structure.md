# Structure survey: front matter, chapter openers, running heads, subheads, back pages

All page numbers are PDF pages (printed page = PDF page - 16). Page size is 384.2 x 619.2 pt
(pdfinfo); coordinates below are in PDF points from the top-left, taken from
`pdftotext -bbox` on the existing text layer, cross-checked against out/render/p-NNN.png.

## 1. Front matter (PDF 1-16) and back pages (340-344)

| PDF | What it is | Disposition |
|---|---|---|
| 1 | Front board: dark cloth cover, no lettering at all (library rebind; a lighter patch at lower-left). | Drop (or keep as cover image only if nothing better exists; it is a featureless dark rectangle, so I would drop it). |
| 2 | Inside front board / pastedown, blank grey. | Drop |
| 3 | Free endpaper, blank. | Drop |
| 4 | Blank (verso of endpaper). | Drop |
| 5 | Half-title: "Memories That Shaped an Industry" in sans-serif, top-left. Bleed-through of p7 visible. | Keep as text (a half-title `<h1>` page) or drop; nothing else on it. |
| 6 | Series page: "The MIT Press Series in the History of Computing / I. Bernard Cohen and William Aspray, series editors" then two italic titles: *The Computer Comes of Age*, R. Moreau, 1984 / *Memories That Shaped an Industry*, Emerson W. Pugh, 1984. | Keep as text (short). |
| 7 | Title page (see quote below). | Keep as text; also a good candidate for an image if you want a cover. |
| 8 | Copyright page + CIP data + library stamp + handwritten call number in left margin. | Keep as text (strip stamp/handwriting). |
| 9 | Contents (see quote below). | Regenerate as EPUB nav; the printed page numbers are not useful. |
| 10 | Blank verso of Contents (only bleed-through; OCR text is garbage). | Drop |
| 11-12 | Series Foreword, pp. vii-viii (signed "I. Bernard Cohen / William Aspray", right-aligned on p12). | Keep as text. |
| 13-14 | Preface, pp. ix-x (signed "Emerson W. Pugh", right-aligned on p14). | Keep as text. |
| 15 | Second half-title: "Memories That Shaped an Industry" (same as p5). | Drop |
| 16 | Blank verso of second half-title (OCR text "yas | ... ioRpysokre" is bleed-through noise). | Drop |
| 340 | Blank rear free endpaper with heavy bleed-through/offset of the index; OCR gives 2929 chars of garbage. | Drop |
| 341 | Blank endpaper (faint marks only). | Drop |
| 342 | Blank (inside back board). | Drop |
| 343 | Library artefacts: barcode label ("R RBK QA76.8.I1015 P83 1984 / Pugh, Emerson W. / Memories that shaped an industry : decis / Messler Library/Fairleigh Dickinson Univ / 3 3208 00113 1184") and a "DATE DUE" card pocket with typed catalog card. | Drop |
| 344 | Back board, dark cloth, no lettering. | Drop |

There is no dedication page, no epigraph page, no list of figures, no acknowledgments section
(acknowledgments are the last paragraph of the Preface, p14).

### Title page text (p7)

Layout: bold sans-serif, centred. "MEMORIES" / "THAT SHAPED AN INDUSTRY" (large caps, two lines),
then subtitle "Decisions Leading to IBM System/360" (sans, smaller), then "Emerson W. Pugh"
(serif, smaller), then at the bottom "The MIT Press / Cambridge, Massachusetts / London, England".

- Title: **Memories That Shaped an Industry**
- Subtitle: **Decisions Leading to IBM System/360**
- Author: **Emerson W. Pugh**
- Publisher: **The MIT Press, Cambridge, Massachusetts; London, England**

### Copyright page text (p8) - clean transcription from the image

```
(c) 1984 by The Massachusetts Institute of Technology

All rights reserved. No part of this book may be reproduced in any form by any
electronic or mechanical means (including photocopying, recording, or information
storage and retrieval) without permission in writing from the publisher.

This book was computer composed by Caroline C. Coppola using YFL/PREP2 under VM/SP
at the IBM Thomas J. Watson Research Center.

Printed and bound in the United States of America.

Library of Congress Cataloging in Publication Data

Pugh, Emerson W.
    Memories that shaped an industry.

    (MIT Press series in the history of computing)
    Bibliography: p.
    Includes index.

    1. IBM computers--History.  2. IBM 360 (Computer)--History.
3. Computer industry--United States--History.
I. Title.  II. Series.
QA76.8.I1015P83 1984  001.64  83-18787
ISBN 0-262-16094-3
```

Metadata: ISBN 0-262-16094-3; year 1984; LCCN 83-18787; LC class QA76.8.I1015 P83 1984; Dewey
001.64; series "The MIT Press Series in the History of Computing" (eds. I. Bernard Cohen and
William Aspray); language en. Note the OCR text layer reads the LC number as "QA76.8.11015P83"
(I -> 1) - the image shows "I1015".

Things to strip on p8: handwritten call number in the left margin ("QA 76.8 I1015 P83 1984",
OCR'd as fragments "/A 76.8 I1015 P83 1984"), and the rubber stamp "Fairleigh Dickinson
University Library / MADISON, NEW JERSEY" below the ISBN (OCR: "Fairleigh Dickinson University
Library | MADISON, NEW JFRSHY" plus junk "oe hein ate, sae").

### Contents (p9) - authoritative chapter titles and printed page numbers

```
Series Foreword                      vii   (PDF 11)
Preface                               ix   (PDF 13)
1  The Postwar Challenge               1   (PDF 17)
2  Searching for Memory               34   (PDF 50)
3  A Memory from Whirlwind            62   (PDF 78)
4  Project SAGE                       93   (PDF 109)
5  Commercial Ferrite Core Memories  129   (PDF 145)
6  Project Stretch                   160   (PDF 176)
7  The Road to System/360            187   (PDF 203)
8  System/360 Memories               213   (PDF 229)
9  Managing Technological Change     248   (PDF 264)
References and Notes                 265   (PDF 281)
Chronology                           301   (PDF 317)
Index                                313   (PDF 329)
```

Capitalisation caveat: the Contents says "Searching for Memory" and "A Memory from Whirlwind",
but the chapter opener pages (p50, p78) and the running heads (p51 "Searching For Memory")
capitalise "For"/"From". Pick one (I'd follow the Contents).

## 2. Chapter opening page layout

Reference: p17 (chapter 1); measured in points from the page top.

- No running head and no page number on any opener page (p17, 50, 78, 109, 145, 176, 203, 229,
  264 all confirmed; same for p11, 13, 281, 317, 329).
- Chapter number: a single large bold sans-serif (Helvetica-like) digit, flush left at the text
  margin (x = 39 pt), top y = 42 pt, glyph height 29-30 pt (roughly a 40 pt font; body glyphs are
  ~9.3 pt tall incl. ascenders, i.e. roughly 11 pt type, so the number is ~3.5-4x body).
- Chapter title: directly under the number, flush left, same sans-serif face, regular weight
  (looks medium), Title Case, top y = 72.6 pt, glyph height 14 pt (about 18-20 pt type, ~1.7x
  body). The number and title nearly touch (number bottom 71.4 pt, title top 72.6 pt).
- Then a large gap: first body baseline area begins at y = 153 pt (about 80 pt / 5.5 lines of
  white below the title).
- No epigraph on any chapter opener. No author/date lines.
- First paragraph: NOT indented (flush left) on all nine chapters (p17, 50, 78, 109, 145, 176,
  203, 229, 264). Subsequent paragraphs are indented ~15 pt (first-line indent, no extra space
  between paragraphs).
- Body face is a serif (Times-like), justified, very loose word spacing (computer-composed
  with YFL/PREP2, see p8), ~14.7 pt leading, ~26 lines to a full page.
- The chapter's first subhead can appear on the opener page itself (p50 "A First Choice",
  p109 "Selecting IBM", p145 "A Small 3D Array Memory Product", p176 "Seeking Government
  Funds", p203 "Competitive Chaos in IBM", p229 "Management Pressures"); chapters 1, 3 and 9
  have none on the opener.
- Layout is identical for all nine chapters. Only the horizontal scan offset differs (text
  margin drifts between x = 29 and 45 pt from page to page, e.g. p51 body x 29-342 vs p52
  x 35-354), so match by "big glyph near top-left, nothing above y = 40 pt" rather than by exact x.
- p78 (chapter 3) is the only opener with a block quotation in the first screen (see section 5).

Non-chapter openers (Series Foreword p11, Preface p13, References and Notes p281, Chronology
p317, Index p329) differ from chapter openers in these ways:
- No number. The heading is the same sans-serif face at the title size (glyph height 12.5-14 pt),
  flush left, but positioned higher, at y = 46-55 pt (where the chapter number would be).
- The gap below the heading is slightly smaller: body starts at y = 127-141 pt instead of 153.
- First paragraph is indented on p11 (Series Foreword) - the only opener with an indented first
  paragraph - and flush on p13 (Preface) and p281 (References intro paragraph).
- p281 has an unindented intro paragraph, then a bold serif "Chapter 1" subhead (see section 4),
  then numbered notes.
- p317 (Chronology) is a two-column hanging list: bold-ish date column (x = 33 pt: "1911",
  "8/44", "3/45", ...) and an entry column starting at x = 94 pt, one blank line between entries.
- p329 (Index) is set in two columns (gutter at x ~ 190 pt), first column starts y = 140 pt,
  entries with hanging indents; there are NO letter-group headings (A, B, ...), entries just run
  on alphabetically (checked p329-339; grep for single-letter lines finds none).

## 3. Running heads and page numbers on body pages

Every non-opener page from p12 through p339 carries a single running-head line and nothing
else above the body:

- Vertical position: the head sits at y = 14-24 pt from the top (yMin 13.6-20.7, yMax 22-29;
  varies with scan skew). The first body line starts at y = 47-61 pt. So anything whose bbox top
  is < ~32 pt is running head; body always begins > 45 pt.
- Font: same serif as the body, roman, glyph height ~8.5 pt (versus ~9.3 for body), i.e. a
  point or so smaller than body; page numbers use the same size as the head.
- Verso (even printed page = even PDF page): **page number at the left margin, chapter label
  at the right margin**. E.g. p18: "2" at x = 40, "Chapter 1" right-aligned ending x = 352;
  p52: "36" left, "Chapter 2" right; p20 "4 ... Chapter 1"; p40 "24 ... Chapter 1";
  p62 "46 ... Chapter 2"; p88 "72 ... Chapter 3"; p188 "172 ... Chapter 6".
- Recto (odd printed = odd PDF): **chapter title at the left margin, page number at the right
  margin**. E.g. p19: "The Postwar Challenge" at x = 37, "3" at x = 345-349; p51 "Searching
  For Memory ... 35"; p25 "The Postwar Challenge ... 9"; p261 "System/360 Memories ... 245";
  p263 "System/360 Memories ... 247".
- Front matter uses the section title instead of "Chapter N" and roman numerals: p12 "viii
  ... Series Foreword" (verso), p14 "x ... Preface" (verso). (p11/p13 are openers, no head.)
- Back matter same pattern with section title on both sides: p282 "266 ... References and
  Notes", p283 "References and Notes ... 267", p318 "302 ... Chronology", p319 "Chronology
  ... 303", p330 "314 ... Index", p331 "Index ... 315", p339 "Index ... 323".
- No footers anywhere; page numbers are only in the head line.
- The OCR text layer renders the head as the first non-blank line of each page in the form
  "N<spaces>Chapter M" (verso) or "Title<spaces>N" (recto), but with frequent errors:
  "Chapter 16" for "Chapter 1 / 6" (p22), "Chapter 110" (p26), "Chapter |" (p28),
  "The Postwar Challenge i" (p23), "Chonology" (p318, p320), "Index ply]" (p333). So match the
  head by position (top of bbox < 32 pt) and by pattern (`^\d+\s+Chapter \d` / `^Chapter`,
  `\s\d+$`, known section titles), not by exact string.
- Pages where a chapter ends early (e.g. p263: two lines of text then blank) still carry the
  head.

Scanner artefact worth knowing about: a small dark speck/tick (looks like "^" or "*") appears in
the left margin at roughly x = 43 pt, y = 405-410 pt (150 dpi render: x ~ 70-90 px, y ~ 840-860
px) on most pages (seen on p12, 14, 18, 20, 36, 40, 50, 52, 62, 88, 176, 188, 264, 282, 340,
341, 343). The IA OCR sometimes turns it into a leading "*" on the adjacent body line (p20
line 29, p178 line 52, p188 line 52, p232 line 35). These are NOT footnotes - the book has no
footnotes, only superscript endnote numbers - so a pipeline should ignore an isolated
"*"/"^" at the start of a line at that height.

## 4. Internal section headings (subheads)

Yes. Every chapter has 4-10 subheads; a single level only (no sub-subheads seen). Style,
confirmed on p19 ("The Watson Legacy"), p50 ("A First Choice"), p109 ("Selecting IBM"),
p145 ("A Small 3D Array Memory Product"), p176 ("Seeking Government Funds"), p203
("Competitive Chaos in IBM"), p229 ("Management Pressures"):

- Bold serif, Title Case, flush left at the text margin, same point size as body (glyph
  height 9.1 pt = body). Bold weight is the only distinction, plus spacing.
- About one blank line (~14 pt extra) above the subhead; NO extra space below - the next
  line follows at normal leading.
- The paragraph immediately after a subhead is flush left (not indented); later paragraphs
  are indented. So "bold short line + following unindented paragraph" is the signature.
- Subheads never carry numbers, never end in punctuation, are 2-6 words, typically 1 line.

Full candidate list from the text layer (PDF page: text), after removing chapter titles and
figure-caption headings (see section 5):

- Ch 1: 19 The Watson Legacy; 27 Developments at the Moore School; 31 UNIVAC, ERA, and
  Remington Rand; 36 New Leaders at IBM; 42 Entering the Electronic Computer Market
- Ch 2: 50 A First Choice; 61 Magnetic Cores at IBM; 68 Feasibility Models
- Ch 3: 82 Memory Considerations; 83 New Objectives for Whirlwind; 84 A New Memory Proposal;
  90 Joining Forces; 94 First Ferrite Core Main Memory; 97 The Disputed Invention;
  103 More Inventors; 106 A Retrospective View
- Ch 4: 109 Selecting IBM; 110 Staffing the Project; 113 Cooperative Design Effort;
  121 Ferrite Core Procurement; 123 Core Testing and Wiring; 128 XD-1 and XD-2 Memories;
  133 Core Fabrication in IBM; 140 Realizing the Dream
- Ch 5: 145 A Small 3D Array Memory Product; 148 Indecisions on Main Memories; 153 The
  Decision; 154 Developing Main Memories; 160 Introducing Transistors; 166 Ferrite Core
  Fabrication; 170 Array Wiring Improvements
- Ch 6: 176 Seeking Government Funds; 182 Problems at Two Microseconds; 187 Program
  Management; 199 The Stretch Problem
- Ch 7: 203 Competitive Chaos in IBM; 206 A Plan Evolves; 210 The SPREAD Report; 215 Computer
  Control Stores; 220 Growing Requirements for Memory; 224 The Forrester Patent Settlement
- Ch 8: 229 Management Pressures; 236 Transformer Read-Only Store (TROS); 239 Card Capacitor
  Read-Only Store (CCROS); 241 A Fast Control Store; 243 Balanced Capacitor Read-Only Store
  (BCROS); 245 Memories from Mecca; 250 A Fast Main Memory; 253 Progress in Manufacturing;
  258 A Successful Technology; 259 Cost-Performance Limits
- Ch 9: 265 The Manufacturing Buildup; 267 Not Good Enough; 272 Innovation and Risk;
  273 Management Precepts; 278 The IBM Team

(Only p19/50/109/145/176/203/229 were visually verified; the rest come from the short-line
grep and should be verified by the body-page survey. Lines such as "Ralph Palmer" (p39, 40),
"Mike Haynes" (p55, 62), "Erich Bloch" (p73, 76), "Jay Forrester" (p79), "John Gibson" (p134),
"Moe Every" (p191), "Moe Every with Bill Rhodes" (p198) are bold caption headings under
photographs, not subheads - see below.)

Back matter subheads: References and Notes uses a bold serif "Chapter 1" ... "Chapter 9"
heading (p281 shows "Chapter 1"; same size as body, bold, flush left, blank line above and
below) to group the numbered notes.

## 5. Other structural features

- **Endnotes.** Body text carries superscript arabic note numbers (p17: "peace.1",
  "munitions.2", "production.3", "Channel.3" - numbers can repeat when the same source is
  re-cited). They are keyed per chapter to "References and Notes" (p281-316), which is a
  numbered hanging-indent list: number at x = 31 pt, text at x = 72 pt, a blank line between
  entries (p281, p282); entries contain italic journal/book titles. Italic book titles also
  occur in body text (p11 *Encyclopedia of Computer Science and Engineering*, p14 *From ENIAC
  to UNIVAC*, p264 *Fortune*). Small caps are used for acronyms in body text (p50 "SSEC",
  p145 "TPM", p264 "I/O", p51 "EDVAC", "UNIVAC", "ENIAC") - the OCR lowercases or mangles
  these sometimes.
- **Block quotations.** p78: a multi-line extract set full measure, NOT indented, same type
  size as body (glyph height 9.1-9.3 pt) but tighter leading (12.2 pt vs 14.7 pt body), with a
  blank line (~28 pt gap) above and below. The paragraph after it resumes with a normal
  indent. Expect more of these where the text quotes memos/reports at length; detection cue
  is the leading change plus surrounding whitespace, not indentation.
- **Figures and captions.** Many pages are wholly or mostly photographs/diagrams (text layer
  < 1500 chars on p25, 32, 37, 40, 47, 49, 58, 62, 66, 69, 71, 76, 81, 86, 88, 91, 96, 99, 126,
  127, 136, 138, 141, 144, 163, 172, 174, 185, 190, 193, 198, 202, 217, 236, 239, 243, 248, 254,
  256, 261, 263, 271, 280 - p263 and p280 are simply short chapter-end pages). Figures are
  unnumbered (no "Figure N"/"Table N" anywhere in the text layer). Caption format (p25, p40,
  p62, p88, p261): a bold serif one-line caption title flush left ("IBM punched card",
  "SSEC memory components", "Ralph Palmer", "Mike Haynes", "Early magnetic cores", "Ferrite
  core production") immediately followed by a caption paragraph in a slightly smaller serif
  (glyph height 8.2-8.6 pt vs 9.1-9.3 body), tight leading (~11 pt), unindented, often ending
  with "(Photograph courtesy of IBM Archives.)" / "(Photograph courtesy of the MITRE Corporation
  Archives.)". Figure pages keep the normal running head. Diagrams (p261 log-log chart) contain
  their own axis text in sans-serif caps ("BILLIONS OF FERRITE CORES PER YEAR", "YEAR OF
  PRODUCTION") that leaks into the OCR as short all-caps lines (also p32 "ELECTRO-ACOUSTIC
  DELAY LINE", p157 "X SWITCHES", p196 "BIT A CURRENT", p236 "WORD CONDUCTOR", p99 "JAN A.
  RAJCHMAN" from a patent drawing).
- **Tables / lists.** None found in the body via the text layer (no "Table" anywhere).
  The Chronology (p317-328) is effectively a two-column table (date | event); the Index
  (p329-339) is a two-column hanging-indent list with "See"/"See also" in italics.
- **Signatures.** Series Foreword ends (p12) with two right-aligned lines "I. Bernard Cohen /
  William Aspray"; Preface ends (p14) with right-aligned "Emerson W. Pugh".
- **No dedication, no epigraphs, no footnotes, no part divisions.** Chapters simply end
  mid-page (p263) and the next chapter starts on a new page (recto not enforced: chapter
  openers fall on PDF 50 (even) and 78 (even), i.e. printed 34 and 62, so openers can be
  versos).
