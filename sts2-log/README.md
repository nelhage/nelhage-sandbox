# sts2-log → HTML

Turns a Slay the Spire II run log (`*.run`, JSON) into a self-contained,
interactive HTML page you can share with friends.

Live output: <https://nelhage.com/f/sts2.runs/>

## What you get

Per run, a single page with:

- **Header** — character, win/loss/abandoned, final HP & gold, floor count,
  run time, ascension, date, seed, and game version.
- **The Climb** — the act-by-act path as a row of nodes (combat / elite / boss /
  rest / shop / treasure / event / ancient). **Hover any node** for that floor's
  detail; **click a node** to jump to its row in the floor-by-floor table below.
- **Vitals** — an HP-over-time chart for the whole run (current vs. max HP, with
  act dividers). Hovering a point cross-highlights the matching floor node.
- **Relics & Potions** — **hover for** name, rarity, description, and flavor;
  **click** to open the item's sts2-wiki page.
- **Deck** — every card (grouped with counts), color-coded by rarity, with `+`
  for upgrades and `✦` for enchantments. **Hover for** the full card (cost,
  type, rarity, art, description, keywords, floor added); **click** to open the
  card's sts2-wiki page.
- **Floor by floor** — a table with every floor's information laid out
  explicitly (no hovering): room/encounter, enemies, HP & gold deltas, the full
  card reward **offered** (taken vs. skipped), cards removed/upgraded/
  transformed, relics & potions gained, and event/rest choices. Every entity
  links to its sts2-wiki page.

An `index.html` lists every run.

Wiki links point at <https://drmaciver.github.io/sts2-wiki/>. Slugs are derived
the same way the wiki generates them — `slugify(title)`, with a `-character`
(cards) / `-class_name` (everything else) suffix on title collisions (e.g. each
character's Strike/Defend).

## Usage

```sh
node generate.mjs                  # build every runs/*.run + the index
node generate.mjs runs/NAME.run    # build specific run(s) (index still covers all)
node generate.mjs --og-force       # also re-render social cards (default: only missing ones)
node generate.mjs --no-og          # skip social cards entirely
```

Output is written to `out/` (a symlink to the public web dir). Card / relic /
potion / character images are copied, de-duplicated, into `out/assets/`.

### Social previews

Each run page carries OpenGraph / Twitter-card metadata, so sharing a link
yields a preview. The card image (`out/assets/og/<id>.png`, 1200×630) is
composed from the run — character portrait, result, key stats, and relic strip —
and rasterized with the Chromium from the dev shell. Cards are cached: a normal
build only renders ones that don't exist yet (use `--og-force` to redo them).

If Chromium isn't available, the build still emits the metadata but points
`og:image` at the character portrait (a smaller `summary` card) and prints a
warning. Run inside `nix develop ..#sts2-log` (or via direnv) to get Chromium.

## How it works

- **Data** comes from the [`sts2-wiki`](../sts2-wiki) data dumps under
  `sts2-wiki/data/v<version>/`. The run's `build_id` is matched to the nearest
  available data version (dumps only exist for a subset of releases).
- Run IDs (`CARD.DAGGER_SPRAY`, `RELIC.…`, `ENCOUNTER.…`, `EVENT.…`) are mapped
  to wiki entries by `loc_key` (cards/relics/potions/monsters/encounters) or by
  PascalCased `class_name` (events).
- Each page is a single HTML file with inline CSS/JS and an embedded tooltip
  registry; the only external references are the shared `assets/` images.

Source lives in `generate.mjs` (build), `style.css` (styling), `tip.js`
(tooltip + chart interactivity).

## Dev environment

`nix develop ..#sts2-log` provides Node and Chromium (Chromium renders the
social-preview cards, and is handy for screenshot-based verification). The
directory's `.envrc` selects this shell.
