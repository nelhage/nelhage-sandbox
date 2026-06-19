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
  detail: enemies fought, turns, HP/damage, gold, cards picked or removed,
  upgrades, relics & potions gained, and event/rest-site choices.
- **Vitals** — an HP-over-time chart for the whole run (current vs. max HP, with
  act dividers). Hovering a point cross-highlights the matching floor node.
- **Relics & Potions** — **hover for** name, rarity, description, and flavor.
- **Deck** — every card (grouped with counts), color-coded by rarity, with `+`
  for upgrades and `✦` for enchantments. **Hover for** the full card: cost,
  type, rarity, art, description (upgraded if upgraded), keywords, and the floor
  it was added.

An `index.html` lists every run.

## Usage

```sh
node generate.mjs                  # build every runs/*.run + the index
node generate.mjs runs/NAME.run    # build specific run(s) (index still covers all)
```

Output is written to `out/` (a symlink to the public web dir). Card / relic /
potion / character images are copied, de-duplicated, into `out/assets/`.

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

`nix develop ..#sts2-log` provides Node and Chromium (the latter only used for
screenshot-based verification). The directory's `.envrc` selects this shell.
