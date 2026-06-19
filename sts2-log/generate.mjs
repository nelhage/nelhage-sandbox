#!/usr/bin/env node
// Slay the Spire II run-log -> shareable interactive HTML.
//
// Usage:
//   node generate.mjs                 # build all runs in runs/ + index
//   node generate.mjs runs/x.run ...  # build specific runs (+ index of all)
//
// Output goes to out/  (symlinked to a public web dir).
import fs from 'fs';
import path from 'path';
import { spawn, execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const RUNS_DIR = path.join(ROOT, 'runs');
const WIKI = path.join(ROOT, 'sts2-wiki');
const DATA_DIR = path.join(WIKI, 'data');
const IMG_SRC = path.join(WIKI, 'site', 'public', 'images');
const OUT = path.join(ROOT, 'out');
const ASSETS = path.join(OUT, 'assets');

// ---------------------------------------------------------------------------
// Version selection: data/ only has a subset of build versions. Pick the
// nearest available data dump (preferring not-newer on ties).
// ---------------------------------------------------------------------------
const AVAILABLE_VERSIONS = fs.readdirSync(DATA_DIR)
  .filter(d => /^v\d/.test(d) && fs.statSync(path.join(DATA_DIR, d)).isDirectory())
  .map(d => d.slice(1));

const vtuple = v => v.replace(/^v/, '').split('.').map(Number);
const vnum = v => { const t = vtuple(v); return (t[0] || 0) * 1e6 + (t[1] || 0) * 1e3 + (t[2] || 0); };

function pickVersion(buildId) {
  const target = vnum(buildId);
  let best = null, bestScore = Infinity;
  for (const v of AVAILABLE_VERSIONS) {
    const d = Math.abs(vnum(v) - target);
    // tie-break: prefer the older/equal version
    const score = d * 2 + (vnum(v) > target ? 1 : 0);
    if (score < bestScore) { bestScore = score; best = v; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Wiki data loading + indexing (cached per version).
// ---------------------------------------------------------------------------
const dataCache = new Map();
function loadData(version) {
  if (dataCache.has(version)) return dataCache.get(version);
  const dir = path.join(DATA_DIR, `v${version}`);
  const readJson = f => { try { return JSON.parse(fs.readFileSync(path.join(dir, f))); } catch { return null; } };
  const indexByLocKey = arr => {
    const m = new Map();
    for (const e of arr || []) if (e.loc_key) m.set(e.loc_key, e);
    return m;
  };
  const indexByClass = arr => {
    const m = new Map();
    for (const e of arr || []) if (e.class_name) m.set(e.class_name, e);
    return m;
  };
  // events are individual files keyed by class_name
  const events = new Map();
  const evDir = path.join(dir, 'events');
  if (fs.existsSync(evDir)) {
    for (const f of fs.readdirSync(evDir)) {
      if (!f.endsWith('.json')) continue;
      try {
        const e = JSON.parse(fs.readFileSync(path.join(evDir, f)));
        if (e.class_name) events.set(e.class_name, e);
      } catch {}
    }
  }
  const enchArr = readJson('enchantments.json') || [];
  const enchants = indexByClass(enchArr);
  const cardsArr = readJson('cards.json') || [];
  const relicsArr = readJson('relics.json') || [];
  const potionsArr = readJson('potions.json') || [];
  const monstersArr = readJson('monsters.json') || [];
  const encountersArr = readJson('encounters.json') || [];
  const data = {
    version,
    cards: indexByLocKey(cardsArr),
    relics: indexByLocKey(relicsArr),
    potions: indexByLocKey(potionsArr),
    monsters: indexByLocKey(monstersArr),
    encounters: indexByLocKey(encountersArr),
    ancients: indexByLocKey(readJson('ancients.json')),
    characters: indexByLocKey(readJson('characters.json')),
    acts: indexByClass(readJson('acts.json')),
    events,
    enchants,
    // class_name -> wiki slug
    cardSlugs: buildSlugMap(cardsArr, 'character'),
    relicSlugs: buildSlugMap(relicsArr, 'class_name'),
    potionSlugs: buildSlugMap(potionsArr, 'class_name'),
    monsterSlugs: buildSlugMap(monstersArr, 'class_name'),
    encounterSlugs: buildSlugMap(encountersArr, 'class_name'),
    eventSlugs: buildSlugMap([...events.values()], 'class_name'),
  };
  dataCache.set(version, data);
  return data;
}

// ---------------------------------------------------------------------------
// id helpers
// ---------------------------------------------------------------------------
const idSuffix = id => (id || '').split('.').slice(1).join('.'); // CARD.DAGGER_SPRAY -> DAGGER_SPRAY
const pascal = snake => (snake || '').toLowerCase().split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join('');
const snakeLower = cls => (cls || '').replace(/([A-Z])/g, (m, p, off) => (off > 0 ? '_' : '') + p).toLowerCase();
const esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// ---------------------------------------------------------------------------
// sts2-wiki linking. Slugs mirror the wiki's generators: slugify(title), with a
// `-{character}` (cards) / `-{class_name}` (everything else) suffix appended to
// later entries on a title collision (e.g. each character's Strike/Defend).
// ---------------------------------------------------------------------------
const WIKI_BASE = 'https://drmaciver.github.io/sts2-wiki';
// Public location these pages are served from (for absolute og: URLs).
const SITE_BASE = 'https://nelhage.com/f/sts2.runs';
const cleanTitle = t => String(t || '').replace(/#[A-Z]\{[^}]*\}/g, '').trim(); // strip game template artifacts
const slugify = title => cleanTitle(title).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
const wikiUrl = (kind, slug) => slug ? `${WIKI_BASE}/${kind}/${slug}/` : null;
const uniqByTitle = refs => { const seen = new Set(); return refs.filter(r => r && !seen.has(r.title) && seen.add(r.title)); };

// Build class_name -> slug for an ordered list, replicating collision handling.
function buildSlugMap(entries, suffixField) {
  const seen = new Set();
  const map = new Map();
  for (const e of entries) {
    if (!e || !e.class_name) continue;
    let slug = slugify(e.title || e.class_name);
    if (!slug) continue;
    if (seen.has(slug)) slug = `${slug}-${String(e[suffixField] || e.class_name).toLowerCase()}`;
    seen.add(slug);
    map.set(e.class_name, slug);
  }
  return map;
}

// Convert in-game BBCode-ish / wiki markup to safe HTML.
function markup(str, vars) {
  if (str == null) return '';
  let s = String(str);
  // fill {Var} placeholders from a vars array [{type,base_value}]
  if (vars && vars.length) {
    const vmap = {};
    for (const v of vars) if (v.type != null) vmap[v.type] = v.base_value;
    s = s.replace(/\{([A-Za-z0-9_]+)(?::[^}]*)?\}/g, (m, name) => {
      for (const t in vmap) if (name === t || name === t + 'Power' || name.startsWith(t)) return vmap[t];
      return m;
    });
  }
  s = esc(s);
  // already-escaped <br>/<span> from description_html got escaped; restore known-safe ones
  s = s.replace(/&lt;br&gt;/g, '<br>')
       .replace(/&lt;span class=&quot;desc-gold&quot;&gt;/g, '<span class="kw">')
       .replace(/&lt;span class=&quot;desc-purple&quot;&gt;/g, '<span class="kw kw-p">')
       .replace(/&lt;\/span&gt;/g, '</span>');
  // bbcode tags
  s = s.replace(/\[gold\]/g, '<span class="kw">').replace(/\[\/gold\]/g, '</span>')
       .replace(/\[red\]/g, '<span class="kw kw-r">').replace(/\[\/red\]/g, '</span>')
       .replace(/\[purple\]/g, '<span class="kw kw-p">').replace(/\[\/purple\]/g, '</span>')
       .replace(/\[green\]/g, '<span class="kw kw-g">').replace(/\[\/green\]/g, '</span>')
       .replace(/\[star\]/g, '★').replace(/\[energy\]/g, '⚡')
       .replace(/\[[^\]]*\]/g, ''); // drop any remaining tags
  s = s.replace(/\{([A-Za-z0-9_]+)(?::[^}]*)?\}/g, '<span class="kw">$1</span>'); // leftover placeholders
  s = s.replace(/\n/g, '<br>');
  return s;
}

// ---------------------------------------------------------------------------
// asset copying (dedup into out/assets/<kind>/)
// ---------------------------------------------------------------------------
const copied = new Set();
function copyAsset(srcAbs, kind, name) {
  const rel = path.join(kind, name);
  const dest = path.join(ASSETS, rel);
  const webPath = `assets/${kind}/${name}`;
  if (copied.has(rel)) return webPath;
  if (!fs.existsSync(srcAbs)) return null;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(srcAbs, dest);
  copied.add(rel);
  return webPath;
}

function cardImage(card) {
  if (!card || !card.character || !card.class_name) return null;
  const charDir = card.character.toLowerCase();
  const file = snakeLower(card.class_name) + '.png';
  // try main atlas, then beta fallback
  for (const sub of [path.join(charDir, file), path.join(charDir, 'beta', file)]) {
    const abs = path.join(IMG_SRC, 'card_atlas', sub);
    if (fs.existsSync(abs)) return copyAsset(abs, 'cards', `${charDir}__${file}`);
  }
  return null;
}
function relicImage(relic) {
  if (!relic || !relic.image) return null;
  const abs = path.join(IMG_SRC, 'relic_atlas', relic.image + '.png');
  return copyAsset(abs, 'relics', relic.image + '.png');
}
function potionImage(potion) {
  if (!potion || !potion.image) return null;
  const abs = path.join(IMG_SRC, 'potion_atlas', potion.image + '.png');
  return copyAsset(abs, 'potions', potion.image + '.png');
}
function characterImage(charKey) {
  const file = `character_icon_${(charKey || '').toLowerCase()}.png`;
  const abs = path.join(IMG_SRC, 'characters', file);
  return copyAsset(abs, 'characters', file);
}

// ---------------------------------------------------------------------------
// Run enrichment
// ---------------------------------------------------------------------------
const RARITY_ORDER = { Starter: 0, Common: 1, Uncommon: 2, Rare: 3, Special: 4, Boss: 5 };
const TYPE_ORDER = { Attack: 0, Skill: 1, Power: 2, Status: 3, Curse: 4, Quest: 5 };

function resolveCardEntry(deckCard, data) {
  const key = idSuffix(deckCard.id);
  const wiki = data.cards.get(key);
  const upgraded = deckCard.current_upgrade_level > 0;
  let title = wiki ? wiki.title : key.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
  let ench = null;
  if (deckCard.enchantment && deckCard.enchantment.id) {
    const ecls = pascal(idSuffix(deckCard.enchantment.id));
    const e = data.enchants.get(ecls);
    ench = { title: e ? e.title : ecls, amount: deckCard.enchantment.amount };
  }
  return {
    key, title, upgraded, ench,
    floor: deckCard.floor_added_to_deck,
    img: cardImage(wiki),
    url: wiki ? wikiUrl('cards', data.cardSlugs.get(wiki.class_name)) : null,
    cost: wiki ? (wiki.x_cost ? 'X' : (wiki.energy_cost >= 0 ? wiki.energy_cost : null)) : null,
    type: wiki ? wiki.type : null,
    rarity: wiki ? wiki.rarity : null,
    character: wiki ? wiki.character : null,
    keywords: wiki ? (wiki.keywords || []) : [],
    desc: wiki ? markup(upgraded ? (wiki.upgraded_description_html || wiki.upgraded_description_plain || wiki.upgraded_description) : (wiki.description_html || wiki.description_plain || wiki.description)) : '',
  };
}

// Lightweight {title, url} references for floor-level listings.
function refCard(id, data) {
  const key = idSuffix(id);
  const w = data.cards.get(key);
  return { title: w ? w.title : titleCase(key), url: w ? wikiUrl('cards', data.cardSlugs.get(w.class_name)) : null, rarity: w ? w.rarity : null };
}
function refRelic(idOrKey, data) {
  const key = idOrKey.includes('.') ? idSuffix(idOrKey) : idOrKey;
  const w = data.relics.get(key);
  return { title: w ? w.title : titleCase(key), url: w ? wikiUrl('relics', data.relicSlugs.get(w.class_name)) : null };
}
function refPotion(id, data) {
  const key = idSuffix(id);
  const w = data.potions.get(key);
  return { title: w ? w.title : titleCase(key), url: w ? wikiUrl('potions', data.potionSlugs.get(w.class_name)) : null };
}
function refMonster(id, data) {
  const key = idSuffix(id);
  const w = data.monsters.get(key);
  return { title: cleanTitle(w ? w.title : titleCase(key)), url: w ? wikiUrl('monsters', data.monsterSlugs.get(w.class_name)) : null };
}

function resolveRelic(id, data) {
  const key = idSuffix(id);
  const w = data.relics.get(key);
  return {
    key,
    title: w ? w.title : key.replace(/_/g, ' '),
    rarity: w ? w.rarity : null,
    img: relicImage(w),
    url: w ? wikiUrl('relics', data.relicSlugs.get(w.class_name)) : null,
    desc: w ? markup(w.description, w.vars) : '',
    flavor: w ? markup(w.flavor) : '',
  };
}

function resolvePotion(id, data) {
  const key = idSuffix(id);
  const w = data.potions.get(key);
  return {
    key,
    title: w ? w.title : key.replace(/_/g, ' '),
    rarity: w ? w.rarity : null,
    img: potionImage(w),
    url: w ? wikiUrl('potions', data.potionSlugs.get(w.class_name)) : null,
    desc: w ? markup(w.description, w.vars) : '',
  };
}

function resolveEncounter(modelId, monsterIds, data) {
  const key = idSuffix(modelId);
  const w = data.encounters.get(key);
  const seen = new Set();
  const monsters = [];
  for (const m of monsterIds || []) {
    const r = refMonster(m, data);
    if (seen.has(r.title)) continue;
    seen.add(r.title);
    monsters.push(r);
  }
  return {
    title: cleanTitle(w ? w.title : (key ? key.replace(/_/g, ' ') : null)),
    url: w ? wikiUrl('encounters', data.encounterSlugs.get(w.class_name)) : null,
    isWeak: w ? w.is_weak : false,
    monsters,
  };
}

function eventName(modelId, data) {
  const cls = pascal(idSuffix(modelId));
  const w = data.events.get(cls);
  return {
    title: w ? w.title : titleCase(idSuffix(modelId)),
    url: w ? wikiUrl('events', data.eventSlugs.get(w.class_name)) : null,
  };
}

// Pull a readable label out of an event_choice loc title key.
function eventChoiceLabel(choice) {
  if (!choice || !choice.title || !choice.title.key) {
    if (choice && choice.variables) {
      const v = Object.values(choice.variables)[0];
      if (v && v.string_value) return v.string_value;
    }
    return null;
  }
  const parts = choice.title.key.split('.');
  const oi = parts.indexOf('options');
  let label = oi >= 0 && parts[oi + 1] ? parts[oi + 1] : parts[parts.length - 1];
  label = label.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
  const v = choice.variables ? Object.values(choice.variables)[0] : null;
  if (v && v.string_value) label += ` (${v.string_value})`;
  return label;
}

const MAP_ICON = {
  monster: { cls: 'combat', label: 'Combat' },
  elite: { cls: 'elite', label: 'Elite' },
  boss: { cls: 'boss', label: 'Boss' },
  rest_site: { cls: 'rest', label: 'Rest Site' },
  shop: { cls: 'shop', label: 'Shop' },
  treasure: { cls: 'treasure', label: 'Treasure' },
  unknown: { cls: 'event', label: 'Unknown' },
  event: { cls: 'event', label: 'Event' },
  ancient: { cls: 'ancient', label: 'Ancient' },
};

// Inline SVG room icons (inherit color via currentColor). Hand-drawn so they
// read crisply at node size and don't depend on an emoji font.
const ICONS = {
  combat: `<path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" d="M5.5 18.5 18 6M18.5 18.5 6 6"/><circle cx="5.5" cy="18.5" r="1.5" fill="currentColor"/><circle cx="18.5" cy="18.5" r="1.5" fill="currentColor"/>`,
  elite: `<path fill="currentColor" fill-rule="evenodd" d="M12 3c-4.1 0-7 3-7 6.8 0 2 .9 3.7 2.3 4.9.3.2.4.5.4.8v1.3c0 .9.7 1.6 1.6 1.6h.1v-1.7h1.3v1.7h1.6v-1.7h1.3v1.7h.1c.9 0 1.6-.7 1.6-1.6v-1.3c0-.3.1-.6.4-.8C18.1 13.5 19 11.8 19 9.8 19 6 16.1 3 12 3ZM9 12.7a1.8 1.8 0 1 0 0-3.6 1.8 1.8 0 0 0 0 3.6Zm6 0a1.8 1.8 0 1 0 0-3.6 1.8 1.8 0 0 0 0 3.6Z"/>`,
  boss: `<path fill="currentColor" d="M4.2 17 2.4 6.6l5.2 3.3L12 4l4.4 5.9 5.2-3.3L19.8 17H4.2Z"/><rect x="4.4" y="18.4" width="15.2" height="2.3" rx="0.7" fill="currentColor"/>`,
  rest: `<path fill="currentColor" d="M12.6 2.3c.4 2.6-.8 3.9-2 5.1-1.3 1.3-2.6 2.6-2.6 5.1A5.6 5.6 0 0 0 12 18.2a5.4 5.4 0 0 0 5.4-5.4c0-3.3-2.3-5.5-3.5-7 .2 1.4-.3 2.3-1.1 2.6-.9.3-1.6-.3-1.5-1.6.1-1.6.6-2.7 1.3-4.1Z"/>`,
  shop: `<g fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"><path d="M5.5 8h13l-1 11.5h-11L5.5 8Z"/><path d="M9 8.5V6.4a3 3 0 0 1 6 0v2.1"/></g>`,
  treasure: `<g fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"><path d="M4 11c0-3 3.6-5 8-5s8 2 8 5v8.2H4V11Z"/><path d="M4 12.6h16"/></g><rect x="10.3" y="11.1" width="3.4" height="3.6" rx="0.6" fill="currentColor"/>`,
  event: `<path fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" d="M8.7 9.1a3.4 3.4 0 1 1 6 2.3c-1 1-2.6 1.6-2.6 3.3"/><circle cx="12.1" cy="18.2" r="1.35" fill="currentColor"/>`,
  ancient: `<g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M5 9.6 8 5h8l3 4.6-7 9.9z"/><path d="M5 9.6h14M9 5 8 9.6l4 9.9M15 5l1 4.6-4 9.9"/></g>`,
};
const iconSvg = cls => `<svg class="ic" viewBox="0 0 24 24" aria-hidden="true">${ICONS[cls] || ICONS.event}</svg>`;

function cardListHtml(cards) {
  // group cards (by id/upgrade/ench) -> chips
  if (!cards || !cards.length) return '';
  const groups = new Map();
  for (const c of cards) {
    const k = c.id + (c.current_upgrade_level || 0);
    groups.set(k, (groups.get(k) || 0) + 1);
  }
  const parts = [];
  for (const c of cards) {
    parts.push(idSuffix(c.id).replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, x => x.toUpperCase()) + (c.current_upgrade_level > 0 ? '+' : ''));
  }
  return parts.join(', ');
}

function enrichRun(run, fileName) {
  const data = loadData(pickVersion(run.build_id));
  // primary player = first
  const player = run.players[0];
  const pid = player.id;

  // map id -> player_stats for this player at each map point
  const acts = [];
  let floorNum = 0;
  const hpSeries = [];
  let finalStats = null;

  run.map_point_history.forEach((actPoints, ai) => {
    const actId = run.acts[ai] || `ACT ${ai + 1}`;
    const actKey = pascal(idSuffix(actId));
    const actWiki = data.acts.get(actKey);
    const points = [];
    for (const mp of actPoints) {
      const ps = (mp.player_stats || []).find(s => s.player_id === pid) || (mp.player_stats || [])[0] || {};
      const room = (mp.rooms || [])[0] || {};
      floorNum += 1;
      const icon = MAP_ICON[mp.map_point_type] || MAP_ICON.unknown;

      let encounter = null;
      if (room.model_id && room.model_id.startsWith('ENCOUNTER')) {
        encounter = resolveEncounter(room.model_id, room.monster_ids, data);
      }
      let event = null;
      if (room.model_id && room.model_id.startsWith('EVENT')) {
        event = eventName(room.model_id, data);
      }

      // card rewards offered (all, with which was taken)
      const cardRewards = (ps.card_choices || []).map(c => ({
        ...refCard(c.card?.id, data), picked: !!c.was_picked,
      }));
      const pickedKeys = new Set((ps.card_choices || []).filter(c => c.was_picked).map(c => idSuffix(c.card?.id)));

      // detail bits (as {title,url} refs)
      const cardsGainedAll = (ps.cards_gained || []).map(c => c.id);
      // cards added outside a reward pick (events, Neow, generators...)
      const cardsGainedOther = cardsGainedAll.filter(id => !pickedKeys.has(idSuffix(id))).map(id => refCard(id, data));
      const cardsRemoved = (ps.cards_removed || []).map(c => refCard(typeof c === 'string' ? c : (c.id || c.card?.id), data));
      const cardsUpgraded = (ps.upgraded_cards || []).map(id => refCard(id, data));
      const cardsTransformed = (ps.cards_transformed || []).map(t => ({
        from: refCard(t.original_card?.id, data), to: refCard(t.final_card?.id, data),
      }));
      const relicsGained = uniqByTitle([]
        .concat((ps.relic_choices || []).filter(r => r.was_picked).map(r => refRelic(r.choice, data)))
        .concat((ps.bought_relics || []).map(id => refRelic(id, data)))
        .concat((ps.ancient_choice || []).filter(a => a.was_chosen).map(a => refRelic(a.TextKey, data))));
      const potionsGained = uniqByTitle([]
        .concat((ps.potion_choices || []).filter(p => p.was_picked).map(p => refPotion(p.choice, data)))
        .concat((ps.bought_potions || []).map(id => refPotion(id, data))));
      const eventChoices = (ps.event_choices || []).map(eventChoiceLabel).filter(Boolean);
      const restChoices = (ps.rest_site_choices || []).map(c => titleCase(c));

      const dmg = ps.damage_taken || 0;
      hpSeries.push({ floor: floorNum, hp: ps.current_hp, maxHp: ps.max_hp, act: ai });
      finalStats = ps;

      points.push({
        floor: floorNum,
        type: mp.map_point_type,
        icon,
        roomType: room.room_type,
        turns: room.turns_taken,
        encounter, event,
        hp: ps.current_hp, maxHp: ps.max_hp,
        damageTaken: dmg,
        hpHealed: ps.hp_healed || 0,
        maxHpGained: ps.max_hp_gained || 0,
        maxHpLost: ps.max_hp_lost || 0,
        gold: ps.current_gold,
        goldGained: ps.gold_gained || 0,
        goldSpent: ps.gold_spent || 0,
        cardRewards,
        cardsGainedOther, cardsRemoved, cardsUpgraded, cardsTransformed,
        relicsGained, potionsGained,
        eventChoices, restChoices,
      });
    }
    acts.push({ id: actId, title: actWiki ? actWiki.title : actKey, points });
  });

  // deck
  const deck = (player.deck || []).map(c => resolveCardEntry(c, data));
  // group deck for display
  const deckGroups = [];
  const gmap = new Map();
  for (const c of deck) {
    const k = c.key + '|' + (c.upgraded ? 'U' : '') + '|' + (c.ench ? c.ench.title + c.ench.amount : '');
    if (gmap.has(k)) { gmap.get(k).count++; }
    else { const g = { ...c, count: 1 }; gmap.set(k, g); deckGroups.push(g); }
  }
  deckGroups.sort((a, b) =>
    (TYPE_ORDER[a.type] ?? 9) - (TYPE_ORDER[b.type] ?? 9) ||
    (RARITY_ORDER[b.rarity] ?? -1) - (RARITY_ORDER[a.rarity] ?? -1) ||
    a.title.localeCompare(b.title));

  const relics = (player.relics || []).map(r => resolveRelic(r.id, data));
  const potions = (player.potions || []).map(p => resolvePotion(p.id, data));

  // counts by category
  const deckCounts = {};
  for (const c of deck) deckCounts[c.rarity || 'Unknown'] = (deckCounts[c.rarity || 'Unknown'] || 0) + 1;
  const relicCounts = {};
  for (const r of relics) relicCounts[r.rarity || 'Unknown'] = (relicCounts[r.rarity || 'Unknown'] || 0) + 1;

  const charKey = idSuffix(player.character);
  const charWiki = data.characters.get(charKey);

  return {
    file: fileName,
    id: path.basename(fileName, '.run'),
    version: data.version,
    buildId: run.build_id,
    win: run.win,
    abandoned: run.was_abandoned,
    ascension: run.ascension,
    seed: run.seed,
    mode: run.game_mode,
    runTime: run.run_time,
    startTime: run.start_time,
    killedByEncounter: idSuffix(run.killed_by_encounter),
    killedByEvent: idSuffix(run.killed_by_event),
    character: charWiki ? charWiki.title : charKey,
    characterKey: charKey,
    charImg: characterImage(charKey),
    floors: floorNum,
    finalHp: finalStats ? finalStats.current_hp : null,
    finalMaxHp: finalStats ? finalStats.max_hp : null,
    finalGold: finalStats ? finalStats.current_gold : null,
    playerCount: run.players.length,
    acts, deck: deckGroups, deckCount: deck.length, relics, potions,
    deckCounts, relicCounts,
    hpSeries,
  };
}

// ---------------------------------------------------------------------------
// HTML rendering
// ---------------------------------------------------------------------------
function fmtTime(sec) {
  if (sec == null) return '';
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return (h ? h + ':' : '') + String(m).padStart(h ? 2 : 1, '0') + ':' + String(s).padStart(2, '0');
}
function fmtDate(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) +
    ', ' + d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}
function titleCase(s) { return String(s || '').replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase()); }

// tooltip registry: returns a key and registers html
function buildTooltips(r) {
  const tips = {};
  const reg = (key, html) => { tips[key] = html; return key; };

  // cards
  r.deck.forEach((c, i) => {
    reg('card:' + i, cardTipHtml(c));
  });
  r.relics.forEach((rl, i) => reg('relic:' + i, relicTipHtml(rl)));
  r.potions.forEach((p, i) => reg('potion:' + i, potionTipHtml(p)));
  r.acts.forEach((act, ai) => act.points.forEach((p, pi) => reg(`floor:${ai}:${pi}`, floorTipHtml(p, r))));
  return tips;
}

function cardTipHtml(c) {
  const rar = c.rarity ? `<span class="rar rar-${c.rarity.toLowerCase()}">${esc(c.rarity)}</span>` : '';
  const cost = c.cost != null ? `<span class="cost">${esc(c.cost)}</span>` : '';
  const ench = c.ench ? `<div class="tip-ench">✦ ${esc(c.ench.title)}${c.ench.amount > 1 ? ' ×' + c.ench.amount : ''}</div>` : '';
  const kws = (c.keywords && c.keywords.length) ? `<div class="tip-kws">${c.keywords.map(k => `<span class="kwtag">${esc(k)}</span>`).join('')}</div>` : '';
  return `<div class="tip-card">
    <div class="tip-head">${cost}<span class="tip-title">${esc(c.title)}${c.upgraded ? '<span class="up">+</span>' : ''}</span></div>
    <div class="tip-sub">${c.type ? esc(c.type) : ''}${rar ? ' · ' + rar : ''}</div>
    ${c.img ? `<img class="tip-art" src="${c.img}" alt="">` : ''}
    <div class="tip-desc">${c.desc || ''}</div>
    ${ench}${kws}
    <div class="tip-foot">Added floor ${c.floor ?? '?'}${c.count > 1 ? ` · ×${c.count} in deck` : ''}${c.url ? ' · click → wiki ↗' : ''}</div>
  </div>`;
}
function relicTipHtml(r) {
  return `<div class="tip-card">
    <div class="tip-head"><span class="tip-title">${esc(r.title)}</span></div>
    ${r.rarity ? `<div class="tip-sub"><span class="rar rar-${r.rarity.toLowerCase()}">${esc(r.rarity)}</span></div>` : ''}
    ${r.img ? `<img class="tip-art tip-art-relic" src="${r.img}" alt="">` : ''}
    <div class="tip-desc">${r.desc || ''}</div>
    ${r.flavor ? `<div class="tip-flavor">${r.flavor}</div>` : ''}
  </div>`;
}
function potionTipHtml(p) {
  return `<div class="tip-card">
    <div class="tip-head"><span class="tip-title">${esc(p.title)}</span></div>
    ${p.rarity ? `<div class="tip-sub"><span class="rar rar-${p.rarity.toLowerCase()}">${esc(p.rarity)}</span></div>` : ''}
    ${p.img ? `<img class="tip-art tip-art-relic" src="${p.img}" alt="">` : ''}
    <div class="tip-desc">${p.desc || ''}</div>
  </div>`;
}
function floorHeading(p) {
  if (p.encounter && p.encounter.title) return p.encounter.title;
  if (p.event && p.event.title) return p.event.title;
  return p.icon.label;
}
const refNames = refs => refs.map(c => esc(c.title)).join(', ');

function floorTipHtml(p, r) {
  const rows = [];
  const line = (label, val) => rows.push(`<div class="fr"><span class="frl">${label}</span><span class="frv">${val}</span></div>`);

  if (p.encounter && p.encounter.monsters.length) line('Enemies', refNames(p.encounter.monsters));
  if (p.turns) line('Turns', p.turns);
  if (p.hp != null) line('HP', `${p.hp}/${p.maxHp}`);
  if (p.damageTaken) line('Damage taken', `<span class="bad">−${p.damageTaken}</span>`);
  if (p.hpHealed) line('Healed', `<span class="good">+${p.hpHealed}</span>`);
  if (p.maxHpGained) line('Max HP', `<span class="good">+${p.maxHpGained}</span>`);
  if (p.maxHpLost) line('Max HP', `<span class="bad">−${p.maxHpLost}</span>`);
  if (p.gold != null) line('Gold', p.gold + (p.goldGained ? ` <span class="good">(+${p.goldGained})</span>` : '') + (p.goldSpent ? ` <span class="bad">(−${p.goldSpent})</span>` : ''));
  if (p.restChoices.length) line('Rest', p.restChoices.map(esc).join(', '));
  if (p.eventChoices.length) line('Chose', p.eventChoices.map(esc).join('; '));
  const offered = p.cardRewards.filter(c => !c.picked);
  if (p.cardRewards.some(c => c.picked)) line('Took', refNames(p.cardRewards.filter(c => c.picked)));
  if (offered.length) line('Skipped', refNames(offered));
  if (p.cardsGainedOther.length) line('Cards +', refNames(p.cardsGainedOther));
  if (p.cardsRemoved.length) line('Cards −', refNames(p.cardsRemoved));
  if (p.cardsUpgraded.length) line('Upgraded', refNames(p.cardsUpgraded));
  if (p.cardsTransformed.length) line('Transformed', p.cardsTransformed.map(t => `${esc(t.from.title)} → ${esc(t.to.title)}`).join(', '));
  if (p.relicsGained.length) line('Relics +', refNames(p.relicsGained));
  if (p.potionsGained.length) line('Potions +', refNames(p.potionsGained));

  return `<div class="tip-card tip-floor">
    <div class="tip-head"><span class="floornum ${p.icon.cls}">${iconSvg(p.icon.cls)}</span><span class="tip-title">${esc(floorHeading(p))}</span></div>
    <div class="tip-sub">Floor ${p.floor} · ${esc(p.icon.label)} · click to jump down ↓</div>
    <div class="tip-rows">${rows.join('') || '<div class="fr"><span class="frv">—</span></div>'}</div>
  </div>`;
}

// ---------------------------------------------------------------------------
// OpenGraph / social preview
// ---------------------------------------------------------------------------
let ogEnabled = false; // set in main() once chromium availability is known

function ogResultText(r) {
  if (r.win) return 'Victory';
  if (r.abandoned) return 'Abandoned';
  const k = r.killedByEncounter && r.killedByEncounter !== 'NONE' ? titleCase(r.killedByEncounter)
    : (r.killedByEvent && r.killedByEvent !== 'NONE' ? titleCase(r.killedByEvent) : '');
  return k ? `Defeated by ${k}` : 'Defeated';
}
function ogDescription(r) {
  const bits = [ogResultText(r)];
  if (r.ascension) bits.push('Ascension ' + r.ascension);
  bits.push(`${r.floors} floors`);
  if (r.runTime) bits.push(fmtTime(r.runTime));
  bits.push(`${r.deckCount} cards`, `${r.relics.length} relics`);
  bits.push('seed ' + r.seed);
  return bits.join(' · ');
}
function ogMetaTags(r) {
  const title = `${r.character} — ${ogResultText(r)} · Slay the Spire II`;
  const desc = ogDescription(r);
  const url = `${SITE_BASE}/${r.id}.html`;
  // a rendered card if chromium was available, else the character portrait
  const img = ogEnabled ? `${SITE_BASE}/assets/og/${r.id}.png` : (r.charImg ? `${SITE_BASE}/${r.charImg}` : null);
  const tags = [
    `<meta property="og:type" content="article">`,
    `<meta property="og:site_name" content="StS II Run Archive">`,
    `<meta property="og:title" content="${esc(title)}">`,
    `<meta property="og:description" content="${esc(desc)}">`,
    `<meta property="og:url" content="${esc(url)}">`,
    `<meta name="description" content="${esc(desc)}">`,
    `<meta name="theme-color" content="#16110d">`,
  ];
  if (img) {
    tags.push(
      `<meta property="og:image" content="${esc(img)}">`,
      `<meta name="twitter:card" content="${ogEnabled ? 'summary_large_image' : 'summary'}">`,
      `<meta name="twitter:image" content="${esc(img)}">`);
    if (ogEnabled) tags.push(`<meta property="og:image:width" content="1200">`, `<meta property="og:image:height" content="630">`);
  } else {
    tags.push(`<meta name="twitter:card" content="summary">`);
  }
  tags.push(`<meta name="twitter:title" content="${esc(title)}">`, `<meta name="twitter:description" content="${esc(desc)}">`);
  return tags.join('\n');
}

// 1200x630 (rendered @2x from 600x315) social card, composed from run data.
function ogCardHtml(r) {
  const a = p => p ? p.replace(/^assets\//, '../') : null; // og/ -> assets/ sibling
  const accent = r.win ? 'linear-gradient(90deg,#f2c14e,#7fd07f)' : r.abandoned ? 'linear-gradient(90deg,#a8957a,#5a4a38)' : 'linear-gradient(90deg,#e0533f,#7a2018)';
  const badgeColor = r.win ? '#7fd07f' : r.abandoned ? '#4a3c2c' : '#e0533f';
  const badgeInk = r.win ? '#0d0d0d' : '#fff';
  const relicRow = r.relics.slice(0, 13).map(rl => rl.img
    ? `<img src="${a(rl.img)}">` : `<span class="ni">${esc((rl.title[0] || '?'))}</span>`).join('');
  const moreRelics = r.relics.length > 13 ? `<span class="more">+${r.relics.length - 13}</span>` : '';
  const chip = (label, val) => `<span class="chip"><b>${esc(val)}</b> ${esc(label)}</span>`;
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  *{margin:0;box-sizing:border-box;font-family:"Segoe UI",system-ui,sans-serif}
  html{zoom:2}
  body{width:600px;height:315px;overflow:hidden;background:radial-gradient(600px 320px at 30% -40px,#2a1f16,#16110d 70%);color:#f3e9d8}
  .bar{height:6px;background:${accent}}
  .pad{padding:20px 26px 16px;height:309px;display:flex;flex-direction:column}
  .top{display:flex;gap:18px;align-items:center}
  .portrait{width:96px;height:96px;border-radius:14px;object-fit:cover;background:#000;border:1px solid #3a2d20;flex:none}
  .badge{display:inline-block;font-size:13px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;
    color:${badgeInk};background:${badgeColor};border-radius:99px;padding:4px 12px}
  h1{font-size:35px;line-height:1.1;margin:6px 0 0}
  .sub{color:#a8957a;font-size:15px;margin-top:3px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
  .chip{background:#1f1813;border:1px solid #3a2d20;border-radius:8px;padding:5px 11px;font-size:15px}
  .chip b{color:#f2c14e;font-weight:700}
  .relics{display:flex;gap:6px;align-items:center;margin-top:auto;padding-top:12px}
  .relics img{width:36px;height:36px;border-radius:8px;background:#0c0907;border:1px solid #3a2d20;object-fit:contain}
  .relics .ni{width:36px;height:36px;border-radius:8px;background:#1f1813;border:1px solid #3a2d20;display:flex;align-items:center;justify-content:center;color:#a8957a;font-weight:700}
  .more{color:#a8957a;font-size:15px;margin-left:2px}
  .foot{display:flex;justify-content:space-between;align-items:flex-end;margin-top:11px;color:#a8957a;font-size:13px}
  .wordmark{color:#f2c14e;font-weight:700;letter-spacing:.02em}
  </style></head><body>
  <div class="bar"></div>
  <div class="pad">
    <div class="top">
      ${r.charImg ? `<img class="portrait" src="${a(r.charImg)}">` : ''}
      <div>
        <span class="badge">${esc(ogResultText(r))}</span>
        <h1>${esc(r.character)}</h1>
        <div class="sub">${r.ascension ? `Ascension ${r.ascension} · ` : ''}Seed ${esc(r.seed)} · ${esc(r.buildId)}</div>
      </div>
    </div>
    <div class="chips">
      ${r.finalHp != null ? chip('HP', `${r.finalHp}/${r.finalMaxHp}`) : ''}
      ${chip('floors', r.floors)}
      ${r.runTime ? chip('time', fmtTime(r.runTime)) : ''}
      ${chip('cards', r.deckCount)}
      ${chip('relics', r.relics.length)}
    </div>
    <div class="relics">${relicRow}${moreRelics}</div>
    <div class="foot"><span class="wordmark">Slay the Spire II — Run Archive</span><span>nelhage.com/f/sts2.runs</span></div>
  </div>
  </body></html>`;
}

// Rasterize OG cards with the chromium from the dev shell. Concurrency-limited,
// and skips runs whose image already exists unless `force`.
async function renderOgImages(runs, force) {
  const bin = process.env.CHROMIUM_BIN || 'chromium';
  const ogDir = path.join(ASSETS, 'og');
  fs.mkdirSync(ogDir, { recursive: true });
  const work = runs.filter(r => force || !fs.existsSync(path.join(ogDir, r.id + '.png')));
  if (!work.length) return;

  const shot = (r, idx) => new Promise((res) => {
    const html = path.join(ogDir, r.id + '.html');
    const png = path.join(ogDir, r.id + '.png');
    fs.writeFileSync(html, ogCardHtml(r));
    // The card body is 600x315 but zoomed 2x in CSS, so it lays out at 1200x630
    // and fills a 1200x630 window exactly (heights below ~600 get clamped, which
    // is what produced earlier letterboxing).
    const args = ['--headless', '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
      '--window-size=1200,630',
      `--user-data-dir=${path.join(ogDir, '.prof-' + (idx % OG_CONCURRENCY))}`,
      `--screenshot=${png}`, `file://${html}`];
    const p = spawn(bin, args, { stdio: 'ignore' });
    p.on('error', () => res(false));
    p.on('exit', () => { if (!process.env.OG_KEEP) { try { fs.unlinkSync(html); } catch {} } res(fs.existsSync(png)); });
  });

  let i = 0, done = 0;
  const worker = async () => { while (i < work.length) { const idx = i++; await shot(work[idx], idx); done++; } };
  await Promise.all(Array.from({ length: Math.min(OG_CONCURRENCY, work.length) }, worker));
  // clean up the throwaway chromium profiles
  for (let k = 0; k < OG_CONCURRENCY; k++) fs.rmSync(path.join(ogDir, '.prof-' + k), { recursive: true, force: true });
  console.log(`rendered ${done} OG image(s)`);
}
const OG_CONCURRENCY = 6;

function renderRun(r) {
  const tips = buildTooltips(r);

  // header chips
  const result = r.win ? '<span class="badge win">Victory</span>'
    : r.abandoned ? '<span class="badge aband">Abandoned</span>'
    : '<span class="badge loss">Defeated</span>';
  let killedBy = '';
  if (!r.win && !r.abandoned) {
    const k = r.killedByEncounter && r.killedByEncounter !== 'NONE' ? titleCase(r.killedByEncounter)
      : (r.killedByEvent && r.killedByEvent !== 'NONE' ? titleCase(r.killedByEvent) : '');
    if (k) killedBy = `<div class="killed">Killed by ${esc(k)}</div>`;
  }

  // map path rows
  const actRows = r.acts.map((act, ai) => {
    const cells = act.points.map((p, pi) => {
      return `<a class="node ${p.icon.cls}" href="#frow-${ai}-${pi}" data-tip="floor:${ai}:${pi}">
        <span class="node-glyph">${iconSvg(p.icon.cls)}</span>
        <span class="node-floor">${p.floor}</span>
      </a>`;
    }).join('<span class="node-link"></span>');
    return `<div class="act-row">
      <div class="act-name">${esc(act.title)}</div>
      <div class="act-path">${cells}</div>
    </div>`;
  }).join('');

  // hp chart
  const chart = hpChartSvg(r);

  // relics
  const iconTag = (cls, tip, url, img, title) => {
    const inner = img ? `<img src="${img}" alt="${esc(title)}">` : `<span class="noimg">${esc(title[0] || '?')}</span>`;
    return url
      ? `<a class="${cls}" data-tip="${tip}" href="${esc(url)}" target="_blank" rel="noopener">${inner}</a>`
      : `<div class="${cls}" data-tip="${tip}" tabindex="0">${inner}</div>`;
  };
  const relicHtml = r.relics.map((rl, i) => iconTag('relic', `relic:${i}`, rl.url, rl.img, rl.title)).join('');
  const potionHtml = r.potions.length
    ? r.potions.map((p, i) => iconTag('relic potion', `potion:${i}`, p.url, p.img, p.title)).join('') : '';

  // deck
  const deckHtml = r.deck.map((c, i) => {
    const cls = `dcard rar-${(c.rarity || 'common').toLowerCase()} ${c.upgraded ? 'upg' : ''}`;
    const inner = `${c.count > 1 ? `<span class="dcount">${c.count}×</span>` : ''}<span class="dname">${esc(c.title)}${c.upgraded ? '<span class="up">+</span>' : ''}${c.ench ? '<span class="ench">✦</span>' : ''}</span>`;
    return c.url
      ? `<a class="${cls}" data-tip="card:${i}" href="${esc(c.url)}" target="_blank" rel="noopener">${inner}</a>`
      : `<div class="${cls}" data-tip="card:${i}" tabindex="0">${inner}</div>`;
  }).join('');

  const summarize = (counts, order) => Object.entries(counts)
    .sort((a, b) => (order[a[0]] ?? 9) - (order[b[0]] ?? 9))
    .map(([k, v]) => `${v} ${k}`).join(', ');

  const metaChips = [
    r.finalHp != null ? `<span class="chip"><span class="ico">❤</span>${r.finalHp}/${r.finalMaxHp}</span>` : '',
    r.finalGold != null ? `<span class="chip"><span class="ico gold">⬤</span>${r.finalGold}</span>` : '',
    `<span class="chip"><span class="ico">▦</span>${r.floors} floors</span>`,
    r.runTime ? `<span class="chip"><span class="ico">◷</span>${fmtTime(r.runTime)}</span>` : '',
    r.ascension ? `<span class="chip">Ascension ${r.ascension}</span>` : '',
    `<span class="chip">${esc(titleCase(r.mode))}</span>`,
    r.playerCount > 1 ? `<span class="chip">${r.playerCount} players</span>` : '',
  ].filter(Boolean).join('');

  return `<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(r.character)} — ${r.win ? 'Victory' : (r.abandoned ? 'Abandoned' : 'Defeat')} · StS II run ${esc(r.id)}</title>
${ogMetaTags(r)}
<style>${CSS}</style>
</head><body>
<div class="wrap">
  <header class="hdr ${r.win ? 'is-win' : r.abandoned ? 'is-aband' : 'is-loss'}">
    <div class="hdr-char">
      ${r.charImg ? `<img class="char-icon" src="${r.charImg}" alt="${esc(r.character)}">` : ''}
    </div>
    <div class="hdr-main">
      <div class="hdr-top">${result}<h1>${esc(r.character)}</h1></div>
      <div class="chips">${metaChips}</div>
      ${killedBy}
    </div>
    <div class="hdr-meta">
      <div>${esc(fmtDate(r.startTime))}</div>
      <div class="seed">Seed: ${esc(r.seed)}</div>
      <div class="ver">${esc(r.buildId)}${r.version !== r.buildId.replace(/^v/, '') ? ` <span class="vnote">(data v${esc(r.version)})</span>` : ''}</div>
    </div>
  </header>

  <section class="panel">
    <h2>The Climb</h2>
    <div class="map">${actRows}</div>
  </section>

  <section class="panel">
    <h2>Vitals</h2>
    ${chart}
  </section>

  <section class="panel">
    <div class="panel-h2"><h2>Relics</h2><span class="count">${r.relics.length} · ${esc(summarize(r.relicCounts, RARITY_ORDER))}</span></div>
    <div class="relics">${relicHtml}</div>
    ${potionHtml ? `<h3 class="sub">Potions</h3><div class="relics">${potionHtml}</div>` : ''}
  </section>

  <section class="panel">
    <div class="panel-h2"><h2>Deck</h2><span class="count">${r.deckCount} cards · ${esc(summarize(r.deckCounts, RARITY_ORDER))}</span></div>
    <div class="deck">${deckHtml}</div>
  </section>

  <section class="panel">
    <div class="panel-h2"><h2>Floor by floor</h2><span class="count">click a node above to jump here · links open the sts2-wiki</span></div>
    ${floorTableHtml(r)}
  </section>

  <footer class="foot"><a href="index.html">← all runs</a> · Generated from Slay the Spire II run log · data from <a href="${WIKI_BASE}/" target="_blank" rel="noopener">sts2-wiki</a></footer>
</div>
<div id="tip" class="tooltip" role="tooltip"></div>
<script>
const TIPS = ${JSON.stringify(tips)};
${JS}
</script>
</body></html>`;
}

// linked reference -> <a> (or plain text if no wiki page)
function lref(ref) {
  if (!ref || !ref.title) return '';
  const t = esc(ref.title);
  return ref.url ? `<a class="wl" href="${esc(ref.url)}" target="_blank" rel="noopener">${t}</a>` : t;
}
const lrefs = refs => refs.map(lref).join(', ');

function floorDetailsHtml(p) {
  const out = [];
  const seg = (label, html) => out.push(`<div class="seg"><span class="seg-l">${label}</span> ${html}</div>`);
  if (p.cardRewards.length) {
    const parts = p.cardRewards.map(c =>
      `<span class="rw ${c.picked ? 'took' : 'skip'} rar-${(c.rarity || 'common').toLowerCase()}">${c.picked ? '✓ ' : ''}${lref(c)}</span>`).join('');
    seg('Card reward', `<span class="rwlist">${parts}</span>`);
  }
  if (p.cardsGainedOther.length) seg('Cards +', lrefs(p.cardsGainedOther));
  if (p.cardsRemoved.length) seg('Cards −', lrefs(p.cardsRemoved));
  if (p.cardsUpgraded.length) seg('Upgraded', lrefs(p.cardsUpgraded));
  if (p.cardsTransformed.length) seg('Transformed', p.cardsTransformed.map(t => `${lref(t.from)} → ${lref(t.to)}`).join(', '));
  if (p.relicsGained.length) seg('Relic +', lrefs(p.relicsGained));
  if (p.potionsGained.length) seg('Potion +', lrefs(p.potionsGained));
  if (p.restChoices.length) seg('Rest', esc(p.restChoices.join(', ')));
  if (p.eventChoices.length) seg('Chose', esc(p.eventChoices.join('; ')));
  return out.join('') || '<span class="muted">—</span>';
}

function floorTableHtml(r) {
  let rows = '';
  r.acts.forEach((act, ai) => {
    rows += `<tr class="act-head"><td colspan="6">${esc(act.title)}</td></tr>`;
    act.points.forEach((p, pi) => {
      const loc = p.encounter
        ? `${lref({ title: p.encounter.title, url: p.encounter.url })}${p.encounter.isWeak ? ' <span class="weak">weak</span>' : ''}`
        : p.event ? lref(p.event) : esc(p.icon.label);
      const enemies = p.encounter && p.encounter.monsters.length
        ? `<div class="enemies">${lrefs(p.encounter.monsters)}</div>` : '';
      let hp = p.hp != null ? `${p.hp}<span class="slash">/${p.maxHp}</span>` : '';
      const hpd = [];
      if (p.damageTaken) hpd.push(`<span class="bad">−${p.damageTaken}</span>`);
      if (p.hpHealed) hpd.push(`<span class="good">+${p.hpHealed}</span>`);
      if (p.maxHpGained) hpd.push(`<span class="good">+${p.maxHpGained} max</span>`);
      if (p.maxHpLost) hpd.push(`<span class="bad">−${p.maxHpLost} max</span>`);
      if (hpd.length) hp += ` <span class="delta">${hpd.join(' ')}</span>`;
      let gold = p.gold != null ? `${p.gold}` : '';
      const gd = [];
      if (p.goldGained) gd.push(`<span class="good">+${p.goldGained}</span>`);
      if (p.goldSpent) gd.push(`<span class="bad">−${p.goldSpent}</span>`);
      if (gd.length) gold += ` <span class="delta">${gd.join(' ')}</span>`;
      rows += `<tr id="frow-${ai}-${pi}" class="frow">
        <td class="c-floor">${p.floor}</td>
        <td class="c-type"><span class="floornum ${p.icon.cls}" title="${esc(p.icon.label)}">${iconSvg(p.icon.cls)}</span></td>
        <td class="c-loc">${loc}${enemies}</td>
        <td class="c-hp">${hp}</td>
        <td class="c-gold">${gold}</td>
        <td class="c-det">${floorDetailsHtml(p)}</td>
      </tr>`;
    });
  });
  return `<table class="ftable">
    <thead><tr><th>#</th><th></th><th>Room</th><th>HP</th><th>Gold</th><th>Rewards &amp; choices</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function hpChartSvg(r) {
  const pts = r.hpSeries.filter(p => p.hp != null);
  if (pts.length < 2) return '<div class="muted">No vitals data.</div>';
  const W = 1000, H = 180, padL = 36, padR = 12, padT = 12, padB = 22;
  const maxHp = Math.max(...pts.map(p => p.maxHp || p.hp), 1);
  const n = pts.length;
  const x = i => padL + (W - padL - padR) * (n === 1 ? 0.5 : i / (n - 1));
  const y = v => padT + (H - padT - padB) * (1 - v / maxHp);
  const hpLine = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.hp).toFixed(1)}`).join(' ');
  const maxLine = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.maxHp || p.hp).toFixed(1)}`).join(' ');
  const area = `M${x(0).toFixed(1)},${y(0).toFixed(1)} ` + pts.map((p, i) => `L${x(i).toFixed(1)},${y(p.hp).toFixed(1)}`).join(' ') + ` L${x(n - 1).toFixed(1)},${y(0).toFixed(1)} Z`;
  // act separators
  let seps = '';
  for (let i = 1; i < n; i++) if (pts[i].act !== pts[i - 1].act) {
    const xx = (x(i) + x(i - 1)) / 2;
    seps += `<line x1="${xx}" y1="${padT}" x2="${xx}" y2="${H - padB}" class="sep"/>`;
  }
  const dots = pts.map((p, i) =>
    `<circle cx="${x(i).toFixed(1)}" cy="${y(p.hp).toFixed(1)}" r="3" class="hpdot" data-floor="${p.floor}" data-hp="${p.hp}" data-max="${p.maxHp}"/>`
  ).join('');
  const yticks = [0, 0.5, 1].map(f => {
    const v = Math.round(maxHp * f);
    return `<text x="${padL - 6}" y="${(y(v) + 3).toFixed(1)}" class="ytick">${v}</text>`;
  }).join('');
  return `<svg class="hpchart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${seps}
    <path d="${area}" class="hparea"/>
    <path d="${maxLine}" class="maxline"/>
    <path d="${hpLine}" class="hpline"/>
    ${dots}
    ${yticks}
  </svg>
  <div class="chart-legend"><span class="lg hp">Current HP</span><span class="lg max">Max HP</span></div>`;
}

// ---------------------------------------------------------------------------
// index page
// ---------------------------------------------------------------------------
function renderIndex(runs) {
  const rows = runs.map(r => {
    const res = r.win ? '<span class="badge win">Win</span>' : r.abandoned ? '<span class="badge aband">Aband</span>' : '<span class="badge loss">Loss</span>';
    return `<a class="run-row ${r.win ? 'is-win' : ''}" href="${esc(r.id)}.html">
      <span class="rr-char">${r.charImg ? `<img src="${r.charImg}" alt="">` : ''}</span>
      <span class="rr-name">${esc(r.character)}</span>
      <span class="rr-res">${res}</span>
      <span class="rr-meta">${r.ascension ? 'A' + r.ascension + ' · ' : ''}${r.floors} floors · ${fmtTime(r.runTime)}</span>
      <span class="rr-date">${esc(fmtDate(r.startTime))}</span>
      <span class="rr-ver">${esc(r.buildId)}</span>
    </a>`;
  }).join('');
  return `<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slay the Spire II — Run Archive</title>
<meta property="og:type" content="website">
<meta property="og:title" content="Slay the Spire II — Run Archive">
<meta property="og:description" content="${runs.length} Slay the Spire II runs, with full deck, relic, and floor-by-floor detail.">
<meta property="og:url" content="${SITE_BASE}/index.html">
<meta name="description" content="${runs.length} Slay the Spire II runs, with full deck, relic, and floor-by-floor detail.">
<meta name="twitter:card" content="summary">
<meta name="theme-color" content="#16110d">
<style>${CSS}</style></head>
<body><div class="wrap">
  <header class="idx-hdr"><h1>Slay the Spire II — Run Archive</h1><p class="muted">${runs.length} runs</p></header>
  <div class="run-list">${rows}</div>
  <footer class="foot">Generated from Slay the Spire II run logs</footer>
</div></body></html>`;
}

// ---------------------------------------------------------------------------
const CSS = fs.readFileSync(path.join(ROOT, 'style.css'), 'utf8');
const JS = fs.readFileSync(path.join(ROOT, 'tip.js'), 'utf8');

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  fs.mkdirSync(ASSETS, { recursive: true });

  const args = process.argv.slice(2);
  const noOg = args.includes('--no-og');
  const ogForce = args.includes('--og-force');
  let files = args.filter(a => !a.startsWith('--'));
  const onlyThese = files.length > 0;
  if (!onlyThese) files = fs.readdirSync(RUNS_DIR).filter(f => f.endsWith('.run')).map(f => path.join(RUNS_DIR, f));

  // Decide up front whether we can render social cards (chromium present), so
  // the og:image meta points at the right place.
  if (!noOg) {
    const bin = process.env.CHROMIUM_BIN || 'chromium';
    try { execFileSync(bin, ['--version'], { stdio: 'ignore' }); ogEnabled = true; }
    catch { console.warn('chromium not found — og:image falls back to character portrait (pass --no-og to silence)'); }
  }

  const enriched = [];
  for (const f of files) {
    try {
      const run = JSON.parse(fs.readFileSync(f));
      const r = enrichRun(run, f);
      fs.writeFileSync(path.join(OUT, r.id + '.html'), renderRun(r));
      enriched.push(r);
      console.log(`built ${r.id}.html  (${r.character}, ${r.win ? 'win' : r.abandoned ? 'aband' : 'loss'}, ${r.floors} floors, data v${r.version})`);
    } catch (e) {
      console.error(`FAILED ${f}: ${e.stack}`);
    }
  }

  if (ogEnabled && enriched.length) {
    try { await renderOgImages(enriched, ogForce); }
    catch (e) { console.error('OG render failed: ' + e.message); }
  }

  // (re)build index over ALL runs so the listing stays complete
  const allFiles = fs.readdirSync(RUNS_DIR).filter(f => f.endsWith('.run')).map(f => path.join(RUNS_DIR, f));
  const indexRuns = [];
  for (const f of allFiles) {
    try {
      const run = JSON.parse(fs.readFileSync(f));
      const data = loadData(pickVersion(run.build_id));
      const player = run.players[0];
      const charKey = idSuffix(player.character);
      const charWiki = data.characters.get(charKey);
      let floors = 0; for (const a of run.map_point_history) floors += a.length;
      indexRuns.push({
        id: path.basename(f, '.run'),
        character: charWiki ? charWiki.title : charKey,
        charImg: characterImage(charKey),
        win: run.win, abandoned: run.was_abandoned, ascension: run.ascension,
        floors, runTime: run.run_time, startTime: run.start_time, buildId: run.build_id,
      });
    } catch (e) { console.error(`index skip ${f}: ${e.message}`); }
  }
  indexRuns.sort((a, b) => (b.startTime || 0) - (a.startTime || 0));
  fs.writeFileSync(path.join(OUT, 'index.html'), renderIndex(indexRuns));
  console.log(`built index.html (${indexRuns.length} runs)`);
}

main().catch(e => { console.error(e); process.exit(1); });
