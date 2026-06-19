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
  const data = {
    version,
    cards: indexByLocKey(readJson('cards.json')),
    relics: indexByLocKey(readJson('relics.json')),
    potions: indexByLocKey(readJson('potions.json')),
    monsters: indexByLocKey(readJson('monsters.json')),
    encounters: indexByLocKey(readJson('encounters.json')),
    ancients: indexByLocKey(readJson('ancients.json')),
    characters: indexByLocKey(readJson('characters.json')),
    acts: indexByClass(readJson('acts.json')),
    events,
    enchants,
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
    cost: wiki ? (wiki.x_cost ? 'X' : (wiki.energy_cost >= 0 ? wiki.energy_cost : null)) : null,
    type: wiki ? wiki.type : null,
    rarity: wiki ? wiki.rarity : null,
    character: wiki ? wiki.character : null,
    keywords: wiki ? (wiki.keywords || []) : [],
    desc: wiki ? markup(upgraded ? (wiki.upgraded_description_html || wiki.upgraded_description_plain || wiki.upgraded_description) : (wiki.description_html || wiki.description_plain || wiki.description)) : '',
  };
}

function resolveRelic(id, data) {
  const key = idSuffix(id);
  const w = data.relics.get(key);
  return {
    key,
    title: w ? w.title : key.replace(/_/g, ' '),
    rarity: w ? w.rarity : null,
    img: relicImage(w),
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
    desc: w ? markup(w.description, w.vars) : '',
  };
}

function resolveEncounter(modelId, monsterIds, data) {
  const key = idSuffix(modelId);
  const w = data.encounters.get(key);
  const monsters = (monsterIds || []).map(m => {
    const mw = data.monsters.get(idSuffix(m));
    return mw ? mw.title : idSuffix(m).replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
  });
  return {
    title: w ? w.title : (key ? key.replace(/_/g, ' ') : null),
    isWeak: w ? w.is_weak : false,
    monsters: [...new Set(monsters)],
  };
}

function eventName(modelId, data) {
  const cls = pascal(idSuffix(modelId));
  const w = data.events.get(cls);
  return w ? w.title : idSuffix(modelId).replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
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
  monster: { glyph: '⚔', cls: 'combat', label: 'Combat' },
  elite: { glyph: '☠', cls: 'elite', label: 'Elite' },
  boss: { glyph: '♛', cls: 'boss', label: 'Boss' },
  rest_site: { glyph: '🔥', cls: 'rest', label: 'Rest Site' },
  shop: { glyph: '🛒', cls: 'shop', label: 'Shop' },
  treasure: { glyph: '🎁', cls: 'treasure', label: 'Treasure' },
  unknown: { glyph: '?', cls: 'event', label: 'Unknown' },
  event: { glyph: '?', cls: 'event', label: 'Event' },
  ancient: { glyph: '𖣘', cls: 'ancient', label: 'Ancient' },
};

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
      let eventTitle = null;
      if (room.model_id && room.model_id.startsWith('EVENT')) {
        eventTitle = eventName(room.model_id, data);
      }

      // detail bits
      const cardsGained = (ps.cards_gained || []).map(c => idSuffix(c.id));
      const cardsRemoved = (ps.cards_removed || []).map(c => idSuffix(typeof c === 'string' ? c : c.id || c.card?.id));
      const cardsUpgraded = (ps.upgraded_cards || []).map(idSuffix);
      const cardsTransformed = (ps.cards_transformed || []).map(t => ({
        from: idSuffix(t.original_card?.id), to: idSuffix(t.final_card?.id),
      }));
      const relicsGained = [...new Set([]
        .concat((ps.relic_choices || []).filter(r => r.was_picked).map(r => idSuffix(r.choice)))
        .concat((ps.bought_relics || []).map(idSuffix))
        .concat((ps.ancient_choice || []).filter(a => a.was_chosen).map(a => a.TextKey)))];
      const potionsGained = [...new Set([]
        .concat((ps.potion_choices || []).filter(p => p.was_picked).map(p => idSuffix(p.choice)))
        .concat((ps.bought_potions || []).map(idSuffix)))];
      const eventChoices = (ps.event_choices || []).map(eventChoiceLabel).filter(Boolean);
      const restChoices = (ps.rest_site_choices || []).map(c => String(c).replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, x => x.toUpperCase()));

      const dmg = ps.damage_taken || 0;
      hpSeries.push({ floor: floorNum, hp: ps.current_hp, maxHp: ps.max_hp, act: ai });
      finalStats = ps;

      points.push({
        floor: floorNum,
        type: mp.map_point_type,
        icon,
        roomType: room.room_type,
        turns: room.turns_taken,
        encounter, eventTitle,
        hp: ps.current_hp, maxHp: ps.max_hp,
        damageTaken: dmg,
        hpHealed: ps.hp_healed || 0,
        maxHpGained: ps.max_hp_gained || 0,
        maxHpLost: ps.max_hp_lost || 0,
        gold: ps.current_gold,
        goldGained: ps.gold_gained || 0,
        goldSpent: ps.gold_spent || 0,
        cardsGained, cardsRemoved, cardsUpgraded, cardsTransformed,
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
    <div class="tip-foot">Added floor ${c.floor ?? '?'}${c.count > 1 ? ` · ×${c.count} in deck` : ''}</div>
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
function floorTipHtml(p, r) {
  const rows = [];
  let heading = p.icon.label;
  if (p.encounter && p.encounter.title) heading = (p.encounter.isWeak ? '' : '') + p.encounter.title;
  else if (p.eventTitle) heading = p.eventTitle;

  const line = (label, val) => rows.push(`<div class="fr"><span class="frl">${label}</span><span class="frv">${val}</span></div>`);

  if (p.encounter && p.encounter.monsters.length) line('Enemies', p.encounter.monsters.map(esc).join(', '));
  if (p.turns) line('Turns', p.turns);
  if (p.hp != null) line('HP', `${p.hp}/${p.maxHp}`);
  if (p.damageTaken) line('Damage taken', `<span class="bad">−${p.damageTaken}</span>`);
  if (p.hpHealed) line('Healed', `<span class="good">+${p.hpHealed}</span>`);
  if (p.maxHpGained) line('Max HP', `<span class="good">+${p.maxHpGained}</span>`);
  if (p.maxHpLost) line('Max HP', `<span class="bad">−${p.maxHpLost}</span>`);
  if (p.gold != null) line('Gold', p.gold + (p.goldGained ? ` <span class="good">(+${p.goldGained})</span>` : '') + (p.goldSpent ? ` <span class="bad">(−${p.goldSpent})</span>` : ''));
  if (p.restChoices.length) line('Rest', p.restChoices.map(esc).join(', '));
  if (p.eventChoices.length) line('Chose', p.eventChoices.map(esc).join('; '));
  if (p.cardsGained.length) line('Cards +', p.cardsGained.map(c => esc(titleCase(c))).join(', '));
  if (p.cardsRemoved.length) line('Cards −', p.cardsRemoved.map(c => esc(titleCase(c))).join(', '));
  if (p.cardsUpgraded.length) line('Upgraded', p.cardsUpgraded.map(c => esc(titleCase(c))).join(', '));
  if (p.cardsTransformed.length) line('Transformed', p.cardsTransformed.map(t => `${esc(titleCase(t.from))} → ${esc(titleCase(t.to))}`).join(', '));
  if (p.relicsGained.length) line('Relics +', p.relicsGained.map(c => esc(titleCase(c))).join(', '));
  if (p.potionsGained.length) line('Potions +', p.potionsGained.map(c => esc(titleCase(c))).join(', '));

  return `<div class="tip-card tip-floor">
    <div class="tip-head"><span class="floornum ${p.icon.cls}">${p.icon.glyph}</span><span class="tip-title">${esc(heading)}</span></div>
    <div class="tip-sub">Floor ${p.floor} · ${esc(p.icon.label)}</div>
    <div class="tip-rows">${rows.join('') || '<div class="fr"><span class="frv">—</span></div>'}</div>
  </div>`;
}

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
      return `<div class="node ${p.icon.cls}" data-tip="floor:${ai}:${pi}" tabindex="0">
        <span class="node-glyph">${p.icon.glyph}</span>
        <span class="node-floor">${p.floor}</span>
      </div>`;
    }).join('<span class="node-link"></span>');
    return `<div class="act-row">
      <div class="act-name">${esc(act.title)}</div>
      <div class="act-path">${cells}</div>
    </div>`;
  }).join('');

  // hp chart
  const chart = hpChartSvg(r);

  // relics
  const relicHtml = r.relics.map((rl, i) =>
    `<div class="relic" data-tip="relic:${i}" tabindex="0">${rl.img ? `<img src="${rl.img}" alt="${esc(rl.title)}">` : `<span class="noimg">${esc(rl.title[0])}</span>`}</div>`
  ).join('');

  const potionHtml = r.potions.length ? r.potions.map((p, i) =>
    `<div class="relic potion" data-tip="potion:${i}" tabindex="0">${p.img ? `<img src="${p.img}" alt="${esc(p.title)}">` : `<span class="noimg">${esc(p.title[0])}</span>`}</div>`
  ).join('') : '';

  // deck
  const deckHtml = r.deck.map((c, i) => {
    return `<div class="dcard rar-${(c.rarity || 'common').toLowerCase()} ${c.upgraded ? 'upg' : ''}" data-tip="card:${i}" tabindex="0">
      ${c.count > 1 ? `<span class="dcount">${c.count}×</span>` : ''}
      <span class="dthumb">${c.img ? `<img src="${c.img}" alt="" loading="lazy">` : ''}</span>
      <span class="dname">${esc(c.title)}${c.upgraded ? '<span class="up">+</span>' : ''}${c.ench ? '<span class="ench">✦</span>' : ''}</span>
    </div>`;
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

  <footer class="foot"><a href="index.html">← all runs</a> · Generated from Slay the Spire II run log</footer>
</div>
<div id="tip" class="tooltip" role="tooltip"></div>
<script>
const TIPS = ${JSON.stringify(tips)};
${JS}
</script>
</body></html>`;
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
<title>Slay the Spire II — Run Archive</title><style>${CSS}</style></head>
<body><div class="wrap">
  <header class="idx-hdr"><h1>Slay the Spire II — Run Archive</h1><p class="muted">${runs.length} runs</p></header>
  <div class="run-list">${rows}</div>
  <footer class="foot">Generated from Slay the Spire II run logs</footer>
</div></body></html>`;
}

// ---------------------------------------------------------------------------
const CSS = fs.readFileSync(path.join(ROOT, 'style.css'), 'utf8');
const JS = fs.readFileSync(path.join(ROOT, 'tip.js'), 'utf8');

function main() {
  fs.mkdirSync(OUT, { recursive: true });
  fs.mkdirSync(ASSETS, { recursive: true });

  let files = process.argv.slice(2);
  const onlyThese = files.length > 0;
  if (!onlyThese) files = fs.readdirSync(RUNS_DIR).filter(f => f.endsWith('.run')).map(f => path.join(RUNS_DIR, f));

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

main();
