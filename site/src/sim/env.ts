/**
 * TypeScript port of rl/pragyan/env.py. outcomes() must match the Python
 * transition model exactly; env.test.ts replays every recorded Python step.
 */
import raw from "../data/pragyan.json";
import type { MapData, PragyanData } from "./types";

export const DATA = raw as unknown as PragyanData;
const K = DATA.consts;

export const REGOLITH = 0, SLOPE = 1, BOULDER = 2, CRATER = 3, SHADOW = 4, VIKRAM = 5;
export const DIRS: [number, number][] = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]];
export const LIBS = 8, APXS = 9, TRANSMIT = 10, RECHARGE = 11, SLEEP = 12, N_ACTIONS = 13;
export const UNSAMPLED = 0, HELD = 1, SENT = 2;
export const ACTIONS = DATA.actions;
export const MAX_BATTERY = K.MAX_BATTERY;
export const MAX_STEPS = K.MAX_STEPS;
export const VIEW = K.VIEW;
export const COMM_RADIUS = K.COMM_RADIUS;
export const TERMINAL_EVENTS = ["crater", "battery_dead", "night", "sleep_safe", "sleep_unsafe"];

export interface LunarMap extends MapData {
  H: number;
  W: number;
  siteAt: Map<number, number>;
}

/** state = [row, col, battery, site statuses, sun phase] */
export type State = [number, number, number, number[], number];
/** [prob, next state (null = keep last), reward, terminal, event, crater direction] */
export type Outcome = [number, State | null, number, boolean, string, number?];

export function makeMap(md: MapData): LunarMap {
  const terrain = md.terrain.map((r) => r.slice());
  const sites = md.sites.map((s) => s.slice() as [number, number, number]);
  return {
    terrain, sites, vikram: [md.vikram[0], md.vikram[1]], start: [md.start[0], md.start[1]],
    H: terrain.length, W: terrain[0].length,
    siteAt: new Map(sites.map((s, i) => [s[0] * 100 + s[1], i])),
  };
}

export const passable = (m: LunarMap, r: number, c: number) =>
  r >= 0 && r < m.H && c >= 0 && c < m.W && m.terrain[r][c] !== BOULDER && m.terrain[r][c] !== VIKRAM;
export const inRange = (m: LunarMap, r: number, c: number) =>
  Math.max(Math.abs(r - m.vikram[0]), Math.abs(c - m.vikram[1])) <= K.COMM_RADIUS;
export const startState = (m: LunarMap): State => [m.start[0], m.start[1], K.MAX_BATTERY, m.sites.map(() => 0), 0];

export function outcomes(m: LunarMap, st: State, a: number): Outcome[] {
  const [r, c, bat, sites, phase] = st;
  const base: Outcome[] = [];
  if (a < 8) {
    const branches: [number, number][] = [[a, 1 - K.SLIP_P], [(a + 7) % 8, K.SLIP_P / 2], [(a + 1) % 8, K.SLIP_P / 2]];
    for (const [d, p] of branches) {
      const nr = r + DIRS[d][0], nc = c + DIRS[d][1];
      const ev = d !== a ? "slip" : "move";
      if (!passable(m, nr, nc)) { base.push([p, [r, c, bat, sites, phase], K.R_STEP + K.R_BUMP, false, "blocked"]); continue; }
      if (m.terrain[nr][nc] === CRATER) { base.push([p, null, K.R_STEP + K.R_CRATER, true, "crater", d]); continue; }
      const slope = m.terrain[nr][nc] === SLOPE;
      const rew = K.R_STEP + (slope ? K.R_SLOPE : 0), q = slope ? K.DRAIN_P_SLOPE : K.DRAIN_P;
      base.push([p * (1 - q), [nr, nc, bat, sites, phase], rew, false, ev]);
      if (bat - 1 <= 0) base.push([p * q, null, rew + K.R_DEAD, true, "battery_dead"]);
      else base.push([p * q, [nr, nc, bat - 1, sites, phase], rew, false, ev]);
    }
  } else if (a === LIBS || a === APXS) {
    const want = a === LIBS ? 0 : 1, i = m.siteAt.get(r * 100 + c);
    const ok = i !== undefined && m.sites[i][2] === want && sites[i] === UNSAMPLED;
    if (bat - 1 <= 0) base.push([1, null, K.R_STEP + K.R_DEAD, true, "battery_dead"]);
    else if (ok) {
      const ns = sites.slice(); ns[i!] = HELD;
      base.push([1, [r, c, bat - 1, ns, phase], K.R_STEP + K.R_SAMPLE, false, a === LIBS ? "libs_ok" : "apxs_ok"]);
    } else base.push([1, [r, c, bat - 1, sites, phase], K.R_STEP + K.R_WRONG, false, "wrong_instrument"]);
  } else if (a === TRANSMIT) {
    const held = sites.filter((s) => s === HELD).length;
    if (bat - 1 <= 0) base.push([1, null, K.R_STEP + K.R_DEAD, true, "battery_dead"]);
    else if (held && inRange(m, r, c))
      base.push([1, [r, c, bat - 1, sites.map((s) => (s === HELD ? SENT : s)), phase], K.R_STEP + K.R_SENT * held, false, "tx_ok"]);
    else base.push([1, [r, c, bat - 1, sites, phase], K.R_STEP + K.R_WRONG, false, "tx_fail"]);
  } else if (a === RECHARGE) {
    if (m.terrain[r][c] === SHADOW) base.push([1, st, K.R_STEP, false, "recharge_shadow"]);
    else base.push([1, [r, c, Math.min(K.MAX_BATTERY, bat + (phase === 1 ? 2 : 1)), sites, phase], K.R_STEP, false, "recharge"]);
  } else {
    const safe = phase === 2 && m.terrain[r][c] !== SHADOW && bat >= 3;
    const bonus = safe && sites.every((s) => s === SENT) ? K.R_MISSION : 0;
    base.push([1, null, K.R_STEP + bonus + (safe ? K.R_SLEEP_SAFE : K.R_SLEEP_UNSAFE), true, safe ? "sleep_safe" : "sleep_unsafe"]);
  }
  const out: Outcome[] = [];
  for (const [p, ns, rew, term, ev, d] of base) {
    if (term) { out.push([p, ns, rew, true, ev, d]); continue; }
    const s = ns!;
    out.push([p * (1 - K.PHASE_P), s, rew, false, ev]);
    if (s[4] === 2) out.push([p * K.PHASE_P, s, rew + K.R_NIGHT, true, "night"]);
    else out.push([p * K.PHASE_P, [s[0], s[1], s[2], s[3], s[4] + 1], rew, false, ev]);
  }
  return out;
}

export function validActions(m: LunarMap, st: State): boolean[] {
  const [r, c, bat, sites, phase] = st;
  const mask = new Array(N_ACTIONS).fill(false);
  DIRS.forEach(([dr, dc], d) => { mask[d] = passable(m, r + dr, c + dc); });
  const i = m.siteAt.get(r * 100 + c);
  if (i !== undefined && sites[i] === UNSAMPLED) mask[m.sites[i][2] === 0 ? LIBS : APXS] = true;
  mask[TRANSMIT] = sites.includes(HELD) && inRange(m, r, c);
  mask[RECHARGE] = m.terrain[r][c] !== SHADOW && bat < K.MAX_BATTERY;
  mask[SLEEP] = phase === 2;
  if (!mask.some(Boolean)) mask[SLEEP] = true;
  return mask;
}

export function features(m: LunarMap, st: State): Float32Array {
  const [r, c, bat, sites, phase] = st, V = K.VIEW, f = new Float32Array(K.FEATURE_DIM);
  let k = 0;
  for (let dr = -V; dr <= V; dr++) for (let dc = -V; dc <= V; dc++) {
    const rr = r + dr, cc = c + dc;
    if (rr < 0 || rr >= m.H || cc < 0 || cc >= m.W) f[k] = 1;
    else {
      const t = m.terrain[rr][cc];
      if (t === BOULDER || t === VIKRAM) f[k] = 1; else if (t === CRATER) f[k + 1] = 1;
      else if (t === SLOPE) f[k + 2] = 1; else if (t === SHADOW) f[k + 3] = 1;
    }
    k += 4;
  }
  f[k++] = bat / K.MAX_BATTERY;
  f[k + phase] = 1; k += 3;
  m.sites.forEach(([sr, sc], i) => { f[k + sites[i]] = 1; f[k + 3] = (sr - r) / m.H; f[k + 4] = (sc - c) / m.W; k += 5; });
  f[k] = (m.vikram[0] - r) / m.H; f[k + 1] = (m.vikram[1] - c) / m.W; f[k + 2] = inRange(m, r, c) ? 1 : 0; k += 3;
  const i = m.siteAt.get(r * 100 + c);
  if (i !== undefined && sites[i] === UNSAMPLED) f[k + m.sites[i][2]] = 1;
  k += 2;
  f[k] = m.terrain[r][c] === SHADOW ? 1 : 0;
  return f;
}

export function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sample(outs: Outcome[], u: number): Outcome {
  for (const o of outs) { u -= o[0]; if (u < 0) return o; }
  return outs[outs.length - 1];
}

export const sameState = (a: State, b: State) =>
  a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[4] === b[4] && a[3].every((x, i) => x === b[3][i]);
