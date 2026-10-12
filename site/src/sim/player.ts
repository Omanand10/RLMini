/**
 * Mission playback engine shared by the hero, replay, compare and live views.
 * Plays recorded Python episodes, or drives live with the in-browser DQN.
 */
import { DIRS, MAX_STEPS, mulberry32, outcomes, sample, startState, type LunarMap, type State } from "./env";
import { dqnAct } from "./dqn";
import type { Step } from "./episodes";

export interface LiveBrain { q: Float32Array; mask: boolean[]; a: number }

export class Player {
  map: LunarMap;
  steps: Step[] = [];
  live = false;
  rng = mulberry32(21);
  seed = 21;
  i = 0;
  t = 0;               // progress through the current step, 0..1
  cur: Step | null = null;
  state: State;
  score = 0;
  track: [number, number][] = [];
  done = false;
  endEv: string | null = null;
  playing = false;
  speed = 1;           // steps per second
  brain: LiveBrain | null = null;
  version = 0;         // bumps on every discrete change, for React subscribers
  private listeners = new Set<() => void>();

  constructor(map: LunarMap) {
    this.map = map;
    this.state = startState(map);
    this.reset();
  }

  subscribe = (fn: () => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; };
  getVersion = () => this.version;
  private emit() { this.version++; this.listeners.forEach((f) => f()); }

  loadSteps(map: LunarMap, steps: Step[]) {
    this.map = map; this.steps = steps; this.live = false; this.reset();
  }

  loadLive(map: LunarMap, seed = 21) {
    this.map = map; this.steps = []; this.live = true; this.seed = seed; this.reset();
  }

  reset() {
    this.rng = mulberry32(this.seed);
    this.state = startState(this.map);
    this.i = 0; this.t = 0; this.cur = null; this.score = 0; this.done = false; this.endEv = null;
    this.track = [[this.map.start[0], this.map.start[1]]];
    this.brain = this.live ? dqnAct(this.map, this.state) : null;
    this.emit();
  }

  /** jump to the end of step n (used by the scrubber) */
  seek(n: number) {
    if (this.live) return;
    const playing = this.playing;
    this.reset();
    for (let k = 0; k < Math.min(n, this.steps.length); k++) this.commit(this.steps[k]);
    this.playing = playing && !this.done;
    this.emit();
  }

  get length() { return this.live ? MAX_STEPS : this.steps.length; }

  private nextStep(): Step | null {
    if (!this.live) return this.steps[this.i] ?? null;
    const st = this.state, brain = dqnAct(this.map, st);
    this.brain = brain;
    const o = sample(outcomes(this.map, st, brain.a), this.rng());
    const next = o[1] ?? st;
    let dir: number | undefined = o[5];
    if (brain.a < 8 && dir === undefined) {
      const dr = next[0] - st[0], dc = next[1] - st[1];
      const d = DIRS.findIndex(([x, y]) => x === dr && y === dc);
      dir = d >= 0 ? d : brain.a;
    }
    return { a: brain.a, prev: st, next, rew: o[2], ev: o[4], term: o[3], dir };
  }

  private commit(st: Step) {
    this.state = st.next; this.score += st.rew; this.i++;
    const last = this.track[this.track.length - 1];
    if (st.next[0] !== last[0] || st.next[1] !== last[1]) this.track.push([st.next[0], st.next[1]]);
    if (st.term || (!this.live && this.i >= this.steps.length) || this.i >= MAX_STEPS) {
      this.done = true; this.playing = false; this.endEv = st.ev;
    }
  }

  /** advance by dt seconds; returns true if something visible changed */
  tick(dt: number) {
    if (!this.playing || this.done) return false;
    if (!this.cur) {
      this.cur = this.nextStep();
      if (!this.cur) { this.done = true; this.playing = false; this.emit(); return true; }
      this.t = 0;
      this.emit();
    }
    this.t += dt * this.speed;
    if (this.t >= 1) {
      const st = this.cur; this.cur = null;
      this.commit(st);
      if (this.live && !this.done) this.brain = dqnAct(this.map, this.state);
      this.emit();
    }
    return true;
  }

  play() { if (this.done) this.reset(); this.playing = true; this.emit(); }
  pause() { this.playing = false; this.emit(); }
  /** the map changed under a live rover (crater dropped): rethink from here */
  replan() { if (this.live && !this.cur) this.brain = dqnAct(this.map, this.state); this.emit(); }

  /** fractional grid position + heading for the renderer */
  pose(): { r: number; c: number; heading: number; fall: number } {
    const s = this.cur;
    if (!s) return { r: this.state[0], c: this.state[1], heading: this.heading ?? 0, fall: this.done && this.endEv === "crater" ? 1 : 0 };
    const [r0, c0] = s.prev;
    let [r1, c1] = s.next;
    const ease = this.t < 0.5 ? 2 * this.t * this.t : 1 - (-2 * this.t + 2) ** 2 / 2;
    if (s.dir !== undefined && s.a < 8) this.heading = Math.atan2(-DIRS[s.dir][0], DIRS[s.dir][1]);
    if (s.ev === "crater" && s.dir !== undefined) { r1 = r0 + DIRS[s.dir][0] * 0.9; c1 = c0 + DIRS[s.dir][1] * 0.9; }
    if (s.ev === "blocked") {
      const k = Math.sin(this.t * Math.PI) * 0.22, d = DIRS[s.a];
      return { r: r0 + d[0] * k, c: c0 + d[1] * k, heading: this.heading ?? 0, fall: 0 };
    }
    return { r: r0 + (r1 - r0) * ease, c: c0 + (c1 - c0) * ease, heading: this.heading ?? 0,
      fall: s.ev === "crater" ? Math.max(0, this.t - 0.5) * 2 : 0 };
  }
  heading?: number;
}
