/** Recorded Python episodes -> step list the 3D player animates. */
import { CRATER, DATA, DIRS, TERMINAL_EVENTS, makeMap, startState, type LunarMap, type State } from "./env";
import type { Episode } from "./types";

export interface Step {
  a: number;
  prev: State;
  next: State;
  rew: number;
  ev: string;
  term: boolean;
  /** direction actually driven (differs from a on a slip) */
  dir?: number;
}

export function buildSteps(E: Episode, m: LunarMap): Step[] {
  let st = startState(m);
  const out: Step[] = [];
  for (const f of E.frames) {
    const next: State = [f[0], f[1], f[2], f[3], f[4]];
    const a = f[5];
    let dir: number | undefined;
    if (a < 8) {
      if (f[7] === "crater") {
        for (const d of [a, (a + 7) % 8, (a + 1) % 8]) {
          const r = st[0] + DIRS[d][0], c = st[1] + DIRS[d][1];
          if (r >= 0 && r < m.H && c >= 0 && c < m.W && m.terrain[r][c] === CRATER) { dir = d; break; }
        }
      } else {
        const dr = next[0] - st[0], dc = next[1] - st[1];
        const d = DIRS.findIndex(([x, y]) => x === dr && y === dc);
        dir = d >= 0 ? d : a;
      }
    }
    out.push({ a, prev: st, next, rew: f[6], ev: f[7], term: TERMINAL_EVENTS.includes(f[7]), dir });
    st = next;
  }
  return out;
}

export const fixedMap = () => makeMap(DATA.map);
export const mapFor = (E: Episode) => (E.map ? makeMap(E.map) : fixedMap());

export const AGENTS = ["Q-learning", "SARSA", "Dyna-Q (k=10)", "n-step SARSA (n=4)", "Monte Carlo", "DQN",
  "Value Iteration", "Heuristic", "Random"] as const;

export const AGENT_INFO: Record<string, { short: string; module: string; idea: string }> = {
  "Value Iteration": { short: "Value Iteration", module: "Module 2 · Dynamic programming",
    idea: "Knows every probability of the Moon in advance and computes the perfect plan. Our upper bound, not a learner." },
  "Monte Carlo": { short: "Monte Carlo", module: "Module 2 · Monte Carlo control",
    idea: "Waits until the mission ends, then credits every decision with the total score that followed." },
  "Q-learning": { short: "Q-learning", module: "Module 3 · Off-policy TD",
    idea: "After every step, nudges its estimate towards the reward plus the value of the best next move." },
  SARSA: { short: "SARSA", module: "Module 3 · On-policy TD",
    idea: "Like Q-learning, but learns from the move it will actually make, exploration included. Plays it safer." },
  "n-step SARSA (n=4)": { short: "n-step SARSA", module: "Module 4 · n-step bootstrapping",
    idea: "Looks four steps ahead before updating, trading bias for variance." },
  "Dyna-Q (k=10)": { short: "Dyna-Q", module: "Module 4 · Planning + learning",
    idea: "Builds its own model of the Moon and rehearses ten imagined steps for every real one." },
  DQN: { short: "Deep Q-Network", module: "Module 5 · Deep RL",
    idea: "A neural network reads a 5×5 camera view, so it can drive on terrain it has never seen." },
  Heuristic: { short: "Rule-based", module: "Baseline",
    idea: "Hand-written rules with the full map and shortest paths. Strong, but given knowledge the learners never see." },
  Random: { short: "Random", module: "Baseline", idea: "Presses random valid buttons." },
};

/** Plain-English narration of what the rover is doing. */
export const NARRATION: Record<string, [string, "good" | "bad" | "warn" | "info" | "dim"]> = {
  move: ["Driving", "dim"],
  slip: ["Wheels slipped on loose regolith and the rover veered off course", "warn"],
  blocked: ["Obstacle ahead, so the drive command was rejected", "warn"],
  crater: ["Drove into a crater. Rover lost.", "bad"],
  libs_ok: ["Firing the LIBS laser: sulphur detected", "good"],
  apxs_ok: ["APXS spectrometer reading: Mg, Al, Si, Ca, Ti, Fe", "good"],
  wrong_instrument: ["Wrong instrument for this site, so nothing was measured", "bad"],
  tx_ok: ["Science relayed to Earth through Vikram", "good"],
  tx_fail: ["Transmit failed: out of radio range or no data", "bad"],
  recharge: ["Charging the solar panel", "info"],
  recharge_shadow: ["No sunlight in the permanent shadow, so no charge", "warn"],
  sleep_safe: ["Parked in sunlight with payloads off. Entering sleep mode.", "good"],
  sleep_unsafe: ["Unsafe hibernation (shadow or low battery)", "bad"],
  battery_dead: ["Battery depleted", "bad"],
  night: ["Lunar night fell while the rover was still awake", "bad"],
};

export const ACTION_LABEL: Record<string, string> = {
  N: "Drive N", NE: "Drive NE", E: "Drive E", SE: "Drive SE", S: "Drive S", SW: "Drive SW", W: "Drive W",
  NW: "Drive NW", LIBS: "LIBS laser", APXS: "APXS scan", TRANSMIT: "Transmit", RECHARGE: "Recharge", SLEEP: "Sleep",
};
