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

export const AGENT_INFO: Record<string, { short: string; tags: string[]; idea: string }> = {
  "Value Iteration": { short: "Value Iteration", tags: ["Model-based", "Planning", "Table"],
    idea: "Is given the exact probabilities of the simulator and solves the Bellman equations. It gives the optimal policy, so it is our reference, not a learner." },
  "Monte Carlo": { short: "Monte Carlo", tags: ["Model-free", "On-policy", "Table"],
    idea: "Waits until the episode ends, then moves every visited Q(s, a) towards the full return that followed." },
  "Q-learning": { short: "Q-learning", tags: ["Model-free", "Off-policy", "Table"],
    idea: "After every step, moves Q(s, a) towards the reward plus the value of the best next action." },
  SARSA: { short: "SARSA", tags: ["Model-free", "On-policy", "Table"],
    idea: "Same update, but uses the next action it will actually take, exploration included, so it learns a more cautious policy." },
  "n-step SARSA (n=4)": { short: "n-step SARSA", tags: ["Model-free", "On-policy", "Table"],
    idea: "Updates from four real rewards before bootstrapping, instead of one." },
  "Dyna-Q (k=10)": { short: "Dyna-Q", tags: ["Learns a model", "Off-policy", "Table"],
    idea: "Q-learning plus a learned model of the simulator: after each real step it replays ten imagined ones." },
  DQN: { short: "Deep Q-Network", tags: ["Model-free", "Off-policy", "Neural network"],
    idea: "A neural network estimates Q(s, a) from a 5 × 5 m camera view, trained with experience replay and a target network." },
  Heuristic: { short: "Rule-based", tags: ["Baseline", "Hand-written"],
    idea: "Hand-written rules that use the full map and shortest paths. No learning." },
  Random: { short: "Random", tags: ["Baseline"], idea: "Picks a random valid action every step." },
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
