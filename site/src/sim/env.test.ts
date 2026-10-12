/** The TypeScript simulator must reproduce the Python environment exactly. */
import { describe, expect, it } from "vitest";
import { DATA, features, makeMap, outcomes, sameState, startState, type State } from "./env";
import { dqnAct } from "./dqn";

describe("TS simulator == Python PragyanEnv", () => {
  it("reproduces every recorded transition", () => {
    const fixed = makeMap(DATA.map);
    let n = 0, bad = 0;
    for (const E of Object.values(DATA.episodes)) {
      const m = E.map ? makeMap(E.map) : fixed;
      let st = startState(m);
      for (const f of E.frames) {
        const next: State = [f[0], f[1], f[2], f[3], f[4]];
        const ok = outcomes(m, st, f[5]).some((o) => o[4] === f[7] && Math.abs(o[2] - f[6]) < 1e-6 &&
          (o[1] ? sameState(o[1], next) : sameState(st, next)));
        n++; if (!ok) bad++;
        st = next;
      }
    }
    expect(n).toBeGreaterThan(1000);
    expect(bad).toBe(0);
  });

  it("reproduces DQN features and decisions", () => {
    const m = makeMap(DATA.map);
    for (const ch of DATA.dqn.checks) {
      const st: State = [ch.state[0], ch.state[1], ch.state[2], ch.state[3], ch.state[4]];
      const f = features(m, st);
      ch.features.forEach((v, i) => expect(Math.abs(v - f[i])).toBeLessThan(1e-4));
      expect(dqnAct(m, st).a).toBe(ch.action);
    }
  });
});
