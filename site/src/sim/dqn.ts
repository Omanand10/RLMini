/** In-browser forward pass of the trained DQN (weights exported from PyTorch). */
import { DATA, N_ACTIONS, features, validActions, type LunarMap, type State } from "./env";

export function qnet(x: Float32Array): Float32Array {
  let h = x;
  const layers = DATA.dqn.layers;
  layers.forEach((L, li) => {
    const out = new Float32Array(L.b.length);
    for (let j = 0; j < L.b.length; j++) {
      let s = L.b[j];
      const w = L.W[j];
      for (let i = 0; i < h.length; i++) s += w[i] * h[i];
      out[j] = li < layers.length - 1 ? Math.max(0, s) : s;
    }
    h = out;
  });
  return h;
}

/** Greedy action among valid commands, plus the Q-values and mask for display. */
export function dqnAct(m: LunarMap, st: State) {
  const q = qnet(features(m, st)), mask = validActions(m, st);
  let a = -1, best = -Infinity;
  for (let i = 0; i < N_ACTIONS; i++) if (mask[i] && q[i] > best) { best = q[i]; a = i; }
  return { a, q, mask };
}
