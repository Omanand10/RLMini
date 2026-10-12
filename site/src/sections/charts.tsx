import type { CSSProperties } from "react";
import { scaleLinear, scaleLog } from "d3-scale";
import { line } from "d3-shape";
import { DATA } from "../sim/env";
import { AGENT_INFO } from "../sim/episodes";

const INK = "#161615", SOFT = "#8e8b84", GOLD = "#c9a13b", SIGNAL = "#d8452e", SOLAR = "#3e64b8";
const LEARNERS = ["Q-learning", "SARSA", "Dyna-Q (k=10)", "n-step SARSA (n=4)", "Monte Carlo"];

/** Reward per mission while learning, one line per tabular agent, labelled at the line end. */
export function LearningCurves() {
  const W = 900, H = 420, m = { l: 56, r: 150, t: 20, b: 44 };
  const x = scaleLog().domain([60, 300000]).range([m.l, W - m.r]);
  const y = scaleLinear().domain([-120, 120]).range([H - m.b, m.t]);
  const refs: [string, number, string][] = [
    ["Optimal plan (Value Iteration)", DATA.table["Value Iteration"].return_mean, INK],
    ["Rule-based planner", DATA.table.Heuristic.return_mean, SOFT],
    ["Random", DATA.table.Random.return_mean, SIGNAL],
  ];
  const labels = LEARNERS.map((n) => ({ n, c: DATA.curves[n].returns }))
    .map(({ n, c }) => ({ n, y: y(c.y[c.y.length - 1]), x: x(c.x[c.x.length - 1]) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 16);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Reward per mission over training episodes">
      {[-100, -50, 0, 50, 100].map((v) => (
        <g key={v}>
          <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke={INK} strokeOpacity={v === 0 ? 0.35 : 0.08} />
          <text x={m.l - 10} y={y(v) + 4} textAnchor="end" className="tick">{v}</text>
        </g>
      ))}
      {[100, 1000, 10000, 100000].map((v) => (
        <text key={v} x={x(v)} y={H - m.b + 22} textAnchor="middle" className="tick">{v >= 1000 ? `${v / 1000}k` : v}</text>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle" className="axis">Training episodes (log scale)</text>
      {refs.map(([n, v, c]) => (
        <g key={n}>
          <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke={c} strokeDasharray="4 5" strokeOpacity={0.7} />
          <text x={m.l + 6} y={y(v) - 6} className="ref" fill={c}>{n}</text>
        </g>
      ))}
      {LEARNERS.map((n) => {
        const c = DATA.curves[n].returns;
        const d = line<number>().x((_, i) => x(c.x[i])).y((v) => y(v))(c.y);
        const hot = n === "Q-learning";
        return <path key={n} className={`curve ${hot ? "hot" : ""}`} pathLength={1} d={d ?? ""} fill="none" stroke={hot ? GOLD : INK} strokeOpacity={hot ? 1 : 0.42} strokeWidth={hot ? 3 : 1.6} />;
      })}
      {labels.map((l) => (
        <text key={l.n} x={l.x + 8} y={l.y + 4} className={`end-label ${l.n === "Q-learning" ? "hot" : ""}`}>{AGENT_INFO[l.n].short}</text>
      ))}
    </svg>
  );
}

/** How missions end, for each frozen policy. */
export function FailureModes() {
  const names = ["Value Iteration", "Q-learning", "SARSA", "Dyna-Q (k=10)", "DQN", "Heuristic", "Random"];
  const W = 640, rowH = 40, m = { l: 128, r: 10, t: 64 };
  const H = m.t + names.length * rowH + 10;
  const x = scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
  const parts: [keyof typeof DATA.table[string], string, string][] = [
    ["success_rate", "Mission complete", GOLD], ["crater_rate", "Lost in a crater", SIGNAL],
    ["battery_rate", "Battery ran out", SOLAR], ["night_rate", "Caught by night", INK],
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="How missions end for each agent">
      {parts.map(([, label, c], i) => (
        <g key={label} transform={`translate(${m.l + (i % 2) * 220}, ${8 + Math.floor(i / 2) * 24})`}>
          <rect width={12} height={12} fill={c} rx={2} />
          <text x={18} y={10} className="tick">{label}</text>
        </g>
      ))}
      {names.map((n, i) => {
        const r = DATA.table[n];
        let acc = 0;
        return (
          <g key={n} transform={`translate(0, ${m.t + i * rowH})`}>
            <text x={m.l - 12} y={rowH / 2 + 4} textAnchor="end" className="row-label">{AGENT_INFO[n].short}</text>
            <rect x={m.l} y={8} width={W - m.l - m.r} height={rowH - 16} fill={INK} fillOpacity={0.06} />
            {parts.map(([k, , c]) => {
              const v = r[k] as number, x0 = x(acc); acc += v;
              return <rect key={k} className="grow-x" style={{ "--r": i } as CSSProperties} x={x0} y={8} width={Math.max(0, x(acc) - x0)} height={rowH - 16} fill={c} />;
            })}
          </g>
        );
      })}
    </svg>
  );
}

/** Generalisation: the same policies on maps nobody trained on. */
export function Generalization() {
  const envs: ["fixed" | "perturbed" | "random", string][] = [["fixed", "Landing zone"], ["perturbed", "New hazards"], ["random", "Random terrain"]];
  const agents: [string, string][] = [["DQN", GOLD], ["Heuristic", SOFT], ["Random", SIGNAL]];
  const W = 640, H = 380, m = { l: 48, r: 10, t: 64, b: 40 };
  const y = scaleLinear().domain([0, 0.8]).range([H - m.b, m.t]);
  const gw = (W - m.l - m.r) / 3, bw = 52;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Success rate on unseen maps">
      {([["Deep Q-Network", GOLD], ["Rule-based", SOFT], ["Random", SIGNAL]] as [string, string][]).map(([label, c], i) => (
        <g key={label} transform={`translate(${m.l + i * 170}, 8)`}>
          <rect width={12} height={12} fill={c} rx={2} />
          <text x={18} y={10} className="tick">{label}</text>
        </g>
      ))}
      {[0, 0.2, 0.4, 0.6, 0.8].map((v) => (
        <g key={v}>
          <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke={INK} strokeOpacity={0.08} />
          <text x={m.l - 8} y={y(v) + 4} textAnchor="end" className="tick">{v * 100}%</text>
        </g>
      ))}
      {envs.map(([e, label], gi) => (
        <g key={e} transform={`translate(${m.l + gi * gw}, 0)`}>
          {agents.map(([a, c], ai) => {
            const v = DATA.generalization[a][e].success_rate;
            const bx = gw / 2 - (agents.length * bw) / 2 + ai * bw;
            return (
              <g key={a}>
                <rect className="grow-y" style={{ "--r": gi * 3 + ai } as CSSProperties} x={bx + 4} y={y(v)} width={bw - 8} height={y(0) - y(v)} fill={c} />
                <text x={bx + bw / 2} y={y(v) - 6} textAnchor="middle" className="tick">{v > 0 && v < 0.01 ? "<1" : Math.round(v * 100)}%</text>
              </g>
            );
          })}
          <text x={gw / 2} y={H - 12} textAnchor="middle" className="row-label">{label}</text>
        </g>
      ))}
    </svg>
  );
}
