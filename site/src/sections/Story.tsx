import { BOULDER, COMM_RADIUS, CRATER, DATA, SHADOW, SLOPE, VIKRAM } from "../sim/env";
import { AGENT_INFO } from "../sim/episodes";
import { FailureModes, Generalization, LearningCurves } from "./charts";
import "./story.css";

const pct = (v: number) => `${v > 0 && v < 0.1 ? (v * 100).toFixed(1) : Math.round(v * 100)}%`;
const REPO = "https://github.com/Omanand10/RLMini";

// ---- where the idea comes from (kept short: the simulation is the point) --------------------------
export function Mission() {
  return (
    <section className="section panel light" id="mission">
      <div className="wrap context">
        <h2 className="t-h3">Where the idea comes from</h2>
        <p>
          In August 2023, India's Chandrayaan-3 landed near the Moon's south pole. Its rover, <b>Pragyan</b>, had
          about 14 Earth days of sunlight to drive around, analyse the soil with two instruments and send the results
          to Earth through the lander, <b>Vikram</b>, before the freezing night. Engineers on Earth planned every move.
          We asked whether a reinforcement-learning agent could learn to plan that day by itself.
        </p>
      </div>
    </section>
  );
}

// ---- the RL problem: loop, map and the MDP in one place ---------------------------------------------
function Loop() {
  return (
    <svg viewBox="0 0 640 300" className="loop" role="img" aria-label="The reinforcement learning loop">
      <defs>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#161615" />
        </marker>
      </defs>
      <rect x="20" y="100" width="190" height="100" rx="8" fill="#161615" />
      <text x="115" y="143" textAnchor="middle" className="loop-big on-ink">Agent</text>
      <text x="115" y="168" textAnchor="middle" className="loop-small on-ink-soft">picks the rover's command</text>
      <rect x="430" y="100" width="190" height="100" rx="8" fill="none" stroke="#161615" strokeWidth="2" />
      <text x="525" y="143" textAnchor="middle" className="loop-big">Environment</text>
      <text x="525" y="168" textAnchor="middle" className="loop-small">the simulated Moon</text>
      <path d="M210 125 C 300 50, 340 50, 430 125" fill="none" stroke="#161615" strokeWidth="2" markerEnd="url(#ah)" />
      <text x="320" y="52" textAnchor="middle" className="loop-label">action aₜ</text>
      <path d="M430 175 C 340 250, 300 250, 210 175" fill="none" stroke="#c9a13b" strokeWidth="3" markerEnd="url(#ah)" />
      <text x="320" y="262" textAnchor="middle" className="loop-label">reward rₜ₊₁ and next state sₜ₊₁</text>
    </svg>
  );
}

function MapChart() {
  const m = DATA.map, S = 40, N = m.terrain.length;
  const fill = (t: number) => (t === SHADOW ? "#3b3f48" : t === SLOPE ? "#a9a59c" : "#c4c1ba");
  const [vr, vc] = m.vikram;
  return (
    <svg viewBox={`-2 -2 ${N * S + 4} ${N * S + 4}`} className="mapchart" role="img" aria-label="Top-down map of the environment">
      {m.terrain.map((row, r) => row.map((t, c) => (
        <rect key={`${r}-${c}`} x={c * S} y={r * S} width={S} height={S} fill={fill(t)} stroke="#161615" strokeOpacity={0.08} />
      )))}
      {m.terrain.map((row, r) => row.map((t, c) => t === CRATER ? <circle key={`c${r}-${c}`} cx={c * S + S / 2} cy={r * S + S / 2} r={S * 0.4} fill="#161615" />
        : t === BOULDER ? <polygon key={`b${r}-${c}`} points={`${c * S + 9},${r * S + 30} ${c * S + 14},${r * S + 12} ${c * S + 27},${r * S + 9} ${c * S + 33},${r * S + 27}`} fill="#5e5b55" />
        : t === VIKRAM ? <rect key="v" x={c * S + 6} y={r * S + 6} width={S - 12} height={S - 12} fill="#c9a13b" /> : null))}
      <rect x={(vc - COMM_RADIUS) * S} y={(vr - COMM_RADIUS) * S} width={(2 * COMM_RADIUS + 1) * S} height={(2 * COMM_RADIUS + 1) * S}
        fill="none" stroke="#161615" strokeDasharray="6 6" strokeWidth={1.5} />
      {m.sites.map(([r, c, typ], i) => (
        <g key={i}>
          <circle cx={c * S + S / 2} cy={r * S + S / 2} r={13} fill="none" stroke={typ === 0 ? "#d8452e" : "#3e64b8"} strokeWidth={3.5} />
          <text x={c * S + S / 2} y={r * S + S / 2 + 4} textAnchor="middle" className="site-letter">{typ === 0 ? "S" : "E"}</text>
        </g>
      ))}
      <circle cx={m.start[1] * S + S / 2} cy={m.start[0] * S + S / 2} r={7} fill="#161615" />
    </svg>
  );
}

const REWARDS = (K: Record<string, number>): [string, number][] => [
  ["Sample a site with the right instrument", K.R_SAMPLE], ["Relay a result to Earth (per sample)", K.R_SENT],
  ["Sleep safely at dusk in sunlight", K.R_SLEEP_SAFE], ["Bonus: all four results relayed before sleeping", K.R_MISSION],
  ["Every step (time and energy)", K.R_STEP], ["Wrong instrument or failed transmit", K.R_WRONG],
  ["Night falls while awake", K.R_NIGHT], ["Sleep in shadow or on low battery", K.R_SLEEP_UNSAFE],
  ["Crater or empty battery", K.R_CRATER],
];

export function RLProblem() {
  return (
    <section className="section panel morning" id="problem">
      <div className="wrap">
        <div className="panel-head">
          <h2 className="t-h2">The reinforcement-learning problem</h2>
          <p className="t-lede muted">The rover's lunar day written as a Markov decision process.</p>
        </div>
        <div className="rl-grid">
          <figure className="map-figure">
            <MapChart />
            <figcaption>
              <span><i className="key-swatch start" />Start state</span>
              <span><i className="key-swatch site-s" />Sulphur site (LIBS)</span>
              <span><i className="key-swatch site-e" />Element site (APXS)</span>
              <span><i className="key-swatch vikram" />Vikram; dashed = relay range</span>
              <span><i className="key-swatch crater" />Crater (terminal)</span>
              <span><i className="key-swatch shadow" />Shadow (no charging)</span>
              <span><i className="key-swatch slope" />Slope (drains battery)</span>
              <span><i className="key-swatch boulder" />Boulder (blocked)</span>
            </figcaption>
          </figure>
          <ul className="mdp-list">
            <li><b>Agent</b> The program that chooses the rover's next command. It starts with no knowledge of the map.</li>
            <li><b>Environment</b> A 12 × 12 m simulated landing zone, one cell per metre. It is stochastic:
              <ul>
                <li>drive commands slip 45° sideways 10% of the time,</li>
                <li>moves may drain the battery, more often on slopes,</li>
                <li>the sun moves morning → noon → dusk at random times.</li>
              </ul>
            </li>
            <li><b>State sₜ</b> The information the agent uses to choose its action at step t:
              <ul>
                <li>rover position (row, column) and battery level 0–5,</li>
                <li>each site: not sampled, sampled, or relayed,</li>
                <li>sun phase: morning, noon or dusk.</li>
              </ul>
            </li>
            <li><b>Actions aₜ</b> 13 commands:
              <ul>
                <li>drive in 8 directions,</li>
                <li>LIBS laser, APXS scan, transmit, recharge, sleep.</li>
              </ul>
            </li>
            <li><b>Episode</b> One lunar day, from the start state to a terminal state:
              <ul>
                <li>goal: all four results relayed, then a safe sleep at dusk,</li>
                <li>terminal: crater, empty battery, nightfall or sleep.</li>
              </ul>
            </li>
            <li><b>Policy π(s)</b> The action to take in each state. Agents learn Q(s, a), the expected return of action a in state s, and act greedily on it.</li>
          </ul>
        </div>
      </div>
    </section>
  );
}

export function Learning() {
  const rewards = REWARDS(DATA.consts);
  return (
    <section className="section panel morning" id="learning">
      <div className="wrap">
        <div className="panel-head">
          <h2 className="t-h2">Rewards and learning</h2>
          <p className="t-lede muted">The agent only ever sees these numbers. Everything it does is learned from them.</p>
        </div>
        <div className="rl-lower">
          <div>
            <Loop />
            <ol className="steps">
              <li><b>Act.</b> In state sₜ choose an action: random with probability ε (exploration), otherwise the best known.</li>
              <li><b>Observe</b> the reward rₜ₊₁ and the next state sₜ₊₁.</li>
              <li><b>Update</b> the estimate:
                <span className="formula">Q(s, a) ← Q(s, a) + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ]</span>
              </li>
              <li><b>Repeat</b> for 300,000 episodes while ε decays from 1.0 to 0.05. Final policy: π(s) = argmaxₐ Q(s, a).</li>
            </ol>
          </div>
          <div>
            <h3 className="t-h3">Reward function</h3>
            <table className="ledger">
              <tbody>
                {rewards.map(([l, v]) => (
                  <tr key={l}><td>{l}</td><td className={`tnum ${v > 0 ? "pos" : "neg"}`}>{v > 0 ? `+${v}` : v}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="t-small muted">Discount factor γ = 0.99. The agent maximises the return G = r₁ + γr₂ + γ²r₃ + …</p>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---- algorithms -----------------------------------------------------------------------------------
export function Algorithms() {
  const order = ["Value Iteration", "Q-learning", "Dyna-Q (k=10)", "SARSA", "DQN", "n-step SARSA (n=4)", "Monte Carlo", "Heuristic", "Random"];
  const T = DATA.table;
  return (
    <section className="section panel noon" id="algorithms">
      <div className="wrap">
        <div className="panel-head">
          <h2 className="t-h2">Seven algorithms, one environment</h2>
          <p className="t-lede muted">All agents trained on the same simulator. Each learned policy was then frozen and run greedily on 500 new episodes. Value Iteration knows the exact model, so it shows the best possible score.</p>
        </div>
        <div className="algo-table-wrap">
          <table className="algo-table">
            <thead>
              <tr><th>Agent</th><th>How it learns</th><th className="num">Avg return</th><th className="num">Goal reached</th><th className="num">Crater losses</th><th className="num">Samples relayed</th></tr>
            </thead>
            <tbody>
              {order.map((n) => {
                const r = T[n], info = AGENT_INFO[n];
                return (
                  <tr key={n} className={n === "Value Iteration" ? "reference" : n === "Q-learning" ? "best" : ["Heuristic", "Random"].includes(n) ? "baseline" : ""}>
                    <td>
                      <b>{info.short}{n === "Q-learning" && <em className="best-tag">Best learning agent</em>}{n === "Value Iteration" && <em className="best-tag ref">Optimal, knows the model</em>}</b>
                      <span className="tags">{info.tags.map((t) => <i key={t}>{t}</i>)}</span>
                    </td>
                    <td>{info.idea}</td>
                    <td className="num tnum">{r.return_mean.toFixed(1)}</td>
                    <td className="num tnum">{pct(r.success_rate)}</td>
                    <td className="num tnum">{pct(r.crater_rate)}</td>
                    <td className="num tnum">{r.sent_mean.toFixed(2)} / 4</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="t-small muted algos-note">Value Iteration is the optimal reference: it is given the exact model and does not learn. Q-learning is the best of the agents that learn from experience. Rule-based and Random are baselines.</p>
      </div>
    </section>
  );
}

// ---- results -----------------------------------------------------------------------------------------
export function Results() {
  const T = DATA.table, q = T["Q-learning"], s = T.SARSA, g = DATA.generalization.DQN;
  const sw = DATA.sweeps;
  return (
    <>
      <section className="section panel morning" id="results">
        <div className="wrap">
          <div className="panel-head">
            <h2 className="t-h2">Results: training</h2>
            <p className="t-lede muted">Return per episode while each agent learned.</p>
          </div>
          <div className="results-a">
            <figure className="result">
              <LearningCurves />
              <figcaption>Smoothed return of the training episodes (ε-greedy). Every agent starts near −110, almost always lost in a crater. Dashed lines are fixed references.</figcaption>
            </figure>
            <ul className="observations">
              <li><b>Off-policy vs on-policy.</b> Q-learning learns the greedy policy and cuts close to craters ({pct(q.crater_rate)} losses). SARSA learns from the exploring policy it follows and keeps a wider margin ({pct(s.crater_rate)}), at a lower return.</li>
              <li><b>Planning saves experience.</b> Dyna-Q replays ten imagined steps from its learned model per real step and plateaus in about five times fewer episodes. Without planning it scored {sw["sweep:planning=0"].return_mean.toFixed(0)} after 20,000 episodes; with k = 5, {sw["sweep:planning=5"].return_mean.toFixed(0)}.</li>
              <li><b>Long returns are noisy here.</b> Slips and the −100 crater penalty make multi-step returns high-variance, so one-step TD learned best. n-step SARSA got worse as n grew (n = 1: {sw["sweep:n=1"].return_mean.toFixed(0)}, n = 8: {sw["sweep:n=8"].return_mean.toFixed(0)}).</li>
              <li><b>Close to optimal, not there yet.</b> Q-learning reaches {Math.round((100 * q.return_mean) / T["Value Iteration"].return_mean)}% of the optimal return and was still improving at 300,000 episodes.</li>
            </ul>
          </div>
        </div>
      </section>
      <section className="section panel morning" id="results-policies">
        <div className="wrap">
          <div className="panel-head">
            <h2 className="t-h2">Results: final policies</h2>
            <p className="t-lede muted">Each trained policy run greedily on 500 new episodes.</p>
          </div>
          <div className="result-pair">
            <figure className="result">
              <h3 className="t-h3">How episodes end</h3>
              <FailureModes />
            </figure>
            <figure className="result">
              <h3 className="t-h3">Goal reached on maps never seen in training</h3>
              <Generalization />
              <figcaption>A Q-table only works on its training map. The DQN reads a local camera window, so it still reaches the goal in {pct(g.perturbed.success_rate)} of episodes when new hazards appear. Fully random terrain is beyond it.</figcaption>
            </figure>
          </div>
        </div>
      </section>
    </>
  );
}

// ---- footer ----------------------------------------------------------------------------------------
export function Night() {
  return (
    <footer className="section night" id="credits">
      <div className="wrap">
        <p className="night-line">Good night, Pragyan.</p>
        <p className="team">Om Dwivedi, 16010123213 &nbsp;&nbsp; Om Anand Jha, 16010123214</p>
        <div className="night-grid">
          <p className="muted">Reinforcement Learning Lab mini project, Department of Computer Engineering, K. J. Somaiya School of Engineering, 2026. The simulation is inspired by ISRO's Chandrayaan-3 mission and is not an ISRO product.</p>
          <ul>
            <li><a href={REPO}>Source code on GitHub</a></li>
            <li><a href={`${REPO}/raw/main/report/Pragyan_RL_Report.docx`}>Download the project report</a></li>
            <li><a href="#mission-control">Back to mission control</a></li>
          </ul>
        </div>
      </div>
    </footer>
  );
}
