import { BOULDER, COMM_RADIUS, CRATER, DATA, SHADOW, SLOPE, VIKRAM } from "../sim/env";
import { AGENT_INFO } from "../sim/episodes";
import { FailureModes, Generalization, LearningCurves } from "./charts";
import "./story.css";

const pct = (v: number) => `${v > 0 && v < 0.1 ? (v * 100).toFixed(1) : Math.round(v * 100)}%`;
const REPO = "https://github.com/Omanand10/RLMini";

// ---- where the idea comes from (kept short: the simulation is the point) --------------------------
export function Mission() {
  return (
    <section className="section light compact" id="mission">
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

export function RLProblem() {
  const K = DATA.consts;
  const rewards: [string, number][] = [
    ["Sample a site with the right instrument", K.R_SAMPLE], ["Relay a result to Earth (per sample)", K.R_SENT],
    ["Sleep safely at dusk in sunlight", K.R_SLEEP_SAFE], ["Bonus: all four results relayed before sleeping", K.R_MISSION],
    ["Every step (time and energy)", K.R_STEP], ["Wrong instrument or failed transmit", K.R_WRONG],
    ["Night falls while awake", K.R_NIGHT], ["Sleep in shadow or on low battery", K.R_SLEEP_UNSAFE],
    ["Crater or empty battery", K.R_CRATER],
  ];
  return (
    <section className="section morning" id="problem">
      <div className="wrap">
        <div className="opener">
          <h2 className="t-h2">The reinforcement-learning problem</h2>
          <p className="t-lede muted">The rover's day as an MDP: what the agent sees, what it can do, how it is scored, and how it learns from that score.</p>
        </div>

        <div className="rl-grid">
          <div className="rl-visuals">
            <Loop />
            <figure className="map-figure">
              <MapChart />
              <figcaption>
                <span><i className="key-swatch start" />Start state</span>
                <span><i className="key-swatch site-s" />Sulphur site (LIBS)</span>
                <span><i className="key-swatch site-e" />Element site (APXS)</span>
                <span><i className="key-swatch vikram" />Vikram; dashed square = relay range</span>
                <span><i className="key-swatch crater" />Crater (terminal)</span>
                <span><i className="key-swatch shadow" />Shadow (no charging)</span>
                <span><i className="key-swatch slope" />Slope (drains battery)</span>
                <span><i className="key-swatch boulder" />Boulder (blocked)</span>
              </figcaption>
            </figure>
          </div>

          <ul className="mdp-list">
            <li><b>Agent</b> The program that chooses the rover's next command. It starts with no knowledge of the map.</li>
            <li><b>Environment</b> A 12 × 12 m simulation of the landing zone, one cell per metre. It is stochastic:
              <ul>
                <li>a drive command slips 45° to the side 10% of the time,</li>
                <li>each move may drain the battery, more often on slopes,</li>
                <li>the sun moves from morning to noon to dusk at random times.</li>
              </ul>
            </li>
            <li><b>State sₜ</b> The information the agent uses to choose its action at step t:
              <ul>
                <li>rover position (row and column),</li>
                <li>battery level, 0 to 5,</li>
                <li>status of each of the 4 sites: not sampled, sampled, or relayed,</li>
                <li>sun phase: morning, noon or dusk.</li>
              </ul>
              <span className="note">{DATA.valueIteration.n_reachable_states.toLocaleString()} different states can occur. The neural-network agent sees a 5 × 5 m window around the rover instead of the whole map.</span>
            </li>
            <li><b>Actions aₜ</b> 13 commands:
              <ul>
                <li>drive N, NE, E, SE, S, SW, W or NW,</li>
                <li>LIBS laser or APXS scan on the current cell,</li>
                <li>transmit held results through Vikram,</li>
                <li>recharge from sunlight,</li>
                <li>sleep for the night (allowed from dusk).</li>
              </ul>
            </li>
            <li><b>Episode</b> One lunar day.
              <ul>
                <li>Start state: next to Vikram, battery full, morning, nothing sampled.</li>
                <li>Goal: all four results relayed, then a safe sleep at dusk.</li>
                <li>Terminal states: crater, empty battery, nightfall, or sleeping.</li>
              </ul>
            </li>
            <li><b>Policy π(s)</b> What training produces: the action to take in each state. The agents learn Q(s, a), the expected return of taking action a in state s, and act greedily on it.</li>
          </ul>
        </div>

        <div className="rl-lower">
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
          <div>
            <h3 className="t-h3">How an agent learns</h3>
            <ol className="steps">
              <li><b>Act.</b> In state sₜ, choose an action. With probability ε it is random (exploration), otherwise the best one known so far.</li>
              <li><b>Observe.</b> The environment returns the reward rₜ₊₁ and the next state sₜ₊₁.</li>
              <li><b>Update.</b> Q-learning moves its estimate towards the target:
                <span className="formula">Q(s, a) ← Q(s, a) + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ]</span>
              </li>
              <li><b>Repeat</b> for 300,000 episodes while ε decays from 1.0 to 0.05. The final policy is greedy: π(s) = argmaxₐ Q(s, a).</li>
            </ol>
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
    <section className="section noon" id="algorithms">
      <div className="wrap">
        <div className="opener">
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
                      <b>{info.short}</b>
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
        <p className="t-small muted algos-note">Avg return = mean total reward per episode. Goal reached = all four results relayed and a safe sleep. Rule-based and Random are baselines, not RL.</p>
      </div>
    </section>
  );
}

// ---- results -----------------------------------------------------------------------------------------
export function Results() {
  const T = DATA.table, q = T["Q-learning"], s = T.SARSA, g = DATA.generalization.DQN;
  const sw = DATA.sweeps;
  return (
    <section className="section morning" id="results">
      <div className="wrap">
        <div className="opener">
          <h2 className="t-h2">Results</h2>
          <p className="t-lede muted">How the return improved during training, how the final policies behave, and what that says about each method.</p>
        </div>

        <figure className="result">
          <h3 className="t-h3">Return per episode during training</h3>
          <LearningCurves />
          <figcaption>Smoothed return of the training episodes (ε-greedy, so slightly below the final greedy score). Every agent starts near −110, almost always lost in a crater. Dashed lines are fixed references.</figcaption>
        </figure>

        <div className="result-pair">
          <figure className="result">
            <h3 className="t-h3">How episodes end with the final policy</h3>
            <FailureModes />
          </figure>
          <figure className="result">
            <h3 className="t-h3">Goal reached on maps never seen in training</h3>
            <Generalization />
          </figure>
        </div>

        <h3 className="t-h3 obs-title">What the results show</h3>
        <ul className="observations">
          <li><b>Off-policy vs on-policy.</b> Q-learning learns the greedy policy and takes shorter routes past craters ({pct(q.crater_rate)} losses). SARSA learns from the exploring policy it actually follows, so it keeps a wider margin ({pct(s.crater_rate)} losses) at the cost of a lower return.</li>
          <li><b>Planning saves experience.</b> Dyna-Q replays ten imagined transitions from its learned model for every real one, and reached its plateau in about five times fewer episodes than Q-learning. Without planning (k = 0) it scored {sw["sweep:planning=0"].return_mean.toFixed(0)} after 20,000 episodes; with k = 5 it scored {sw["sweep:planning=5"].return_mean.toFixed(0)}.</li>
          <li><b>Long returns are noisy here.</b> Wheel slips and the −100 crater penalty make multi-step returns high-variance, so one-step TD learned best. n-step SARSA got worse as n grew (n = 1: {sw["sweep:n=1"].return_mean.toFixed(0)}, n = 8: {sw["sweep:n=8"].return_mean.toFixed(0)}) and Monte Carlo was weakest.</li>
          <li><b>Function approximation generalises.</b> A Q-table only works on the map it was trained on. The DQN sees a local camera window and still reaches the goal in {pct(g.perturbed.success_rate)} of episodes on maps with new hazards. It fails on fully random terrain, a limit of this project.</li>
          <li><b>Close to optimal, not there yet.</b> Q-learning reaches {Math.round((100 * q.return_mean) / T["Value Iteration"].return_mean)}% of the optimal return and was still improving at 300,000 episodes. The rule-based planner scores slightly higher but uses the full map, which the learners never get.</li>
        </ul>
      </div>
    </section>
  );
}

// ---- footer ----------------------------------------------------------------------------------------
export function Night() {
  return (
    <footer className="section night">
      <div className="wrap">
        <p className="night-line">Good night, Pragyan.</p>
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
