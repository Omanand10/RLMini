import { BOULDER, CRATER, DATA, SHADOW, SLOPE, VIKRAM, COMM_RADIUS } from "../sim/env";
import { AGENT_INFO } from "../sim/episodes";
import { FailureModes, Generalization, LearningCurves } from "./charts";
import "./story.css";

const pct = (v: number) => `${v > 0 && v < 0.1 ? (v * 100).toFixed(1) : Math.round(v * 100)}%`;
const REPO = "https://github.com/Omanand10/RLMini";

function Clock({ children }: { children: string }) {
  return <p className="clock"><i />{children}</p>;
}

// ---- the real mission, explained plainly -------------------------------------------------------
export function Mission() {
  const mapping: [string, string, string][] = [
    ["Pragyan carried two instruments: a laser (LIBS) and an X-ray spectrometer (APXS).",
      "Four science sites on the map. Sulphur sites need the laser, element sites need the X-ray scan.", "Actions and goal states"],
    ["Pragyan could not talk to Earth. Only Vikram could, so the rover had to stay close enough to relay.",
      "Results count only when the rover transmits from inside Vikram's radio range.", "Reward for relaying"],
    ["It ran on solar power for one lunar day, about 14 Earth days, then the freezing night.",
      "A battery that drains as it drives, a sun that moves from morning to dusk, and an episode that ends at night.", "State and episode end"],
    ["On 27 August a 4 m crater appeared 3 m ahead and the team on Earth had to reroute it.",
      "Craters end the episode, and wheels can slip sideways into them.", "Terminal hazard states"],
    ["Before night it was parked in sunlight with its instruments off and put to sleep.",
      "A sleep action that pays off only at dusk, in sunlight, with enough charge.", "Final goal"],
  ];
  return (
    <section className="section light" id="mission">
      <div className="wrap">
        <div className="opener">
          <Clock>Sunrise: the real mission</Clock>
          <h2 className="t-h2">What Chandrayaan-3 did</h2>
          <p className="t-lede muted">The short version, and why it makes a good reinforcement-learning problem.</p>
        </div>
        <div className="mission-story">
          <div className="story-text">
            <p>On 23 August 2023, India's Chandrayaan-3 landed near the Moon's south pole, closer to it than any mission before. It had two parts. <b>Vikram</b> was the lander. It stayed where it touched down. <b>Pragyan</b> was a 26 kg six-wheeled rover that rolled down a ramp from Vikram.</p>
            <p>Pragyan's job was to drive to interesting patches of ground, measure what they are made of, and get those results back to Earth. Its laser instrument found sulphur in the soil. Over about ten days it drove roughly 100 metres.</p>
            <p>Every move was planned by engineers on Earth and sent up as commands. The rover had limited power, one radio link and only one lunar day, and the ground was full of craters. So each command was a trade-off: drive further for more science, or save battery and stay safe.</p>
          </div>
          <ol className="timeline vertical">
            <li><time>23 Aug</time><p>Vikram lands. Pragyan rolls out.</p></li>
            <li><time>27 Aug</time><p>Crater ahead. Pragyan is commanded to back up and take a new route.</p></li>
            <li><time>29 Aug</time><p>ISRO reports sulphur, measured on site by the laser.</p></li>
            <li><time>2 Sep</time><p>Parked in sunlight and put to sleep before the lunar night.</p></li>
          </ol>
        </div>
        <h3 className="t-h3 mapping-title">From the real mission to our simulation</h3>
        <table className="mapping">
          <thead><tr><th>On the Moon</th><th>In our simulation</th><th>RL term</th></tr></thead>
          <tbody>
            {mapping.map(([real, sim, term]) => (
              <tr key={term}><td>{real}</td><td>{sim}</td><td>{term}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ---- what we built, in plain words + RL vocabulary ------------------------------------------------
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
      <text x="115" y="168" textAnchor="middle" className="loop-small on-ink-soft">the rover's decision-maker</text>
      <rect x="430" y="100" width="190" height="100" rx="8" fill="none" stroke="#161615" strokeWidth="2" />
      <text x="525" y="143" textAnchor="middle" className="loop-big">Environment</text>
      <text x="525" y="168" textAnchor="middle" className="loop-small">the simulated Moon</text>
      <path d="M210 125 C 300 50, 340 50, 430 125" fill="none" stroke="#161615" strokeWidth="2" markerEnd="url(#ah)" />
      <text x="320" y="52" textAnchor="middle" className="loop-label">action aₜ: one command</text>
      <path d="M430 175 C 340 250, 300 250, 210 175" fill="none" stroke="#c9a13b" strokeWidth="3" markerEnd="url(#ah)" />
      <text x="320" y="262" textAnchor="middle" className="loop-label">reward rₜ and next state sₜ₊₁</text>
    </svg>
  );
}

export function HowItWorks() {
  const terms: [string, string][] = [
    ["Agent", "The program that drives the rover. It starts knowing nothing about the Moon."],
    ["Environment", "Our simulated landing zone: a 12 × 12 m grid with craters, slopes, shadow, the sun and Vikram."],
    ["State s", "What the rover knows right now: where it is, battery level, which sites are done, and the time of day."],
    ["Action a", "One of 13 commands: drive in 8 directions, fire the laser, run the X-ray scan, transmit, recharge, or sleep."],
    ["Reward r", "Points after every action. Science and relaying earn points; wasted steps cost a little; craters cost a lot."],
    ["Episode", "One full lunar day, from rolling off Vikram until the rover sleeps, crashes, runs flat or night falls."],
    ["Start state", "Next to Vikram, battery full, morning, nothing sampled yet."],
    ["Goal states", "All four results sent to Earth and the rover asleep in sunlight at dusk. This earns the biggest reward."],
    ["Terminal states", "Moments that end the episode: falling into a crater, an empty battery, nightfall, or going to sleep."],
    ["Policy π", "The rover's rulebook: for each state, which action to take. Training produces it."],
    ["Q-value Q(s, a)", "The agent's estimate of the total reward it will collect if it takes action a in state s and plays well afterwards."],
    ["Exploration", "Trying random actions early on (ε-greedy), so the agent discovers what works instead of repeating its first guess."],
  ];
  return (
    <section className="section morning" id="how">
      <div className="wrap">
        <div className="opener">
          <Clock>Early morning: what we built</Clock>
          <h2 className="t-h2">A rover that learns its own plan</h2>
          <p className="t-lede muted">We built a simulator of Pragyan's landing site. A program then practised the mission hundreds of thousands of times, learning only from points and penalties. Nobody told it the route.</p>
        </div>
        <div className="how-grid">
          <div>
            <Loop />
            <ol className="steps">
              <li><b>Try.</b> In each state the agent picks an action. Early on it is mostly random.</li>
              <li><b>Get scored.</b> The simulator moves the rover, may make the wheels slip, and returns a reward.</li>
              <li><b>Update.</b> The agent adjusts its estimate Q(s, a): new estimate = old estimate + α × (reward + γ × best value of the next state − old estimate).</li>
              <li><b>Repeat.</b> After about 300,000 practice days the estimates settle. The final policy simply picks the action with the highest Q-value.</li>
            </ol>
          </div>
          <dl className="glossary">
            {terms.map(([t, d]) => <div key={t}><dt>{t}</dt><dd>{d}</dd></div>)}
          </dl>
        </div>
      </div>
    </section>
  );
}

// ---- the MDP ---------------------------------------------------------------------------------
function MapChart() {
  const m = DATA.map, S = 40, N = m.terrain.length;
  const fill = (t: number) => (t === SHADOW ? "#3b3f48" : t === SLOPE ? "#a9a59c" : "#c4c1ba");
  const [vr, vc] = m.vikram;
  return (
    <svg viewBox={`-2 -2 ${N * S + 4} ${N * S + 4}`} className="mapchart" role="img" aria-label="Top-down map of the landing zone">
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

export function Problem() {
  const K = DATA.consts;
  const rewards: [string, number][] = [
    ["Each sample taken with the right instrument", K.R_SAMPLE], ["Each result relayed to Earth", K.R_SENT],
    ["Safe sleep at dusk in sunlight", K.R_SLEEP_SAFE], ["Bonus when all four results reached Earth", K.R_MISSION],
    ["Every step (time and energy)", K.R_STEP], ["Wrong instrument or failed transmit", K.R_WRONG],
    ["Caught awake by lunar night", K.R_NIGHT], ["Sleeping in shadow or on low battery", K.R_SLEEP_UNSAFE],
    ["Falling into a crater or running out of battery", K.R_CRATER],
  ];
  return (
    <section className="section morning" id="problem">
      <div className="wrap">
        <div className="opener">
          <Clock>Morning: setting up the problem</Clock>
          <h2 className="t-h2">The day as a decision problem</h2>
          <p className="t-lede muted">At every step the rover picks one command. The Moon answers with a new situation and a score. Reinforcement learning finds the commands that earn the most over the whole day.</p>
        </div>
        <div className="problem-grid">
          <figure className="map-figure">
            <MapChart />
            <figcaption className="t-small muted">The 12 × 12 m landing zone every agent trained on. One square is one metre.</figcaption>
          </figure>
          <dl className="map-key">
            <div><dt><span className="key-swatch site-s" />Sulphur site</dt><dd>Needs the LIBS laser. Two on the map.</dd></div>
            <div><dt><span className="key-swatch site-e" />Element site</dt><dd>Needs the APXS spectrometer. Two on the map.</dd></div>
            <div><dt><span className="key-swatch vikram" />Vikram lander</dt><dd>Results reach Earth only inside the dashed square around it.</dd></div>
            <div><dt><span className="key-swatch crater" />Crater</dt><dd>Driving in ends the mission. Wheels can slip sideways into one.</dd></div>
            <div><dt><span className="key-swatch shadow" />Permanent shadow</dt><dd>The solar panel cannot charge here.</dd></div>
            <div><dt><span className="key-swatch slope" />Slope</dt><dd>Climbing drains the battery faster.</dd></div>
            <div><dt><span className="key-swatch boulder" />Boulder</dt><dd>Cannot be driven through.</dd></div>
            <div><dt><span className="key-swatch start" />Start</dt><dd>Where Pragyan rolls off Vikram's ramp each morning.</dd></div>
          </dl>
        </div>
        <div className="mdp">
          <article>
            <h3 className="t-h3">What the rover knows</h3>
            <p>Its position, battery level (0 to 5), the status of each site (not sampled, sampled, or sent to Earth) and the time of day (morning, noon or dusk). That is {DATA.valueIteration.n_reachable_states.toLocaleString()} reachable situations.</p>
            <p className="muted">The neural-network agent instead sees a 5 × 5 m camera window around itself, like Pragyan's navigation cameras.</p>
          </article>
          <article>
            <h3 className="t-h3">What it can do</h3>
            <ul className="commands">
              <li><b>Drive</b> in any of 8 directions. Ten percent of the time the wheels slip and it veers 45°.</li>
              <li><b>LIBS laser</b> or <b>APXS scan</b> to sample the site underneath.</li>
              <li><b>Transmit</b> held results through Vikram.</li>
              <li><b>Recharge</b> the battery in sunlight.</li>
              <li><b>Sleep</b> for the lunar night, allowed from dusk.</li>
            </ul>
          </article>
          <article>
            <h3 className="t-h3">How it is scored</h3>
            <table className="ledger">
              <tbody>
                {rewards.map(([l, v]) => (
                  <tr key={l}><td>{l}</td><td className={`tnum ${v > 0 ? "pos" : "neg"}`}>{v > 0 ? `+${v}` : v}</td></tr>
                ))}
              </tbody>
            </table>
          </article>
        </div>
      </div>
    </section>
  );
}

// ---- algorithms --------------------------------------------------------------------------------
export function Algorithms() {
  const order = ["Value Iteration", "Q-learning", "Dyna-Q (k=10)", "SARSA", "DQN", "n-step SARSA (n=4)", "Monte Carlo"];
  const max = DATA.table["Value Iteration"].return_mean, min = -40;
  return (
    <section className="section noon" id="algorithms">
      <div className="wrap">
        <div className="opener">
          <Clock>Noon: seven ways to learn</Clock>
          <h2 className="t-h2">Every algorithm from the syllabus, on the same Moon</h2>
          <p className="t-lede muted">Each one started knowing nothing, practised for hundreds of thousands of missions, and was then tested on 500 new ones. The bar shows its average score per mission.</p>
        </div>
        <ul className="algos">
          {order.map((n) => {
            const r = DATA.table[n], info = AGENT_INFO[n];
            const w = Math.max(2, ((r.return_mean - min) / (max - min)) * 100);
            return (
              <li key={n} className={n === "Value Iteration" ? "reference" : n === "Q-learning" ? "best" : ""}>
                <div className="algo-name">
                  <h3 className="t-h3">{info.short}</h3>
                  <span className="t-small muted">{info.module}</span>
                </div>
                <p className="algo-idea">{info.idea}</p>
                <div className="algo-score">
                  <span className="bar"><i style={{ width: `${w}%` }} /></span>
                  <b className="tnum">{r.return_mean.toFixed(0)}</b>
                </div>
              </li>
            );
          })}
        </ul>
        <p className="t-small muted algos-note">Score = average reward per mission. Value Iteration is given the full rules of the Moon in advance, so it marks the best possible score rather than competing. The rule-based planner scores {DATA.table.Heuristic.return_mean.toFixed(0)} and a random rover {DATA.table.Random.return_mean.toFixed(0)}.</p>
      </div>
    </section>
  );
}

// ---- results -------------------------------------------------------------------------------------
export function Results() {
  const T = DATA.table, q = T["Q-learning"], s = T.SARSA, d = T["Dyna-Q (k=10)"], g = DATA.generalization.DQN;
  return (
    <section className="section morning" id="results">
      <div className="wrap">
        <div className="opener">
          <Clock>Afternoon: what they learned</Clock>
          <h2 className="t-h2">From falling into craters to finishing the mission</h2>
          <p className="t-lede muted">All agents start at about −110 per mission, almost always lost in a crater within 20 steps. Here is how far each one got.</p>
        </div>
        <figure className="result">
          <LearningCurves />
          <figcaption>
            <b>Score per mission while learning.</b> Q-learning (gold) climbs highest, reaching {Math.round((100 * q.return_mean) / T["Value Iteration"].return_mean)}% of the optimal plan.
            Dyna-Q gets there with about five times fewer missions because it rehearses moves in its head.
          </figcaption>
        </figure>
        <div className="result-pair">
          <figure className="result">
            <FailureModes />
            <figcaption>
              <b>How missions end.</b> SARSA plays it safest: {pct(s.crater_rate)} crater losses against {pct(q.crater_rate)} for Q-learning, because it
              learns from the cautious moves it actually makes. The rule-based planner completes more missions, but its battery runs out in {pct(T.Heuristic.battery_rate)} of them.
            </figcaption>
          </figure>
          <figure className="result">
            <Generalization />
            <figcaption>
              <b>Terrain it has never seen.</b> The neural network still completes {pct(g.perturbed.success_rate)} of missions when new craters and boulders appear, which
              a lookup table cannot do. On completely random terrain it fails, an honest limit of this project.
            </figcaption>
          </figure>
        </div>
        <ul className="findings">
          <li><b>{q.sent_mean.toFixed(1)} of 4</b> results relayed to Earth per mission by Q-learning, up from 0 when it started.</li>
          <li><b>{pct(q.crater_rate)}</b> crater losses after training, down from {pct(T.Random.crater_rate)} for a random rover.</li>
          <li><b>{pct(d.success_rate)}</b> success for Dyna-Q after only 60,000 practice missions.</li>
          <li><b>{pct(q.battery_rate)}</b> battery failures for Q-learning against {pct(T.Heuristic.battery_rate)} for the hand-written rules.</li>
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
