import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { useFrame } from "@react-three/fiber";
import { View3D } from "../scene/View3D";
import { LunarScene, type AddedHazard, type SceneOptions } from "../scene/LunarScene";
import { Player } from "../sim/player";
import { ACTION_LABEL, AGENT_INFO, NARRATION, buildSteps, fixedMap, mapFor } from "../sim/episodes";
import { ACTIONS, BOULDER, CRATER, DATA, MAX_BATTERY, REGOLITH, SENT, HELD, SHADOW, SLOPE, VIKRAM, makeMap, type LunarMap } from "../sim/env";
import "./mission-control.css";

/*
 * Layout, top to bottom, so it reads at a glance:
 *   controls row  ->  3D view with one status line  ->  goals | state | agent's choice
 */

type Mode = "replay" | "compare" | "live";
const PHASES = ["Morning", "Noon", "Dusk"];
const REPLAY_AGENTS = ["Q-learning", "SARSA", "Dyna-Q (k=10)", "DQN", "Value Iteration", "Heuristic", "Random",
  "n-step SARSA (n=4)", "Monte Carlo"];

function useP(p: Player) { useSyncExternalStore(p.subscribe, p.getVersion); }

function Driver({ players }: { players: Player[] }) {
  useFrame((_, dt) => { for (const p of players) p.tick(Math.min(dt, 0.1)); });
  return null;
}

// ---- 3D view with its status line ------------------------------------------------------------
function World({ player, opts, agent, small }: { player: Player; opts: SceneOptions; agent: string; small?: boolean }) {
  return (
    <View3D className={`mc-world ${small ? "small" : ""} ${opts.onCell ? "clickable" : ""}`}
      overlay={<><StatusLine player={player} agent={agent} />{!small && <Legend />}</>}>
      <Driver players={[player]} />
      <LunarScene player={player} opts={opts} />
    </View3D>
  );
}

function StatusLine({ player, agent }: { player: Player; agent: string }) {
  useP(player);
  const s = player.cur;
  let text = "Ready at the start state, next to Vikram";
  let tone = "dim";
  if (s) { [text, tone] = NARRATION[s.ev] ?? [s.ev, "dim"]; if (s.ev === "move") text = "Driving"; }
  else if (player.done) [text, tone] = outcomeText(player);
  return (
    <div className="mc-status">
      <div className="cell"><span>Agent</span><b>{AGENT_INFO[agent]?.short ?? agent}</b></div>
      <div className="cell"><span>Step t</span><b className="tnum">{player.i}</b></div>
      <div className="cell"><span>Action aₜ</span><b>{s ? ACTION_LABEL[ACTIONS[s.a]] : "–"}</b></div>
      <div className="cell"><span>Reward rₜ</span><b className={`tnum ${s ? (s.rew > 0 ? "pos" : s.rew < -5 ? "neg" : "") : ""}`}>{s ? (s.rew > 0 ? `+${s.rew}` : s.rew) : "–"}</b></div>
      <div className="cell"><span>Return G</span><b className="tnum">{player.score.toFixed(0)}</b></div>
      <p className={`what tone-${tone}`} aria-live="polite">{text}</p>
    </div>
  );
}

function Legend() {
  return (
    <ul className="mc-legend t-small" aria-label="Map key">
      <li><i className="lg-site" />Goal: science site</li>
      <li><i className="lg-range" />Relay zone (Vikram range)</li>
      <li><i className="lg-crater" />Hazard: crater (episode ends)</li>
      <li><i className="lg-shadow" />Shadow: no charging</li>
      <li><i className="lg-path" />Path taken</li>
    </ul>
  );
}

function outcomeText(p: Player): [string, string] {
  const sent = p.state[3].filter((s) => s === SENT).length;
  switch (p.endEv) {
    case "sleep_safe": return sent === 4 ? ["Episode over: goal reached. All four results sent, asleep in sunlight.", "good"]
      : [`Episode over: asleep safely, but only ${sent} of 4 results sent.`, "warn"];
    case "crater": return ["Episode over: fell into a crater.", "bad"];
    case "battery_dead": return ["Episode over: battery ran out.", "bad"];
    case "night": return ["Episode over: night fell before it slept.", "bad"];
    case "sleep_unsafe": return ["Episode over: slept in shadow or on low battery.", "bad"];
    default: return ["Episode over: ran out of steps.", "warn"];
  }
}

// ---- the three panels under the view -----------------------------------------------------------
function Goals({ player }: { player: Player }) {
  useP(player);
  const [, , bat, sites, phase] = player.state, m = player.map;
  const slept = player.done && player.endEv === "sleep_safe";
  return (
    <section className="mc-card">
      <h3>State and goal progress</h3>
      <dl className="state">
        <div><dt>Battery</dt><dd><span className="battery" aria-label={`${bat} of ${MAX_BATTERY}`}>
          {Array.from({ length: MAX_BATTERY }, (_, k) => <i key={k} className={k < bat ? (bat <= 2 ? "low" : "on") : ""} />)}
        </span></dd></div>
        <div><dt>Sun</dt><dd><span className="phases">{PHASES.map((ph, k) => <b key={ph} className={k === phase ? "on" : ""}>{ph}</b>)}</span></dd></div>
      </dl>
      <ul className="goals">
        {m.sites.map(([, , typ], i) => (
          <li key={i}>
            <span>{typ === 0 ? "Sulphur" : "Element"} site {i % 2 + 1} <em>{typ === 0 ? "LIBS" : "APXS"}</em></span>
            <span className="ticks">
              <b className={sites[i] >= HELD ? "on" : ""}>Sampled</b>
              <b className={sites[i] === SENT ? "on" : ""}>Relayed</b>
            </span>
          </li>
        ))}
        <li><span>Safe sleep at dusk</span><span className="ticks"><b className={slept ? "on" : ""}>Done</b></span></li>
      </ul>
    </section>
  );
}

/** Small top-down map: terrain, sites, relay range, path and rover. Clickable in live mode. */
function MiniMap({ player, onCell }: { player: Player; onCell?: (r: number, c: number, shift: boolean) => void }) {
  useP(player);
  const m = player.map, S = 10, N = m.H;
  const [r, c] = player.state, sites = player.state[3];
  const fill = (t: number) => (t === SHADOW ? "#2c3240" : t === SLOPE ? "#6d6a64" : t === BOULDER ? "#4a4743" : t === VIKRAM ? "#c9a13b" : "#8e8b84");
  const pts = player.track.map(([tr, tc]) => `${tc * S + S / 2},${tr * S + S / 2}`).join(" ");
  const [vr, vc] = m.vikram;
  return (
    <section className="mc-card minimap">
      <h3>Top-down map</h3>
      <svg viewBox={`0 0 ${N * S} ${N * S}`} className={onCell ? "clickable" : ""} role="img" aria-label="Top-down map of the episode"
        onClick={(e) => {
          if (!onCell) return;
          const box = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          onCell(Math.floor(((e.clientY - box.top) / box.height) * N), Math.floor(((e.clientX - box.left) / box.width) * N), e.shiftKey);
        }}>
        {m.terrain.map((row, rr) => row.map((t, cc) => (
          <rect key={`${rr}-${cc}`} x={cc * S} y={rr * S} width={S} height={S} fill={fill(t)} stroke="#000" strokeWidth={0.3} />
        )))}
        {m.terrain.map((row, rr) => row.map((t, cc) => t === CRATER ? <circle key={`k${rr}-${cc}`} cx={cc * S + S / 2} cy={rr * S + S / 2} r={S * 0.38} fill="#050505" /> : null))}
        <rect x={(vc - 3) * S} y={(vr - 3) * S} width={7 * S} height={7 * S} fill="none" stroke="#e9e7e2" strokeDasharray="2 2" strokeWidth={0.6} />
        {m.sites.map(([sr, sc, typ], i) => (
          <circle key={i} cx={sc * S + S / 2} cy={sr * S + S / 2} r={3.2} strokeWidth={1.4}
            stroke={typ === 0 ? "#d8452e" : "#6f8fd8"} fill={sites[i] === SENT ? "#c9a13b" : sites[i] === HELD ? "#e9d39b" : "none"} />
        ))}
        {player.track.length > 1 && <polyline points={pts} fill="none" stroke="#c9a13b" strokeWidth={1.2} strokeLinejoin="round" />}
        <circle cx={c * S + S / 2} cy={r * S + S / 2} r={2.8} fill="#fff" stroke="#000" strokeWidth={0.8} />
      </svg>
      <p className="t-small muted">White dot = rover. Rings: red = sulphur site, blue = element site; filled gold once relayed.{onCell ? " Click a cell to drop a crater here too." : ""}</p>
    </section>
  );
}

const TRAIN_EPISODES: Record<string, number> = { "Dyna-Q (k=10)": 60000 };
function epsilonAt(agent: string, ep: number) {
  const n = TRAIN_EPISODES[agent] ?? 300000;
  return Math.max(0.05, Math.pow(0.05, ep / (0.8 * n)));
}

/** What this episode is, in RL terms. */
function EpisodeInfo({ agent, epKey }: { agent: string; epKey: string }) {
  const E = DATA.episodes[epKey];
  const ep = Number((epKey.match(/train(\d+)/) ?? [])[1] ?? 0);
  let text: string;
  if (E.stage === "untrained") text = "The agent's first training episode. Its Q-values are all zero and ε = 1, so every action is random.";
  else if (E.stage === "learning" || E.stage === "improving")
    text = `Training episode ${ep.toLocaleString()}. It still explores (ε ≈ ${epsilonAt(agent, ep).toFixed(2)}) while its Q-values are only partly learned.`;
  else if (E.stage === "unseen") text = "A map the agent never trained on: new craters and boulders, same goals.";
  else if (agent === "Value Iteration") text = "The optimal policy, computed from the exact model. A reference for the learners.";
  else if (agent === "Heuristic" || agent === "Random") text = "A baseline policy, not learned.";
  else text = "A test episode with the final policy: no exploration (ε = 0). The agent always takes the action with the highest Q(s, a).";
  return (
    <section className="mc-card">
      <h3>About this episode</h3>
      <p>{text}</p>
      <p className="t-small muted">The map and the goal are the same in every episode. Only the random events change: wheel slips, battery drain and when the sun moves.</p>
    </section>
  );
}

// ---- controls ---------------------------------------------------------------------------------------
const stageOrder = ["untrained", "learning", "improving", "trained", "unseen"];
function episodesOf(agent: string) {
  return Object.entries(DATA.episodes).filter(([, e]) => e.agent === agent)
    .sort((a, b) => stageOrder.indexOf(a[1].stage) - stageOrder.indexOf(b[1].stage));
}

function EpisodePicker({ agent, setAgent, epKey, setEpKey, agents = REPLAY_AGENTS }: {
  agent: string; setAgent: (a: string) => void; epKey: string; setEpKey: (k: string) => void; agents?: string[];
}) {
  return (
    <>
      <label className="field"><span>Agent</span>
        <select value={agent} onChange={(e) => {
          const a = e.target.value; setAgent(a);
          setEpKey(episodesOf(a).find(([, x]) => x.stage === "trained")?.[0] ?? episodesOf(a)[0][0]);
        }}>
          {agents.map((a) => <option key={a} value={a}>{AGENT_INFO[a].short}</option>)}
        </select>
      </label>
      <label className="field"><span>Episode</span>
        <select value={epKey} onChange={(e) => setEpKey(e.target.value)}>
          {episodesOf(agent).map(([k, e]) => <option key={k} value={k}>{e.label}</option>)}
        </select>
      </label>
    </>
  );
}

function PlayControls({ players, speed, setSpeed }: { players: Player[]; speed: number; setSpeed: (n: number) => void }) {
  players.forEach(useP);
  const playing = players.some((p) => p.playing), allDone = players.every((p) => p.done);
  return (
    <>
      <button className="btn-primary" onClick={() => players.forEach((p) => (playing ? p.pause() : p.play()))}>
        {playing ? "Pause" : allDone ? "Play again" : players[0].i > 0 ? "Resume" : "Play"}
      </button>
      <button className="btn-ghost" onClick={() => players.forEach((p) => p.reset())}>Restart</button>
      <label className="field speed"><span>Speed</span>
        <input type="range" min={1} max={12} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
      </label>
    </>
  );
}

// ---- modes ----------------------------------------------------------------------------------------------
function useEpisode(player: Player, epKey: string) {
  useEffect(() => { const E = DATA.episodes[epKey], m = mapFor(E); player.loadSteps(m, buildSteps(E, m)); }, [epKey, player]);
}

function Replay({ speed, setSpeed }: { speed: number; setSpeed: (n: number) => void }) {
  const player = useMemo(() => new Player(fixedMap()), []);
  const [agent, setAgent] = useState("Q-learning");
  const [epKey, setEpKey] = useState("Q-learning|final0");
  useEpisode(player, epKey);
  useEffect(() => { player.speed = speed; }, [speed, player]);
  useP(player);
  return (
    <>
      <div className="mc-controls">
        <EpisodePicker agent={agent} setAgent={setAgent} epKey={epKey} setEpKey={setEpKey} />
        <PlayControls players={[player]} speed={speed} setSpeed={setSpeed} />
        <input className="scrub" type="range" min={0} max={player.length} value={player.i} aria-label="Jump to step"
          onChange={(e) => player.seek(+e.target.value)} />
      </div>
      <div className="mc-main">
        <World player={player} agent={agent} opts={{ labels: false, range: true, camera: "control" }} />
        <aside className="mc-side">
          <MiniMap player={player} />
          <EpisodeInfo agent={agent} epKey={epKey} />
          <Goals player={player} />
        </aside>
      </div>
    </>
  );
}

function CompareSide({ player, initial }: { player: Player; initial: [string, string] }) {
  const [agent, setAgent] = useState(initial[0]);
  const [epKey, setEpKey] = useState(initial[1]);
  useEpisode(player, epKey);
  return (
    <div className="mc-compare-col">
      <div className="mc-controls compact">
        <EpisodePicker agent={agent} setAgent={setAgent} epKey={epKey} setEpKey={setEpKey}
          agents={["Q-learning", "SARSA", "Dyna-Q (k=10)", "DQN", "Heuristic", "Random"]} />
      </div>
      <World player={player} agent={agent} small opts={{ labels: false, range: true, camera: "control" }} />
      <Goals player={player} />
    </div>
  );
}

function Compare({ speed, setSpeed }: { speed: number; setSpeed: (n: number) => void }) {
  const a = useMemo(() => new Player(fixedMap()), []);
  const b = useMemo(() => new Player(fixedMap()), []);
  useEffect(() => { a.speed = b.speed = speed; }, [speed, a, b]);
  return (
    <>
      <div className="mc-controls">
        <PlayControls players={[a, b]} speed={speed} setSpeed={setSpeed} />
        <p className="t-small muted">Same map, same goals. Pick an agent and episode on each side.</p>
      </div>
      <div className="mc-compare-grid">
        <CompareSide player={a} initial={["Q-learning", "Q-learning|train0"]} />
        <CompareSide player={b} initial={["Q-learning", "Q-learning|final0"]} />
      </div>
    </>
  );
}

const LIVE_MAPS: [string, () => LunarMap][] = [
  ["Landing zone (training map)", () => makeMap(DATA.map)],
  ...Object.values(DATA.episodes).filter((e) => e.map && e.agent === "DQN")
    .map((e, i) => [`Unseen hazard map ${i + 1}`, () => makeMap(e.map!)] as [string, () => LunarMap]),
];

function Live({ speed, setSpeed }: { speed: number; setSpeed: (n: number) => void }) {
  const player = useMemo(() => { const p = new Player(fixedMap()); p.loadLive(makeMap(DATA.map), 21); return p; }, []);
  const [mapIdx, setMapIdx] = useState(0);
  const [added, setAdded] = useState<AddedHazard[]>([]);
  const [note, setNote] = useState("Click the ground to drop a crater in the rover's way. Shift-click places a boulder. Click it again to remove it.");
  useP(player);
  useEffect(() => { player.speed = speed; }, [speed, player]);
  const load = (i: number) => { setMapIdx(i); setAdded([]); player.loadLive(LIVE_MAPS[i][1](), 21); };
  const restart = () => {
    for (const a of added) player.map.terrain[a.r][a.c] = REGOLITH;
    setAdded([]); player.reset();
  };
  const onCell = (r: number, c: number, shift: boolean) => {
    const m = player.map, t = m.terrain[r][c];
    const mine = added.find((a) => a.r === r && a.c === c);
    if (mine) {
      m.terrain[r][c] = REGOLITH;
      setAdded(added.filter((a) => a !== mine));
      setNote(`Removed the ${mine.kind} at row ${r}, column ${c}.`);
      player.replan();
      return;
    }
    const busy = t === VIKRAM || m.siteAt.has(r * 100 + c) || (player.state[0] === r && player.state[1] === c) ||
      (player.cur !== null && player.cur.next[0] === r && player.cur.next[1] === c);
    if (busy) { setNote("That cell holds the rover, Vikram or a science site. Pick another one."); return; }
    if (t === CRATER || t === BOULDER) { setNote("That hazard is part of the original map."); return; }
    const kind = shift ? "boulder" : "crater";
    m.terrain[r][c] = shift ? BOULDER : CRATER;
    setAdded([...added, { r, c, kind, t0: performance.now() }]);
    setNote(`New ${kind} at row ${r}, column ${c}. If it is inside the rover's camera square, the network's values change on the next step.`);
    player.replan();
  };
  const b = player.brain;
  const valid = b ? [...b.q].map((v, i) => ({ v, i })).filter((x) => b.mask[x.i]) : [];
  const lo = Math.min(...valid.map((x) => x.v)), hi = Math.max(...valid.map((x) => x.v));
  return (
    <>
      <div className="mc-controls">
        <label className="field"><span>Terrain</span>
          <select value={mapIdx} onChange={(e) => load(+e.target.value)}>
            {LIVE_MAPS.map(([n], i) => <option key={n} value={i}>{n}</option>)}
          </select>
        </label>
        <button className="btn-primary" onClick={() => (player.playing ? player.pause() : player.play())}>
          {player.playing ? "Pause" : player.done ? "Run again" : player.i > 0 ? "Resume" : "Launch rover"}
        </button>
        <button className="btn-ghost" onClick={restart}>Restart and clear craters</button>
        <label className="field speed"><span>Speed</span>
          <input type="range" min={1} max={12} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
        </label>
      </div>
      <div className="mc-main">
        <World player={player} agent="DQN" opts={{ labels: false, range: true, navcam: true, camera: "control", added, onCell }} />
        <aside className="mc-side">
          <p className="live-note">{note}</p>
          <MiniMap player={player} onCell={onCell} />
          <section className="mc-card brain">
            <h3>Q(s, a) right now</h3>
            <p className="t-small muted">Expected return after each action. The rover takes the top one (gold). It sees only the dashed 5 × 5 m camera square.</p>
            <ol>
              {valid.sort((x, y) => y.v - x.v).slice(0, 6).map(({ v, i }) => (
                <li key={i} className={i === b!.a ? "best" : ""}>
                  <span>{ACTION_LABEL[ACTIONS[i]]}</span>
                  <i style={{ width: `${8 + ((v - lo) / (hi - lo || 1)) * 92}%` }} />
                  <b className="tnum">{v.toFixed(0)}</b>
                </li>
              ))}
            </ol>
          </section>
          <Goals player={player} />
        </aside>
      </div>
    </>
  );
}

export default function MissionControl() {
  const [mode, setMode] = useState<Mode>("replay");
  const [speed, setSpeed] = useState(4);
  return (
    <section className="section panel mission-control" id="mission-control">
      <div className="wrap">
        <div className="mc-head">
          <h2 className="t-h2">Watch the agents drive</h2>
          <div className="mc-modes" role="tablist">
          {([["replay", "Replay an episode"], ["compare", "Compare two agents"], ["live", "Drop a crater (live)"]] as [Mode, string][]).map(([k, label]) => (
            <button key={k} role="tab" aria-selected={mode === k} className={mode === k ? "on" : ""} onClick={() => setMode(k)}>{label}</button>
          ))}
          </div>
        </div>
        {mode === "replay" && <Replay speed={speed} setSpeed={setSpeed} />}
        {mode === "compare" && <Compare speed={speed} setSpeed={setSpeed} />}
        {mode === "live" && <Live speed={speed} setSpeed={setSpeed} />}
        <p className="verify t-small muted">The browser simulator reproduces all {Object.values(DATA.episodes).reduce((n, e) => n + e.frames.length, 0).toLocaleString()} recorded Python steps and {DATA.dqn.checks.length} network decisions exactly, checked by the test suite.</p>
      </div>
    </section>
  );
}
