import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { useFrame } from "@react-three/fiber";
import { View3D } from "../scene/View3D";
import { LunarScene, type AddedHazard, type SceneOptions } from "../scene/LunarScene";
import { Player } from "../sim/player";
import { ACTION_LABEL, AGENT_INFO, NARRATION, buildSteps, fixedMap, mapFor } from "../sim/episodes";
import { ACTIONS, BOULDER, CRATER, DATA, MAX_BATTERY, REGOLITH, SENT, HELD, VIKRAM, makeMap, type LunarMap } from "../sim/env";
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
  const sites = player.state[3], m = player.map;
  const slept = player.done && player.endEv === "sleep_safe";
  return (
    <section className="mc-card">
      <h3>Goals for this episode</h3>
      <p className="t-small muted">Sample each site with the right instrument, send the results through Vikram, then sleep safely at dusk.</p>
      <ul className="goals">
        {m.sites.map(([, , typ], i) => (
          <li key={i}>
            <span>{typ === 0 ? "Sulphur" : "Element"} site {i % 2 + 1} <em>{typ === 0 ? "LIBS" : "APXS"}</em></span>
            <span className="ticks">
              <b className={sites[i] >= HELD ? "on" : ""}>Sampled</b>
              <b className={sites[i] === SENT ? "on" : ""}>Sent</b>
            </span>
          </li>
        ))}
        <li><span>Sleep safely at dusk</span><span className="ticks"><b className={slept ? "on" : ""}>Done</b></span></li>
      </ul>
    </section>
  );
}

function StatePanel({ player }: { player: Player }) {
  useP(player);
  const [r, c, bat, , phase] = player.state;
  return (
    <section className="mc-card">
      <h3>State sₜ</h3>
      <p className="t-small muted">What the agent knows when it chooses. Together with the site progress on the left, this is the full state.</p>
      <dl className="state">
        <div><dt>Position</dt><dd className="tnum">row {r}, column {c}</dd></div>
        <div><dt>Battery</dt><dd><span className="battery" aria-label={`${bat} of ${MAX_BATTERY}`}>
          {Array.from({ length: MAX_BATTERY }, (_, k) => <i key={k} className={k < bat ? (bat <= 2 ? "low" : "on") : ""} />)}
        </span></dd></div>
        <div><dt>Sun</dt><dd><span className="phases">{PHASES.map((ph, k) => <b key={ph} className={k === phase ? "on" : ""}>{ph}</b>)}</span></dd></div>
      </dl>
    </section>
  );
}

function PolicyPanel({ agent }: { agent: string }) {
  const info = AGENT_INFO[agent];
  return (
    <section className="mc-card">
      <h3>Policy π: {info.short}</h3>
      <p className="tags">{info.tags.map((t) => <span key={t}>{t}</span>)}</p>
      <p>{info.idea}</p>
      <p className="t-small muted">During replay the agent acts greedily: in each state it takes the action with the highest learned value.</p>
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
      <World player={player} agent={agent} opts={{ labels: true, range: true, camera: "control" }} />
      <div className="mc-cards">
        <Goals player={player} />
        <StatePanel player={player} />
        <PolicyPanel agent={agent} />
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
      <p className="live-note">{note}</p>
      <World player={player} agent="DQN" opts={{ labels: true, range: true, navcam: true, camera: "control", added, onCell }} />
      <div className="mc-cards">
        <Goals player={player} />
        <StatePanel player={player} />
        <section className="mc-card brain">
          <h3>Q(s, a): the network's choice</h3>
          <p className="t-small muted">Expected total reward still to come after each action. The rover takes the highest (gold). It sees only the dashed 5 × 5 m square around it.</p>
          <ol>
            {valid.sort((x, y) => y.v - x.v).slice(0, 8).map(({ v, i }) => (
              <li key={i} className={i === b!.a ? "best" : ""}>
                <span>{ACTION_LABEL[ACTIONS[i]]}</span>
                <i style={{ width: `${8 + ((v - lo) / (hi - lo || 1)) * 92}%` }} />
                <b className="tnum">{v.toFixed(0)}</b>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </>
  );
}

export default function MissionControl() {
  const [mode, setMode] = useState<Mode>("replay");
  const [speed, setSpeed] = useState(4);
  return (
    <section className="section mission-control" id="mission-control">
      <div className="wrap">
        <div className="opener">
          <h2 className="t-h2">Watch the agents drive</h2>
          <p className="t-lede muted">Each run is one episode: one lunar day. Press play and follow the line above the map. It shows the action the agent picked and the reward it got for it.</p>
        </div>
        <div className="mc-modes" role="tablist">
          {([["replay", "Replay an episode"], ["compare", "Compare two agents"], ["live", "Drop a crater (live)"]] as [Mode, string][]).map(([k, label]) => (
            <button key={k} role="tab" aria-selected={mode === k} className={mode === k ? "on" : ""} onClick={() => setMode(k)}>{label}</button>
          ))}
        </div>
        {mode === "replay" && <Replay speed={speed} setSpeed={setSpeed} />}
        {mode === "compare" && <Compare speed={speed} setSpeed={setSpeed} />}
        {mode === "live" && <Live speed={speed} setSpeed={setSpeed} />}
        <p className="verify t-small muted">The browser simulator reproduces all {Object.values(DATA.episodes).reduce((n, e) => n + e.frames.length, 0).toLocaleString()} recorded Python steps and {DATA.dqn.checks.length} network decisions exactly, checked by the test suite.</p>
      </div>
    </section>
  );
}
