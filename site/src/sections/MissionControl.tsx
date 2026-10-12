import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { LunarScene, type SceneOptions } from "../scene/LunarScene";
import { Player } from "../sim/player";
import { ACTION_LABEL, AGENT_INFO, NARRATION, buildSteps, fixedMap, mapFor } from "../sim/episodes";
import { ACTIONS, BOULDER, CRATER, DATA, MAX_BATTERY, REGOLITH, SENT, HELD, VIKRAM, makeMap, type LunarMap } from "../sim/env";
import "./mission-control.css";

type Mode = "replay" | "compare" | "live";
const PHASES = ["Morning", "Noon", "Dusk"];
const REPLAY_AGENTS = ["Q-learning", "SARSA", "Dyna-Q (k=10)", "DQN", "Value Iteration", "Heuristic", "Random",
  "n-step SARSA (n=4)", "Monte Carlo"];

function useP(p: Player) { useSyncExternalStore(p.subscribe, p.getVersion); }

function Driver({ players }: { players: Player[] }) {
  useFrame((_, dt) => { for (const p of players) p.tick(Math.min(dt, 0.1)); });
  return null;
}

function World({ player, opts, className }: { player: Player; opts: SceneOptions; className?: string }) {
  return (
    <div className={`mc-world ${className ?? ""}`}>
      <Canvas shadows dpr={[1, 1.75]} camera={{ fov: 38, near: 0.1, far: 200 }} gl={{ antialias: true }}>
        <Driver players={[player]} />
        <LunarScene player={player} opts={opts} />
      </Canvas>
      <Narration player={player} />
      <Legend />
    </div>
  );
}

function Narration({ player }: { player: Player }) {
  useP(player);
  const s = player.cur;
  let text = "Ready at the landing site, next to Vikram.";
  let tone = "dim";
  if (s) { [text, tone] = NARRATION[s.ev] ?? [s.ev, "dim"]; if (s.ev === "move") text = `${ACTION_LABEL[ACTIONS[s.a]]}`; }
  else if (player.done) [text, tone] = outcomeText(player);
  else if (player.i > 0) text = "Waiting for the next command";
  return (
    <div className={`mc-caption tone-${tone}`} aria-live="polite">
      <span className="tnum">Step {player.i}</span>
      <p>{text}</p>
    </div>
  );
}

function Legend() {
  return (
    <ul className="mc-legend t-small">
      <li><i className="lg-site" />Science site (ring)</li>
      <li><i className="lg-path" />Rover's path</li>
      <li><i className="lg-range" />Vikram radio range</li>
      <li><i className="lg-shadow" />Permanent shadow</li>
    </ul>
  );
}

function outcomeText(p: Player): [string, string] {
  const sent = p.state[3].filter((s) => s === SENT).length;
  switch (p.endEv) {
    case "sleep_safe": return sent === 4 ? ["Mission complete: all four results reached Earth and the rover is asleep in sunlight.", "good"]
      : [`Asleep safely, but only ${sent} of 4 results reached Earth.`, "warn"];
    case "crater": return ["Lost in a crater.", "bad"];
    case "battery_dead": return ["The battery ran out.", "bad"];
    case "night": return ["Lunar night fell before the rover went to sleep.", "bad"];
    case "sleep_unsafe": return ["Slept in shadow or on low battery, so it will not wake up.", "bad"];
    default: return ["Ran out of time (200 steps).", "warn"];
  }
}

// ---- telemetry --------------------------------------------------------------------------------
function Telemetry({ player, compact }: { player: Player; compact?: boolean }) {
  useP(player);
  const [, , bat, sites, phase] = player.state;
  const m = player.map;
  return (
    <div className={`mc-tele ${compact ? "compact" : ""}`}>
      <div className="tele-row">
        <span className="muted">Battery</span>
        <span className="battery" aria-label={`${bat} of ${MAX_BATTERY}`}>
          {Array.from({ length: MAX_BATTERY }, (_, k) => <i key={k} className={k < bat ? (bat <= 2 ? "low" : "on") : ""} />)}
        </span>
      </div>
      <div className="tele-row">
        <span className="muted">Sun</span>
        <span className="phases">{PHASES.map((ph, k) => <b key={ph} className={k === phase ? "on" : k < phase ? "past" : ""}>{ph}</b>)}</span>
      </div>
      <div className="tele-row"><span className="muted">Score</span><b className="tnum">{player.score.toFixed(0)}</b></div>
      {!compact && <div className="tele-row"><span className="muted">Distance driven</span><b className="tnum">{player.track.length - 1} m</b></div>}
      <ul className="science">
        {m.sites.map(([, , typ], i) => (
          <li key={i} className={sites[i] === SENT ? "sent" : sites[i] === HELD ? "held" : ""}>
            <span>{typ === 0 ? "Sulphur site" : "Element site"} {i % 2 + 1}<em>{typ === 0 ? "LIBS" : "APXS"}</em></span>
            <b>{["Not sampled", "Sampled", "Sent to Earth"][sites[i]]}</b>
          </li>
        ))}
      </ul>
      {player.done && !compact && (
        <p className={`outcome tone-${outcomeText(player)[1]}`}>{outcomeText(player)[0]}</p>
      )}
    </div>
  );
}

function Transport({ players, speed, setSpeed, scrub }: { players: Player[]; speed: number; setSpeed: (n: number) => void; scrub?: Player }) {
  players.forEach(useP);
  const playing = players.some((p) => p.playing);
  const allDone = players.every((p) => p.done);
  return (
    <div className="mc-transport">
      <button className="btn-primary" onClick={() => players.forEach((p) => (playing ? p.pause() : p.play()))}>
        {playing ? "Pause" : allDone ? "Replay" : players[0].i > 0 ? "Resume" : "Play"}
      </button>
      <button className="btn-ghost" onClick={() => players.forEach((p) => p.reset())}>Restart</button>
      <label className="speed t-small">
        <span className="muted">Speed</span>
        <input type="range" min={1} max={12} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
        <span className="tnum">{speed}×</span>
      </label>
      {scrub && !scrub.live && (
        <input className="scrub" type="range" min={0} max={scrub.length} value={scrub.i} aria-label="Mission timeline"
          onChange={(e) => scrub.seek(+e.target.value)} />
      )}
    </div>
  );
}

// ---- episode picker ---------------------------------------------------------------------------
const stageOrder = ["untrained", "learning", "improving", "trained", "unseen"];
function episodesOf(agent: string) {
  return Object.entries(DATA.episodes).filter(([, e]) => e.agent === agent)
    .sort((a, b) => stageOrder.indexOf(a[1].stage) - stageOrder.indexOf(b[1].stage));
}

function Picker({ agent, setAgent, epKey, setEpKey, agents = REPLAY_AGENTS }: {
  agent: string; setAgent: (a: string) => void; epKey: string; setEpKey: (k: string) => void; agents?: string[];
}) {
  return (
    <div className="mc-picker">
      <div className="agents" role="radiogroup" aria-label="Agent">
        {agents.map((a) => (
          <button key={a} role="radio" aria-checked={a === agent} className={a === agent ? "on" : ""}
            onClick={() => { setAgent(a); setEpKey(episodesOf(a).find(([, e]) => e.stage === "trained")?.[0] ?? episodesOf(a)[0][0]); }}>
            {AGENT_INFO[a].short}
          </button>
        ))}
      </div>
      <p className="t-small muted agent-idea">{AGENT_INFO[agent].idea}</p>
      <select value={epKey} onChange={(e) => setEpKey(e.target.value)} aria-label="Mission">
        {episodesOf(agent).map(([k, e]) => <option key={k} value={k}>{e.label}</option>)}
      </select>
    </div>
  );
}

// ---- modes --------------------------------------------------------------------------------------
function Replay({ speed, setSpeed }: { speed: number; setSpeed: (n: number) => void }) {
  const player = useMemo(() => new Player(fixedMap()), []);
  const [agent, setAgent] = useState("Q-learning");
  const [epKey, setEpKey] = useState("Q-learning|final0");
  useEffect(() => {
    const E = DATA.episodes[epKey], m = mapFor(E);
    player.loadSteps(m, buildSteps(E, m));
  }, [epKey, player]);
  useEffect(() => { player.speed = speed; }, [speed, player]);
  return (
    <div className="mc-layout">
      <World player={player} opts={{ labels: true, range: true, camera: "control" }} />
      <aside className="mc-panel">
        <Picker agent={agent} setAgent={setAgent} epKey={epKey} setEpKey={setEpKey} />
        <Transport players={[player]} speed={speed} setSpeed={setSpeed} scrub={player} />
        <Telemetry player={player} />
      </aside>
    </div>
  );
}

function CompareSide({ player, initial }: { player: Player; initial: [string, string] }) {
  const [agent, setAgent] = useState(initial[0]);
  const [epKey, setEpKey] = useState(initial[1]);
  useEffect(() => { const E = DATA.episodes[epKey], m = mapFor(E); player.loadSteps(m, buildSteps(E, m)); }, [epKey, player]);
  return (
    <div className="mc-compare-col">
      <Picker agent={agent} setAgent={setAgent} epKey={epKey} setEpKey={setEpKey}
        agents={["Q-learning", "SARSA", "Dyna-Q (k=10)", "DQN", "Heuristic", "Random"]} />
      <World player={player} opts={{ labels: false, range: true, camera: "control" }} className="half" />
      <Telemetry player={player} compact />
    </div>
  );
}

function Compare({ speed, setSpeed }: { speed: number; setSpeed: (n: number) => void }) {
  const a = useMemo(() => new Player(fixedMap()), []);
  const b = useMemo(() => new Player(fixedMap()), []);
  useEffect(() => { a.speed = b.speed = speed; }, [speed, a, b]);
  return (
    <div className="mc-compare">
      <Transport players={[a, b]} speed={speed} setSpeed={setSpeed} />
      <div className="mc-compare-grid">
        <CompareSide player={a} initial={["Q-learning", "Q-learning|train0"]} />
        <CompareSide player={b} initial={["Q-learning", "Q-learning|final0"]} />
      </div>
    </div>
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
  const [mapKey, setMapKey] = useState(0);
  const [note, setNote] = useState("Click the ground to drop a crater. Shift-click places a boulder.");
  useP(player);
  useEffect(() => { player.speed = speed; }, [speed, player]);
  const load = (i: number) => { setMapIdx(i); player.loadLive(LIVE_MAPS[i][1](), 21); setMapKey((k) => k + 1); };
  const onCell = (r: number, c: number, shift: boolean) => {
    const m = player.map, t = m.terrain[r][c];
    const busy = t === VIKRAM || m.siteAt.has(r * 100 + c) || (player.state[0] === r && player.state[1] === c) ||
      (player.cur && player.cur.next[0] === r && player.cur.next[1] === c);
    if (busy) { setNote("That cell is taken by the rover, Vikram or a science site."); return; }
    const want = shift ? BOULDER : CRATER;
    m.terrain[r][c] = t === want ? REGOLITH : want;
    setNote(m.terrain[r][c] === REGOLITH ? `Cleared the hazard at row ${r}, column ${c}.`
      : `New ${want === CRATER ? "crater" : "boulder"} at row ${r}, column ${c}. The agent replans on its next step.`);
    setMapKey((k) => k + 1);
    player.replan();
  };
  const b = player.brain;
  const valid = b ? [...b.q].map((v, i) => ({ v, i })).filter((x) => b.mask[x.i]) : [];
  const lo = Math.min(...valid.map((x) => x.v)), hi = Math.max(...valid.map((x) => x.v));
  return (
    <div className="mc-layout">
      <World player={player} opts={{ labels: true, range: true, navcam: true, camera: "control", mapKey, onCell }} className="clickable" />
      <aside className="mc-panel">
        <select value={mapIdx} onChange={(e) => load(+e.target.value)} aria-label="Terrain">
          {LIVE_MAPS.map(([n], i) => <option key={n} value={i}>{n}</option>)}
        </select>
        <p className="t-small live-note">{note}</p>
        <Transport players={[player]} speed={speed} setSpeed={setSpeed} />
        <div className="brain">
          <h4>What the network is thinking</h4>
          <p className="t-small muted">How good the DQN expects each command to be from here (higher is better). It only sees the dashed 5×5 camera square around the rover. Commands that cannot run right now are hidden.</p>
          <ol>
            {valid.sort((x, y) => y.v - x.v).map(({ v, i }) => (
              <li key={i} className={i === b!.a ? "best" : ""}>
                <span>{ACTION_LABEL[ACTIONS[i]]}</span>
                <i style={{ width: `${8 + ((v - lo) / (hi - lo || 1)) * 92}%` }} />
              </li>
            ))}
          </ol>
        </div>
        <Telemetry player={player} />
      </aside>
    </div>
  );
}

export default function MissionControl() {
  const [mode, setMode] = useState<Mode>("replay");
  const [speed, setSpeed] = useState(4);
  return (
    <section className="section mission-control" id="mission-control">
      <div className="wrap">
        <div className="opener">
          <p className="clock"><i />Dusk: watch the agents work</p>
          <h2 className="t-h2">Mission control</h2>
          <p className="t-lede muted">Replay any agent's mission, put two of them side by side, or take the controls: drop a crater in front of the neural-network rover and watch it change course.</p>
        </div>
        <div className="mc-modes" role="tablist">
          {([["replay", "Replay a mission"], ["compare", "Compare two agents"], ["live", "Drop a crater (live)"]] as [Mode, string][]).map(([k, label]) => (
            <button key={k} role="tab" aria-selected={mode === k} className={mode === k ? "on" : ""} onClick={() => setMode(k)}>{label}</button>
          ))}
        </div>
        {mode === "replay" && <Replay speed={speed} setSpeed={setSpeed} />}
        {mode === "compare" && <Compare speed={speed} setSpeed={setSpeed} />}
        {mode === "live" && <Live speed={speed} setSpeed={setSpeed} />}
        <p className="verify t-small muted">The browser simulator replays all {Object.values(DATA.episodes).reduce((n, e) => n + e.frames.length, 0).toLocaleString()} recorded Python steps and {DATA.dqn.checks.length} network decisions exactly (checked by the test suite).</p>
      </div>
    </section>
  );
}
