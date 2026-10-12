import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { LunarScene } from "../scene/LunarScene";
import { Player } from "../sim/player";
import { DATA } from "../sim/env";
import { buildSteps, fixedMap } from "../sim/episodes";
import "./hero.css";

function Loop({ player }: { player: Player }) {
  const idle = useRef(0);
  useFrame((_, dt) => {
    player.tick(Math.min(dt, 0.1));
    idle.current = player.done ? idle.current + dt : 0;
    if (idle.current > 4) player.play();       // loop the mission after a short pause
  });
  return null;
}

export default function Hero() {
  const player = useMemo(() => {
    const m = fixedMap(), p = new Player(m);
    p.loadSteps(m, buildSteps(DATA.episodes["Q-learning|final0"], m));
    p.speed = 2.2;
    return p;
  }, []);
  useEffect(() => { player.play(); }, [player]);
  const q = DATA.table["Q-learning"];
  return (
    <header className="hero">
      <div className="hero-canvas" aria-hidden>
        <Canvas shadows dpr={[1, 1.6]} camera={{ fov: 32, near: 0.1, far: 200 }}>
          <Loop player={player} />
          <LunarScene player={player} opts={{ camera: "hero" }} />
        </Canvas>
      </div>
      <div className="hero-shade" />
      <div className="hero-copy">
        <p className="hero-kicker">Chandrayaan-3 landing zone, lunar south pole, 69.36° S</p>
        <h1 className="t-display">Pragyan has one lunar day.</h1>
        <p className="t-lede">
          Fourteen Earth days of sunlight to sample the soil, send results home through Vikram and park safely before
          the cold. We trained reinforcement-learning agents to plan that day themselves.
        </p>
        <div className="hero-actions">
          <a className="btn-primary" href="#mission-control">Watch the agents drive</a>
          <a className="hero-link" href="#problem">How the problem is set up</a>
        </div>
      </div>
      <dl className="hero-facts">
        <div><dt>Samples relayed by the trained agent</dt><dd className="tnum">{q.sent_mean.toFixed(1)} of 4</dd></div>
        <div><dt>Crater losses, trained vs random</dt><dd className="tnum">{(q.crater_rate * 100).toFixed(0)}% vs {(DATA.table.Random.crater_rate * 100).toFixed(0)}%</dd></div>
        <div><dt>Algorithms compared</dt><dd className="tnum">7</dd></div>
      </dl>
    </header>
  );
}
