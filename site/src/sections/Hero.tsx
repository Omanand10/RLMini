import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { View3D } from "../scene/View3D";
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
  return (
    <header className="hero">
      <View3D className="hero-canvas" fov={32}>
        <Loop player={player} />
        <LunarScene player={player} opts={{ camera: "hero" }} />
      </View3D>
      <div className="hero-shade" />
      <div className="hero-copy">
        <p className="hero-kicker">A reinforcement-learning project inspired by Chandrayaan-3's Pragyan rover</p>
        <h1 className="t-display">Teaching a lunar rover to plan its own day.</h1>
        <p className="t-lede">
          Fourteen Earth days of sunlight to sample the soil, send results home through Vikram and park safely before
          the cold. We trained reinforcement-learning agents to plan that day themselves.
        </p>
        <div className="hero-actions">
          <a className="btn-primary" href="#mission-control">Watch the agents drive</a>
          <a className="hero-link" href="#problem">See how the RL problem is set up</a>
        </div>
      </div>
      <dl className="hero-facts">
        <div><dt>Om Dwivedi</dt><dd className="tnum">16010123213</dd></div>
        <div><dt>Om Anand Jha</dt><dd className="tnum">16010123214</dd></div>
        <div><dt>Reinforcement Learning Lab, Semester VII</dt><dd>K. J. Somaiya School of Engineering</dd></div>
      </dl>
    </header>
  );
}
