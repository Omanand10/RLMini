import { useEffect, type ComponentType } from "react";
import Lenis from "lenis";
import Hero from "./sections/Hero";
import MissionControl from "./sections/MissionControl";
import { Algorithms, Mission, Night, Problem, Results } from "./sections/Story";

// The page is one lunar day: sunrise (hero, mission) to night (footer).
const SECTIONS: Record<string, ComponentType> = {
  hero: Hero, mission: Mission, problem: Problem, algorithms: Algorithms, results: Results,
  "mission-control": MissionControl, night: Night,
};

export default function App() {
  useEffect(() => {        // deep links like /#mission-control after React has rendered
    const id = location.hash.slice(1);
    if (id) requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView());
  }, []);
  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const lenis = new Lenis({ lerp: 0.12, anchors: true });
    let id = 0;
    const raf = (t: number) => { lenis.raf(t); id = requestAnimationFrame(raf); };
    id = requestAnimationFrame(raf);
    return () => { cancelAnimationFrame(id); lenis.destroy(); };
  }, []);

  // ?only=<section> renders one section alone (used for screenshots while developing)
  const only = new URLSearchParams(location.search).get("only");
  if (only && SECTIONS[only]) { const S = SECTIONS[only]; return <S />; }
  return (
    <>
      <Hero />
      <main>
        <Mission />
        <Problem />
        <Algorithms />
        <Results />
        <MissionControl />
      </main>
      <Night />
    </>
  );
}
