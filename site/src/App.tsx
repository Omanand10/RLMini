import { useEffect, type ComponentType } from "react";
import Hero from "./sections/Hero";
import MissionControl from "./sections/MissionControl";
import { Algorithms, Learning, Mission, Night, RLProblem, Results } from "./sections/Story";

const SECTIONS: Record<string, ComponentType> = {
  hero: Hero, mission: Mission, problem: RLProblem, learning: Learning, "mission-control": MissionControl,
  algorithms: Algorithms, results: Results, night: Night,
};

export default function App() {
  useEffect(() => {        // deep links like /#mission-control after React has rendered
    const id = location.hash.slice(1);
    if (id) requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView());
  }, []);

  // ?only=<section> renders one section alone (used for screenshots while developing)
  const only = new URLSearchParams(location.search).get("only");
  if (only && SECTIONS[only]) { const S = SECTIONS[only]; return <S />; }
  return (
    <>
      <Hero />
      <main>
        <Mission />
        <RLProblem />
        <Learning />
        <MissionControl />
        <Algorithms />
        <Results />
      </main>
      <Night />
    </>
  );
}
