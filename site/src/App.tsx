import { useEffect } from "react";
import Lenis from "lenis";
import Hero from "./sections/Hero";
import MissionControl from "./sections/MissionControl";

export default function App() {
  useEffect(() => {        // deep links like /#mission-control after React has rendered
    const id = location.hash.slice(1);
    if (id) requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView());
  }, []);
  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const lenis = new Lenis({ lerp: 0.12 });
    let id = 0;
    const raf = (t: number) => { lenis.raf(t); id = requestAnimationFrame(raf); };
    id = requestAnimationFrame(raf);
    return () => { cancelAnimationFrame(id); lenis.destroy(); };
  }, []);
  // ?only=mission-control renders one section alone (used for screenshots while developing)
  const only = new URLSearchParams(location.search).get("only");
  if (only === "mission-control") return <MissionControl />;
  return (
    <>
      <Hero />
      <main>
        <MissionControl />
      </main>
    </>
  );
}
