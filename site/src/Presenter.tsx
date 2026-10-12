import { useEffect, useState } from "react";
import "./presenter.css";

/**
 * Turns the page into a slide deck. Every element with [data-slide] is a slide.
 * - Arrow keys / PageUp / PageDown / Space / Home / End move between slides; F toggles fullscreen.
 * - A slide gets .in-view when most of it is on screen; CSS uses that to play its entrance.
 * - Elements matched by STAGGER get --i so their entrances cascade.
 */
const STAGGER = [
  ".panel-head > *", ".context > *", ".mdp-list > li", ".map-figure", ".steps > li", ".ledger tr", ".rl-lower .loop",
  ".algo-table tbody tr", ".observations > li", ".result", ".mc-head", ".mc-controls", ".mc-main", ".concl-col",
  ".end-grid > *", ".hero-copy > *", ".hero-facts",
].join(",");

export default function Presenter() {
  const [slides, setSlides] = useState<HTMLElement[]>([]);
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    const list = Array.from(document.querySelectorAll<HTMLElement>("[data-slide]"));
    setSlides(list);
    list.forEach((s) => s.querySelectorAll<HTMLElement>(STAGGER).forEach((el, i) => el.style.setProperty("--i", String(i))));
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.intersectionRatio > 0.45) {
          e.target.classList.add("in-view");
          setCurrent(list.indexOf(e.target as HTMLElement));
        }
      }
    }, { threshold: [0, 0.45, 0.8] });
    list.forEach((s) => io.observe(s));
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    const go = (i: number) => slides[Math.max(0, Math.min(slides.length - 1, i))]?.scrollIntoView({ behavior: "smooth" });
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, select, textarea")) return;
      if (e.key === " " && t.closest("button, a")) return;      // let Space press the focused button
      if (["ArrowDown", "ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); go(current + 1); }
      else if (["ArrowUp", "ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); go(current - 1); }
      else if (e.key === "Home") { e.preventDefault(); go(0); }
      else if (e.key === "End") { e.preventDefault(); go(slides.length - 1); }
      else if (e.key.toLowerCase() === "f") {
        if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [slides, current]);

  if (!slides.length) return null;
  const dark = slides[current]?.classList.contains("on-dark-slide");
  return (
    <nav className={`presenter ${dark ? "dark" : ""}`} aria-label="Slides">
      <ol>
        {slides.map((s, i) => (
          <li key={i}>
            <button aria-label={`Go to slide ${i + 1}: ${s.dataset.slide}`} aria-current={i === current}
              className={i === current ? "on" : ""} onClick={() => s.scrollIntoView({ behavior: "smooth" })}>
              <span>{s.dataset.slide}</span>
            </button>
          </li>
        ))}
      </ol>
      <p className="counter tnum">{current + 1} / {slides.length}</p>
    </nav>
  );
}
