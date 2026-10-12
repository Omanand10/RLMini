import { DATA } from "./sim/env";

/** Temporary landing page so the deployment pipeline exists; the 3D site replaces it. */
export default function App() {
  const q = DATA.table["Q-learning"];
  return (
    <main className="dark" style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: "var(--gutter)" }}>
      <div className="wrap">
        <p className="eyebrow">RL Lab mini project · Chandrayaan-3</p>
        <h1 className="display">Teaching Pragyan to plan its own lunar day.</h1>
        <p className="lede">
          Trained Q-learning agent: average mission score <span className="num">{q.return_mean.toFixed(1)}</span>,{" "}
          <span className="num">{q.sent_mean.toFixed(2)}</span> of 4 samples relayed. Full 3D mission simulator coming soon.
        </p>
      </div>
    </main>
  );
}
