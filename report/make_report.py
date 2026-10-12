"""Build report/Pragyan_RL_Report.docx in the LAB CA mini-project template format.

Numbers come from rl/results/metrics.json (and rl/results/run.pkl for convergence
statistics), figures from rl/results/plots/. Run after train.py and plots.py.
"""
import json
import os
import pickle

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "rl", "results")
PLOTS = os.path.join(RESULTS, "plots")
ASSETS = os.path.join(HERE, "assets")
RED = RGBColor(0xBC, 0x20, 0x2E)
FONT = "Times New Roman"
REPO = "https://github.com/Omanand10/RLMini"
SITE = "https://pragyan-rl.vercel.app"

M = json.load(open(os.path.join(RESULTS, "metrics.json")))
T = M["fixed_map"]
G = M["dqn_generalization"]
SW = M["sweeps"]
VI = M["value_iteration"]
HP = M["dqn_hparams"]


def pct(x):
    return f"{100 * x:.1f}%"


def conv_stats():
    """Episodes to reach 90% of final smoothed return, from the training logs."""
    path = os.path.join(RESULTS, "run.pkl")
    if not os.path.exists(path):
        return {}
    run = pickle.load(open(path, "rb"))
    out = {}
    for r in run["results"]:
        if r["name"] in ("Q-learning", "SARSA", "n-step SARSA (n=4)", "Monte Carlo", "Dyna-Q (k=10)"):
            y = r["log"]["returns"].astype(float)
            w = max(50, len(y) // 100)
            c = np.cumsum(np.insert(y, 0, 0.0)); s = (c[w:] - c[:-w]) / w
            out[r["name"]] = {"start": s[0], "end": s[-1], "ep90": int(np.argmax(s >= s[0] + .9 * (s[-1] - s[0]))) + w,
                              "episodes": len(y)}
        if r["name"] == "DQN":
            out["DQN"] = {"episodes": len(r["log"]["returns"])}
    return out


C = conv_stats()

# ---------------------------------------------------------------------------------------------
doc = Document()
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15

sec = doc.sections[0]
sec.page_height, sec.page_width = Inches(11.69), Inches(8.27)      # A4, like the template
sec.left_margin = sec.right_margin = Inches(1)
sec.top_margin, sec.bottom_margin = Inches(1.95), Inches(0.9)
sec.header_distance = Inches(0.45)


def run(p, text, bold=False, italic=False, color=None, size=None, font=None):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.name = font or FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), font or FONT)
    if color is not None:
        r.font.color.rgb = color
    if size:
        r.font.size = Pt(size)
    return r


# ---- header on every page: logos, school name, department in red ---------------------------
hdr = sec.header
hp = hdr.paragraphs[0]
hp.paragraph_format.space_after = Pt(0)
hp.add_run().add_picture(os.path.join(ASSETS, "logo_11.png"), width=Inches(1.62))
hp2 = hdr.add_paragraph(); hp2.paragraph_format.space_after = Pt(0)
hp2.add_run().add_picture(os.path.join(ASSETS, "logo_12.png"), width=Inches(0.55))
hp3 = hdr.add_paragraph(); hp3.paragraph_format.space_after = Pt(0)
hp3.paragraph_format.left_indent = Inches(-0.5)
run(hp3, "K. J. Somaiya School of Engineering, Mumbai-77", bold=True)
hp4 = hdr.add_paragraph(); hp4.alignment = WD_ALIGN_PARAGRAPH.CENTER
run(hp4, "Department of Computer Engineering", bold=True, color=RED)


def para(text="", bold=False, align=None, italic=False, after=6, indent=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align is None else align
    p.paragraph_format.space_after = Pt(after)
    if indent is not None:
        p.paragraph_format.left_indent = Inches(indent)
    # **bold** segments inside text
    parts = text.split("**")
    for i, part in enumerate(parts):
        if part:
            run(p, part, bold=bold or i % 2 == 1, italic=italic)
    return p


def heading(text, size=12, before=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run(p, text, bold=True, size=size)
    return p


def bullets(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        parts = it.split("**")
        for i, part in enumerate(parts):
            if part:
                run(p, part, bold=i % 2 == 1)


def borders(t):
    """Explicit single-line borders (some viewers ignore the Table Grid style)."""
    tblPr = t._element.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)
    t.autofit = False


def set_cell_bg(cell, hexcolor):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def table(rows, header=True, widths=None, size=10.5, bold_first_col=False):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            parts = str(val).split("**")
            for k, part in enumerate(parts):
                if part:
                    run(p, part, bold=(header and i == 0) or (bold_first_col and j == 0) or k % 2 == 1, size=size)
            if header and i == 0:
                set_cell_bg(cell, "E7E6E6")
            if widths:
                cell.width = Inches(widths[j])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


FIG = [0]


def figure(name, caption, width=6.2):
    FIG[0] += 1
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(os.path.join(PLOTS, name), width=Inches(width))
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(c, f"Figure {FIG[0]}: ", bold=True, size=10); run(c, caption, italic=True, size=10)
    return FIG[0]


def placeholder(text):
    t = doc.add_table(rows=1, cols=1); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    cell = t.cell(0, 0); set_cell_bg(cell, "FFF4E5")
    p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(p, text, italic=True, size=10.5, color=RGBColor(0x8A, 0x4B, 0x00))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def code(text):
    t = doc.add_table(rows=1, cols=1); t.style = "Table Grid"
    borders(t)
    cell = t.cell(0, 0); set_cell_bg(cell, "F4F4F4")
    p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
    run(p, text, font="Courier New", size=8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ---- page 1: title block exactly as the template ---------------------------------------------
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run(p, "LAB CA", bold=True)
info = doc.add_table(rows=3, cols=4); info.style = "Table Grid"; info.alignment = WD_TABLE_ALIGNMENT.CENTER
borders(info)
cells = [("Course Name:", "Reinforcement Learning", "Semester:", "VII"),
         ("Date of Performance:", "___ / ___ / ______", "DIV/ Batch No:", ""),
         ("Student Name:", "Om Dwivedi\nOm Anand Jha", "Roll No:", "16010123213\n16010123214")]
for i, row in enumerate(cells):
    for j, val in enumerate(row):
        c = info.cell(i, j); c.text = ""
        pp = c.paragraphs[0]; pp.paragraph_format.space_before = Pt(3); pp.paragraph_format.space_after = Pt(3)
        run(pp, val, bold=True, color=RED if j % 2 == 0 else None)
        c.width = Inches([1.85, 2.2, 1.35, 1.3][j])
doc.add_paragraph()
para("Reinforcement Learning Lab CA – Mini Project Report", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)

p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(6)
run(p, "1. Title: ", bold=True)
run(p, "Project Pragyan – A Reinforcement Learning Mission Planner for the Chandrayaan-3 Lunar Rover", bold=True)

# ================================== 2. PROBLEM DEFINITION ======================================
heading("2. Problem Definition & RL Formulation")
heading("2.1 Problem Statement", before=4)
para("On 23 August 2023, ISRO's Chandrayaan-3 mission soft-landed the Vikram lander at the Shiv Shakti point "
     "(69.36° S, 32.34° E), about 600 km from the lunar south pole, and deployed the six-wheeled Pragyan rover. "
     "Pragyan had a single lunar day (about 14 Earth days) of solar power. In that window it had to drive across "
     "unknown, cratered regolith, take measurements with two instruments – the Laser-Induced Breakdown "
     "Spectroscope (LIBS), which confirmed sulphur near the south pole, and the Alpha Particle X-ray Spectrometer "
     "(APXS) – and send its data to Earth through Vikram, its only radio link. It covered a little over 100 m. On "
     "27 August a 4 m wide crater appeared 3 m ahead and the rover had to be commanded to retrace its path. Before "
     "lunar night it was parked in sunlight, its payloads were switched off and it was put to sleep.")
para("This project treats that day of operations as a sequential decision problem. **Given a map of the landing "
     "zone, a limited and stochastic battery, a moving sun and wheels that slip on loose regolith, which sequence "
     "of drive, science, communication, charging and sleep commands collects and relays the most science while "
     "keeping the rover alive?** We build a simulator inspired by (not a replica of) the Pragyan landing zone, "
     "train reinforcement-learning agents to act as the rover's mission planner, and compare them with an optimal "
     "dynamic-programming planner, a hand-written rule-based planner and a random policy.")

heading("2.2 Motivation", before=4)
para("Planetary rovers cannot be driven in real time: every command is planned on Earth, uplinked, executed and "
     "checked, so a mistake (a wrong turn into a crater, a battery drained in shadow, a missed sleep window) can end "
     "a mission that cost years to build. More onboard autonomy means more science per day and fewer such risks. "
     "NASA's Curiosity rover already selects some of its own laser targets onboard (the AEGIS system), and the next "
     "generation of lunar and Martian rovers is expected to plan more of its own day.")
para("Reinforcement learning fits this problem for four reasons:")
bullets([
    "**Sequential decisions with delayed reward.** Driving towards a site earns nothing immediately; the reward "
    "arrives only after sampling and, later, transmitting from inside Vikram's radio range. RL optimises long-term "
    "return, not the next step.",
    "**Uncertainty.** Wheel slip, random battery drain and an uncertain sun schedule make the outcome of every "
    "command stochastic. A fixed script cannot react; a learned policy maps every state, including unexpected "
    "ones, to an action.",
    "**Competing objectives.** Science, battery, safety and time pull in different directions. Writing these "
    "trade-offs by hand is hard; with RL we only specify a reward and the agent discovers the trade-off.",
    "**Unknown or changing terrain.** The real rover discovered its crater only when the NavCam saw it. A "
    "model-free agent that learns from local observations (our DQN) can replan when the map changes, which a "
    "planner that needs the full model in advance cannot do.",
])

heading("2.3 Objectives", before=4)
bullets([
    "Model a Pragyan-like lunar mission as a Markov Decision Process with a 13-action command set, and implement "
    "it as a custom Gymnasium environment with an exact, enumerable transition model.",
    "Implement and train the syllabus algorithms – Value Iteration and Monte Carlo control (Module 2), Q-learning "
    "and SARSA (Module 3), n-step SARSA and Dyna-Q (Module 4), and a Deep Q-Network with experience replay "
    "(Module 5) – on the same environment.",
    "Evaluate every learned policy quantitatively (return, success rate, safety, convergence speed) against the "
    "optimal planner and two baselines, and study the effect of the main hyper-parameters.",
    "Demonstrate the trained agents in an interactive lunar simulation, including a live mode in which new craters "
    "can be dropped in front of the rover while the DQN agent drives.",
], style="List Number")

heading("2.4 RL Problem Formulation", before=4)
table([
    ["RL Component", "Description"],
    ["Environment", "PragyanEnv, a custom Gymnasium environment: a 12 × 12 grid of the landing zone (1 cell ≈ 1 m) "
                    "with regolith, slopes (extra energy), boulders (impassable), craters (rover lost), a permanently "
                    "shadowed region (no solar charging), the Vikram lander with a radio range of 3 cells, and four "
                    "science sites (2 LIBS, 2 APXS). Moves slip 45° left or right with probability 0.10, each move "
                    "drains one battery unit with probability 0.15 (0.50 on slopes) and the sun advances from morning "
                    "to noon to dusk to night with probability 0.025 per step."],
    ["State (S)", "Tabular agents: s = (row, column, battery level 0–5, status of each of the 4 sites ∈ {not "
                  f"sampled, sampled-and-held, relayed}}, sun phase ∈ {{morning, noon, dusk}}). 209,952 encodable "
                  f"states, {VI['n_reachable_states']:,} reachable from the landing point. DQN: a 130-dimensional "
                  "feature vector – the 5 × 5 NavCam window around the rover (4 terrain channels), battery, sun "
                  "phase, each site's status and relative position, the relative position of Vikram, an in-range "
                  "flag and the terrain under the rover."],
    ["Action (A)", "13 discrete commands: 8 drive directions (N, NE, E, SE, S, SW, W, NW), LIBS, APXS, TRANSMIT, "
                   "RECHARGE, SLEEP. Invalid commands (driving into a boulder, firing an instrument off-site, "
                   "transmitting with no data or out of range, charging in shadow or at full charge, sleeping before "
                   "dusk) are masked out, like an onboard command filter. Craters are not masked – avoiding them "
                   "must be learned."],
    ["Reward (R)", "−1 per step (−2 on a slope, −2 when blocked); +10 for a correct sample; −2 for a wasted "
                   "instrument or transmit; +25 per sample relayed to Earth; +30 for a safe sleep (dusk, sunlight, "
                   "battery ≥ 3) plus a +50 mission bonus if all four samples were relayed; −50 for an unsafe sleep; "
                   "−30 if night falls while the rover is awake; −100 if the rover falls into a crater or its battery "
                   "dies."],
    ["Policy (π)", "During training an ε-greedy policy over Q(s, a) restricted to valid actions (ε decays from 1.0 "
                   "to 0.05). The deployed policy is greedy: π(s) = argmaxₐ Q(s, a). For DQN, Q(s, a) is a neural "
                   "network."],
    ["Discount Factor (γ)", "γ = 0.99, because rewards for relaying data and sleeping safely arrive 50–100 steps "
                            "after the decisions that earn them."],
    ["Episode/Termination Condition", "An episode is one lunar day. It ends when the rover sleeps (safely or "
                                      "not), falls into a crater, runs out of battery, is caught awake by lunar "
                                      "night, or after 200 steps (truncation)."],
], widths=[1.7, 4.6], size=10.5, bold_first_col=True)
para("A mission counts as a **success** only if all four samples are relayed to Earth and the rover then sleeps "
     "safely at dusk.", after=4)

heading("2.5 Selected RL Algorithm", before=4)
bullets([
    "**Name of algorithm:** Q-learning (off-policy temporal-difference control) is the primary agent for the known "
    "landing zone, and a Deep Q-Network (Double DQN with 3-step returns) is the agent for terrain that changes. "
    "SARSA, n-step SARSA, Dyna-Q and Monte Carlo control were trained on the same environment for comparison, and "
    "Value Iteration gives the optimal reference.",
    "**Reason for selecting the algorithm:** The state space is discrete and enumerable on a fixed map, so a "
    "tabular Q-function is exact and easy to inspect. Q-learning is model-free (it does not need the slip or "
    "battery probabilities that a real rover team would not know exactly) and off-policy (it learns the greedy "
    "policy while exploring). In our comparison it gave the highest return of all learning agents "
    f"({T['Q-learning']['return_mean']:.1f}). When the terrain changes, a table indexed by the exact cell no "
    "longer generalises, so we use DQN, which reads a local camera window and can carry what it learned to maps "
    "it has never seen.",
    "**Brief description of the algorithm:** Q-learning keeps an estimate Q(s, a) of the expected discounted "
    "return of taking action a in state s and acting optimally afterwards. After every step (s, a, r, s′) it moves "
    "the estimate towards the Bellman optimality target: Q(s, a) ← Q(s, a) + α [r + γ maxₐ′ Q(s′, a′) − Q(s, a)]. "
    "Actions are chosen ε-greedily, so the agent explores early and exploits later. With enough visits to every "
    "state–action pair and a suitable step size, Q converges to the optimal Q*. DQN replaces the table with a "
    "neural network Q(s, a; θ) trained on mini-batches drawn from an experience-replay buffer, against targets "
    "computed by a periodically copied target network.",
])

# ================================== 3. IMPLEMENTATION =========================================
heading("3. Implementation & Working of RL Agent")
heading("3.1 System Architecture / Workflow", before=4)
para("Figure 1 shows the agent–environment loop: the environment produces a state, the agent selects a valid "
     "action ε-greedily, the environment samples the outcome and returns a reward and the next state, and the "
     "agent updates its Q-table or Q-network. Figure 2 shows the landing-zone map used for training and evaluation.")
figure("architecture.png", "Environment → State → RL Agent → Action → Reward → State update → Learning.", 6.2)
figure("environment_map.png", "Lunar landing-zone environment (12 × 12, 1 cell ≈ 1 m).", 4.9)

heading("3.2 Tools and Technologies", before=4)
bullets([
    "**Programming language:** Python 3.9 for the environment, training and analysis; JavaScript (HTML5 Canvas) "
    "for the simulation demo.",
    "**Libraries/frameworks:** Gymnasium 1.1 (environment API), NumPy 2.0 (tabular agents, Value Iteration), "
    "PyTorch 2.8 (DQN), Matplotlib 3.9 (graphs), python-docx (this report).",
    "**Development environment:** macOS on Apple Silicon (10 CPU cores, no GPU), VS Code / terminal, Git and "
    "GitHub for version control; training runs in parallel with Python's concurrent.futures.",
    "**Dataset/simulator:** No external dataset. The custom PragyanEnv simulator generates all experience. Its "
    "layout and physics are inspired by the published Chandrayaan-3 mission facts (two instruments, relay through "
    "Vikram, a ~100 m traverse, one lunar day, a crater detour, sleep before night).",
])

heading("3.3 Algorithm Implementation", before=4)
para("The implementation (repository linked in the Appendix) has four parts:")
bullets([
    "**Environment (pragyan/env.py).** All dynamics live in one function, outcomes(state, action), which returns "
    "the complete list of (probability, next state, reward, terminal, event). step() samples from this list, and "
    "Value Iteration reads it directly, so the planner's model is exactly the environment. A valid-action mask, a "
    "state-index encoder for tabular methods, a feature encoder for DQN, and generators for perturbed and fully "
    "random maps are also provided. A built-in self-check asserts the key rules (craters terminate, transmit "
    "needs range and data, no charging in shadow, probabilities sum to 1).",
    "**Tabular agents (pragyan/tabular.py).** Q-learning, SARSA, n-step SARSA, every-visit Monte Carlo control and "
    "Dyna-Q share one Q-table layout (209,952 × 13) in which invalid actions are pre-set to −10⁹, so every max and "
    "argmax – including bootstrapped targets and Dyna's planning updates – only considers valid actions. Value "
    f"Iteration enumerates the {VI['n_reachable_states']:,} reachable states by breadth-first search and runs "
    "vectorised Bellman backups until the largest change is below 10⁻⁴. A random policy and a rule-based "
    "heuristic (recharge when low, sample on a site, transmit in range, otherwise follow the shortest path to the "
    "nearest site using the full map, sleep at dusk) are the baselines.",
    "**Deep Q-Network (pragyan/dqn.py).** A 130-128-128-13 multilayer perceptron with ReLU activations, a "
    "100,000-transition replay buffer, a target network synchronised every 1,000 steps, Double-DQN targets, 3-step "
    "returns, Huber loss and gradient clipping. It trains on perturbed landing zones (2–4 new craters and boulders "
    "per episode, sites moved by up to one cell), so it has to rely on its NavCam window rather than memorise the "
    "map. Potential-based reward shaping (Ng et al., 1999) with Φ(s) = 12 × progress − distance-to-next-goal "
    "speeds up learning without changing the optimal policy.",
    "**Experiments and outputs (train.py, plots.py, demo/).** train.py runs all agents and the hyper-parameter "
    "sweeps in parallel, then trains DQN alone on all cores. It evaluates every frozen greedy policy on 500 "
    "missions (5 seeds × 100) and writes the metrics, the figures' data and the demo data. The browser demo "
    "contains a JavaScript port of the environment and of the DQN forward pass. On start-up it checks itself "
    "against Python: all 3,656 recorded transitions and all 88 recorded DQN decisions are reproduced exactly.",
])

heading("3.4 Hyperparameters / Parameter Settings", before=4)
table([
    ["Parameter", "Value"],
    ["Learning Rate (α)", "Tabular TD methods: 0.1 (sweep: 0.05, 0.1, 0.2, 0.5). Monte Carlo: 0.05 (constant "
                          f"step size). DQN: Adam, learning rate {HP['lr']}"],
    ["Discount Factor (γ)", "0.99 for all agents (sweep: 0.90, 0.95, 0.99)"],
    ["Exploration Rate (ε)", "ε-greedy over valid actions. Tabular: exponential decay 1.0 → 0.05 over the first 80% "
                             "of episodes (sweep: 30%, 50%, 80%). DQN: linear decay 1.0 → 0.05 over the first 40% "
                             "of steps"],
    ["Number of Episodes", "Q-learning, SARSA, n-step SARSA, Monte Carlo: 300,000 episodes each. Dyna-Q: 60,000 "
                           f"episodes (similar wall-clock time). DQN: 2,000,000 environment steps "
                           f"({C.get('DQN', {}).get('episodes', 0):,} episodes)"],
    ["Other parameters", "n-step SARSA: n = 4 (sweep 1, 2, 4, 8). Dyna-Q: k = 10 planning updates per real step "
                         "(sweep 0, 5, 10, 25), sample model of the last 32 outcomes per (s, a). DQN: replay 100k, "
                         f"batch {HP['batch']}, warm-up {HP['warmup']:,} steps, update every {HP['train_every']} "
                         f"steps, target sync every {HP['target_every']:,} steps, {HP['nstep']}-step Double-DQN "
                         "targets, Huber loss, gradient clip 10, 2 × 128 hidden units. Value Iteration: tolerance "
                         "10⁻⁴. Evaluation: 5 seeds × 100 missions, greedy policy"],
], widths=[1.8, 4.5], size=10.5, bold_first_col=True)

heading("3.5 Training Procedure", before=4)
para("Every agent interacts with the environment episode by episode. An episode starts with the rover next to "
     "Vikram at full battery in the lunar morning and ends at a terminal event. For each step the agent observes "
     "the state, chooses a valid action ε-greedily, and receives the reward and the next state sampled from "
     "outcomes(). The value function is updated as follows:")
bullets([
    "**Q-learning:** Q(s,a) ← Q(s,a) + α[r + γ maxₐ′ Q(s′,a′) − Q(s,a)] after every step (off-policy target).",
    "**SARSA:** Q(s,a) ← Q(s,a) + α[r + γ Q(s′,a′) − Q(s,a)], where a′ is the action the ε-greedy policy actually "
    "takes next (on-policy target).",
    "**n-step SARSA:** the target is the n-step return Gₜ:ₜ₊ₙ = rₜ₊₁ + γrₜ₊₂ + … + γⁿ⁻¹rₜ₊ₙ + γⁿQ(sₜ₊ₙ, aₜ₊ₙ).",
    "**Monte Carlo control:** at the end of each episode every visited (s, a) moves towards the complete "
    "discounted return Gₜ.",
    "**Dyna-Q:** a Q-learning update on the real step, then k extra Q-learning updates on transitions replayed "
    "from the learned model (planning).",
    "**DQN:** each transition is stored in the replay buffer; every second step a random mini-batch of 128 "
    "minimises the Huber loss between Q(s,a;θ) and y = Rₜ⁽³⁾ + γ³ Q(s′, argmaxₐ′ Q(s′,a′;θ); θ⁻), and θ⁻ is "
    "copied from θ every 1,000 steps.",
    "**Value Iteration (planning, not learning):** V(s) ← maxₐ Σ p(s′,r|s,a)[r + γV(s′)] until convergence "
    f"({VI['sweeps']} sweeps, {VI['plan_time']:.0f} s).",
])
para("Exploration starts fully random (ε = 1) so that every reachable part of the map is tried, and decays so that "
     "the agent increasingly exploits what it has learned. After training, the Q-table or network is frozen and "
     "the greedy policy is evaluated on 500 fresh missions that were never used for training. Training curves "
     "(return, success, steps and samples relayed per episode) are logged for every episode.")

heading("3.6 Working Demonstration", before=4)
para("The trained agents are demonstrated in an interactive 3D lunar simulation on the project website "
     "(https://pragyan-rl.vercel.app, built with React and three.js; it also runs offline from the repository). The site walks through the mission, the MDP and the results, and ends in a 3D Mission Control with three "
     "modes: **Replay** (any agent at episode 1, mid-training or fully trained, with a timeline, narration of each "
     "command, battery, sun phase and science status), **Compare** (two agents side by side on the same map) and "
     "**Drop a crater** (the DQN drives in real time in the browser; clicking the terrain adds a crater or boulder, "
     "and the panel shows how good the network rates each command). In 200 scripted tests in which a crater was dropped on the exact cell the DQN rover was about to enter, it "
     "changed its command every time and avoided the new crater in 90% of cases.")
placeholder("[Screenshots of the demo to be inserted here: Mission Replay (trained Q-learning mission complete), "
            "Head-to-Head (untrained vs trained), Live crater-drop with the Q-value panel, Scoreboard]")

# ================================== 4. RESULTS =================================================
heading("4. Results, Analysis & Viva")
heading("4.1 Experimental Results", before=4)
para("Table 1 reports the frozen greedy policy of every agent on the fixed landing zone (500 missions: 5 "
     "evaluation seeds × 100). ± is the standard deviation of the mean return across the 5 evaluation seeds of one "
     "training run.")
order = ["Value Iteration", "Q-learning", "SARSA", "Dyna-Q (k=10)", "n-step SARSA (n=4)", "Monte Carlo", "DQN",
         "Heuristic", "Random"]
rows = [["Agent", "Avg return", "Success", "Samples relayed (of 4)", "Crater", "Battery dead", "Night", "Avg steps"]]
for n in order:
    r = T[n]
    label = {"Value Iteration": "Value Iteration (optimal)", "Heuristic": "Heuristic (rule-based)",
             "DQN": "DQN (trained on perturbed maps)"}.get(n, n)
    rows.append([label, f"{r['return_mean']:.1f} ± {r['return_std_seeds']:.1f}", pct(r["success_rate"]),
                 f"{r['sent_mean']:.2f}", pct(r["crater_rate"]), pct(r["battery_rate"]), pct(r["night_rate"]),
                 f"{r['steps_mean']:.0f}"])
table(rows, widths=[1.75, 0.95, 0.65, 0.8, 0.6, 0.65, 0.55, 0.55], size=9)
p = doc.paragraphs[-1]
c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
run(c, "Table 1: ", bold=True, size=10); run(c, "Evaluation of frozen greedy policies on the landing zone.", italic=True, size=10)

para("Table 2 tests generalisation: the same policies on landing zones with new hazards and on fully random lunar "
     "maps that none of the agents trained on.")
g_rows = [["Agent", "Landing zone", "Unseen hazards (perturbed)", "Fully random maps"]]
for n in ("DQN", "Heuristic", "Random"):
    g_rows.append([n] + [f"{G[n][e]['return_mean']:.1f} / {pct(G[n][e]['success_rate'])}"
                         for e in ("fixed", "perturbed", "random")])
table(g_rows, widths=[1.4, 1.5, 1.8, 1.6], size=10)
c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
run(c, "Table 2: ", bold=True, size=10); run(c, "Average return / success rate on maps the agents did not train on.", italic=True, size=10)

heading("4.2 Performance Metrics", before=4)
q, vi, he, rn = T["Q-learning"], T["Value Iteration"], T["Heuristic"], T["Random"]
bullets([
    f"**Cumulative reward:** sum of rewards in a mission. Trained Q-learning earns {q['return_mean']:.1f} per "
    f"mission against {rn['return_mean']:.1f} for the random policy, an improvement of "
    f"{q['return_mean'] - rn['return_mean']:.0f} points.",
    f"**Average reward:** mean over 500 evaluation missions (Table 1). Q-learning reaches "
    f"{100 * q['return_mean'] / vi['return_mean']:.0f}% of the optimal Value-Iteration return "
    f"({vi['return_mean']:.1f}).",
    f"**Success rate:** all four samples relayed and a safe sleep: Q-learning {pct(q['success_rate'])}, Value "
    f"Iteration {pct(vi['success_rate'])}, heuristic {pct(he['success_rate'])}, random 0%.",
    "**Number of steps:** trained agents use 75–77 steps per mission. The untrained agents end after about 20 "
    "steps, almost always in a crater.",
    "**Convergence:** episodes needed to reach 90% of the final training return (Figure 9) – Dyna-Q "
    f"{C.get('Dyna-Q (k=10)', {}).get('ep90', 0):,}, Q-learning {C.get('Q-learning', {}).get('ep90', 0):,}, SARSA "
    f"{C.get('SARSA', {}).get('ep90', 0):,}. Value Iteration converges in {VI['sweeps']} sweeps (Figure 10).",
    "**Safety (failure modes):** crater, battery-death and night rates (Table 1, Figure 6) – the most important "
    "metric for a real rover.",
    f"**Accuracy/performance improvement:** samples relayed per mission rose from 0 (episode 1) to "
    f"{q['sent_mean']:.2f} of 4 for Q-learning; crater losses fell from 78% (random) to {pct(q['crater_rate'])}.",
])

heading("4.3 Graphs and Visualization", before=4)
para("Training curves are smoothed with a moving average of 1% of the episodes; the episode axis is logarithmic "
     "so that early learning and the final plateau are both visible. Dashed lines mark the evaluation return of "
     "the reference policies.")
fr = figure("reward_vs_episode.png", "Reward vs Episode for all tabular agents.", 6.0)
fs = figure("success_vs_episode.png", "Success Rate vs Episode.", 6.0)
fst = figure("steps_vs_episode.png", "Steps/Episode vs Episode.", 6.0)
figure("samples_sent_vs_episode.png", "Science return (samples relayed to Earth) vs Episode.", 5.6)
ffc = figure("final_comparison.png", "Final comparison of frozen policies: return, success rate and failure modes.", 6.5)
fvh = figure("value_policy_heatmap.png", "State values and greedy actions at the start of the mission: Value "
             "Iteration (optimal) vs Q-learning (learned). L/A mark sites; text marks non-move actions.", 6.5)
ftr = figure("trajectories_q_vs_sarsa.png", "Greedy trajectories of Q-learning, SARSA and the heuristic on the "
             "same three evaluation missions.", 6.5)
fcv = figure("convergence_speed.png", "Episodes to reach 90% of final performance and wall-clock training time.", 6.2)
fvi = figure("value_iteration_convergence.png", "Value Iteration convergence (max |Vₖ₊₁ − Vₖ| per sweep).", 4.3)
fhp = figure("hyperparameter_sweeps.png", "Hyper-parameter sweeps (100,000 episodes each; Dyna-Q 20,000).", 6.5)
fdq = figure("dqn_training.png", "DQN training on perturbed maps: return, success rate and TD loss.", 6.5)
fdg = figure("dqn_generalization.png", "Generalisation to unseen maps: DQN vs heuristic vs random.", 6.2)

heading("4.4 Result Analysis", before=4)
sa, dy, ns, mc, dq = T["SARSA"], T["Dyna-Q (k=10)"], T["n-step SARSA (n=4)"], T["Monte Carlo"], T["DQN"]
para("**What did the agent learn?** Starting from random behaviour, Q-learning learned a complete mission plan. "
     "In a typical evaluation mission (the first of the 500) it drives to the nearby LIBS site in the south-west "
     "(step 5), immediately returns into Vikram's radio range to relay the result (step 7) and recharges in "
     "sunlight. It then crosses into the permanently shadowed region for the APXS site (step 20), comes back out "
     "to relay (step 23), visits the far APXS site on the east side (step 35), relays again (step 41), recharges, "
     "climbs to the north-east LIBS site (step 55) and relays the last sample (step 68). It then waits in sunlight "
     "near Vikram until dusk and sleeps with 4 battery units (step 84, score 135). The agent learned to **relay "
     "after every sample instead of carrying data**, which protects science against a later crater or battery "
     "failure. It always uses the correct instrument, and it recharges only in sunlight, before long drives. The "
     f"value map (Figure {fvh}) shows the same gradient towards the sites as the optimal Value-Iteration solution.")
para(f"**How did performance change during training?** All agents start at about −110 per mission (they fall into "
     f"a crater or drain the battery within about 20 steps). Q-learning and SARSA stay there for roughly 15,000 "
     f"episodes while ε is high, then improve steadily as ε decays, crossing zero return after about 150,000 "
     f"episodes and reaching a smoothed training return of {C.get('Q-learning', {}).get('end', 0):.0f} (Q-learning) "
     f"and {C.get('SARSA', {}).get('end', 0):.0f} (SARSA) by 300,000 episodes (Figure {fr}). Steps per episode rise "
     f"from about 20 to 75, because surviving agents use the whole lunar day (Figure {fst}). Success rises from 0 "
     f"to about 30% during training (Figure {fs}); it is higher at evaluation because exploration is switched off.")
para(f"**Did the agent converge?** Value Iteration converged exactly ({VI['sweeps']} sweeps, Figure {fvi}). The "
     "learning agents converged in behaviour (their curves flatten after about 200,000 episodes, Figure "
     f"{fcv}) but **not fully in value**: Q-learning and SARSA were still slowly improving at 300,000 episodes, and "
     f"Q-learning's return ({q['return_mean']:.1f}) remains below the optimum ({vi['return_mean']:.1f}). With "
     f"{VI['n_reachable_states']:,} reachable states and 13 actions, many rare state–action pairs (low battery late "
     "in the day far from Vikram) are visited too rarely for their values to settle.")
para("**What observations can be made from the graphs?**")
bullets([
    f"**On-policy vs off-policy (SARSA vs Q-learning).** SARSA's crater rate is {pct(sa['crater_rate'])} against "
    f"{pct(q['crater_rate'])} for Q-learning, but its return is lower ({sa['return_mean']:.1f} vs "
    f"{q['return_mean']:.1f}). SARSA learns the value of the ε-greedy policy it actually follows, so it keeps a "
    "wider margin from craters where a random action could be fatal. Q-learning learns the greedy policy and takes "
    "shorter, riskier routes – the lunar version of the classic cliff-walking result (Figure "
    f"{ftr}).",
    f"**Planning makes learning sample-efficient (Dyna-Q).** Dyna-Q reached 90% of its final return in "
    f"{C.get('Dyna-Q (k=10)', {}).get('ep90', 0):,} episodes compared with {C.get('Q-learning', {}).get('ep90', 0):,} "
    "for Q-learning, about five times fewer real episodes, because each real step is followed by 10 simulated "
    "updates from its learned model. Each episode costs more computation, so with the same wall-clock budget its "
    f"final return ({dy['return_mean']:.1f}) is similar to SARSA's but below Q-learning's. In the sweep (Figure "
    f"{fhp}), Dyna-Q with k = 0 (plain Q-learning) scored {SW['sweep:planning=0']['return_mean']:.0f} after 20,000 "
    f"episodes, while k = 5 scored {SW['sweep:planning=5']['return_mean']:.0f}.",
    f"**Longer returns hurt here (n-step SARSA and Monte Carlo).** n-step SARSA got worse as n grew (n = 1: "
    f"{SW['sweep:n=1']['return_mean']:.0f}, n = 8: {SW['sweep:n=8']['return_mean']:.0f}), and Monte Carlo was the "
    f"weakest learner ({mc['return_mean']:.1f}). Slips and the −100 crater penalty make multi-step returns very "
    "noisy under an exploring policy. Bootstrapping after one step reduces this variance, which is why the "
    "one-step TD methods learned best. (Our check confirmed that n-step SARSA with n = 1 matches plain SARSA, so "
    "this is a property of the problem, not a bug.)",
    f"**Hyper-parameters.** α between 0.05 and 0.2 performed similarly; α = 0.5 was clearly worse "
    f"({SW['sweep:alpha=0.5']['return_mean']:.0f}) because large steps chase the noise of single slips. γ between "
    "0.9 and 0.99 made little difference at 100,000 episodes. A faster ε decay (50% of training) gave the best "
    f"return ({SW['sweep:eps_decay=0.5']['return_mean']:.1f} vs {SW['sweep:eps_decay=0.8']['return_mean']:.1f} for "
    "80%). The main runs used the 80% setting chosen before the sweep, so a re-run with 50% would probably improve "
    "Q-learning further.",
    f"**DQN and generalisation.** The DQN sees only its 5 × 5 NavCam window, yet on the landing zone it relays "
    f"{dq['sent_mean']:.2f} samples per mission with {pct(dq['success_rate'])} success, and on landing zones with "
    f"new hazards it still scores {G['DQN']['perturbed']['return_mean']:.1f} with "
    f"{pct(G['DQN']['perturbed']['success_rate'])} success – something a table indexed by exact cells cannot do. "
    f"Its weaknesses are a high crater rate ({pct(dq['crater_rate'])}) and failure on fully random maps "
    f"({G['DQN']['random']['return_mean']:.1f}), which are very different from its training distribution.",
])
para("**How effective is the learned policy?** Compared with the random policy the improvement is large (+185 "
     f"points, 78% → {pct(q['crater_rate'])} crater losses). Compared with the optimal planner, Q-learning achieves "
     f"{100 * q['return_mean'] / vi['return_mean']:.0f}% of the optimal return. The hand-written heuristic scores "
     f"slightly higher on return ({he['return_mean']:.1f}) and clearly higher on success ({pct(he['success_rate'])} "
     f"vs {pct(q['success_rate'])}). It should be read as a strong, privileged baseline: it is given the full map "
     "and shortest-path search, much like Value Iteration, while the learning agents receive no model at all. "
     f"Even so, the learned policies are much **safer**. Q-learning's battery-death rate is {pct(q['battery_rate'])} "
     f"against {pct(he['battery_rate'])} for the heuristic, and its night-failure rate {pct(q['night_rate'])} against "
     f"{pct(he['night_rate'])}. The learned policy has discovered that recharging before long drives and waiting "
     "safely for dusk is worth a slightly lower science score, a trade-off nobody wrote down by hand.")

heading("4.5 Limitations", before=4)
bullets([
    "The environment is a simplified 12 × 12 grid inspired by the mission; it is not built from a real lunar "
    "elevation model, and slip, battery and sun dynamics are abstract probabilities.",
    "The tabular agents learn one fixed map. Their 209,952-state table does not transfer to new terrain, and "
    "300,000 episodes were not enough for full value convergence.",
    "The DQN generalises only to maps similar to its training distribution. It still falls into craters in about "
    "30% of missions and fails on fully random maps.",
    "All results come from one training run per agent. ± values reflect evaluation randomness, not variation "
    "between training seeds, so close results (e.g. SARSA vs Dyna-Q) should not be ranked.",
    "The main runs used hyper-parameters chosen before the sweep (ε decay over 80% of training), which the sweep "
    "later showed was not the best setting.",
    "Communication delay, thermal limits, wheel wear and partial observability of the tabular state are not "
    "modelled.",
])

heading("4.6 Future Scope", before=4)
bullets([
    "Build the grid from a real lunar digital elevation model (for example LRO or Chandrayaan-2 OHRC data) and "
    "derive slip and energy costs from the slope.",
    "Treat the hidden terrain as a POMDP with a recurrent or attention-based DQN, so that the rover reasons about "
    "what it has not yet seen.",
    "Use stronger deep-RL methods (prioritised replay, dueling networks, PPO) and curriculum training from the "
    "landing zone to fully random maps to close the generalisation gap.",
    "Use multiple training seeds with confidence intervals, and longer or tuned tabular training (ε decay 50%, "
    "1M episodes) to approach the optimal return.",
    "Extend to multi-objective or constrained RL (hard safety constraints on crater risk), and to multi-rover "
    "cooperation for future missions such as Chandrayaan-4 and LUPEX.",
])

heading("4.7 Conclusion", before=4)
para("We formulated one lunar day of the Chandrayaan-3 Pragyan rover as an MDP. The rover must sample four science "
     "sites with the right instrument, relay data through Vikram, manage a stochastic battery and the sun, avoid "
     "craters on slippery regolith and sleep safely before night. We implemented it as a Gymnasium environment "
     "with an exact transition model and trained seven methods from the syllabus on it. Value Iteration gave the "
     f"optimal reference ({vi['return_mean']:.1f}). Q-learning was the best model-free learner "
     f"({q['return_mean']:.1f}, {pct(q['success_rate'])} success, {q['sent_mean']:.2f}/4 samples relayed). SARSA "
     f"learned the safest policy ({pct(sa['crater_rate'])} craters), and Dyna-Q learned about five times faster per "
     "episode thanks to planning. n-step and Monte Carlo methods suffered from the high variance of long returns. "
     "A Double DQN with a 5 × 5 camera view learned to complete missions on terrain it had never seen and to reroute "
     "around newly dropped craters in an interactive simulation. A rule-based planner with full map knowledge still "
     "matches the learners on score, but the learned policies are markedly safer, and closing the remaining gap "
     "to the optimum is a matter of longer training, tuned exploration and richer function approximation.")

# ================================== 5. REFERENCES ==============================================
heading("5. References")
refs = [
    "R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.",
    "C. J. C. H. Watkins and P. Dayan, “Q-learning,” Machine Learning, vol. 8, pp. 279–292, 1992.",
    "G. A. Rummery and M. Niranjan, “On-line Q-learning using connectionist systems,” Tech. Rep. CUED/F-INFENG/TR "
    "166, Cambridge University, 1994 (SARSA).",
    "R. S. Sutton, “Integrated architectures for learning, planning, and reacting based on approximating dynamic "
    "programming,” Proc. ICML, 1990 (Dyna).",
    "V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, pp. 529–533, 2015.",
    "H. van Hasselt, A. Guez and D. Silver, “Deep reinforcement learning with double Q-learning,” Proc. AAAI, 2016.",
    "A. Y. Ng, D. Harada and S. Russell, “Policy invariance under reward transformations: theory and application to "
    "reward shaping,” Proc. ICML, 1999.",
    "R. Bellman, Dynamic Programming. Princeton University Press, 1957.",
    "R. Francis et al., “AEGIS autonomous targeting for ChemCam on Mars Science Laboratory: deployment and results of "
    "initial science team use,” Science Robotics, vol. 2, 2017.",
    "Indian Space Research Organisation, Chandrayaan-3 mission updates, https://www.isro.gov.in/Chandrayaan3.html.",
    "“Chandrayaan-3 rover confirms presence of sulphur on lunar surface,” Gulf News / ISRO statement, Aug. 2023.",
    "“Chandrayaan-3: rover encounters 4 m wide crater, alters path,” The Tribune, 28 Aug. 2023.",
    "“Chandrayaan-3 rover moves 100 metres from the lander,” Deccan Herald, Sep. 2023.",
    "“India says good night to Chandrayaan-3,” SpacePolicyOnline, Sep. 2023 (landing site 69.36° S, 32.34° E).",
    "M. Towers et al., “Gymnasium: a standard interface for reinforcement learning environments,” 2024; "
    "https://gymnasium.farama.org.",
    "PyTorch documentation, https://pytorch.org/docs; NumPy documentation, https://numpy.org/doc; Matplotlib "
    "documentation, https://matplotlib.org.",
]
for r_ in refs:
    p = doc.add_paragraph(style="List Number"); p.paragraph_format.space_after = Pt(2)
    run(p, r_, size=11)

# ================================== APPENDIX ===================================================
heading("Appendix")
heading("A. Source code / GitHub repository link", size=12, before=4)
para(f"Source code: {REPO}", align=WD_ALIGN_PARAGRAPH.LEFT, after=2)
para(f"Live project website and interactive simulator: {SITE}", align=WD_ALIGN_PARAGRAPH.LEFT, after=2)
para("Run locally: in rl/, python -m pragyan.env (self-check), python train.py (all experiments), python plots.py "
     "(figures); in site/, npm install and npm run dev (website).", align=WD_ALIGN_PARAGRAPH.LEFT)
heading("B. Additional screenshots", size=12, before=4)
placeholder("[Additional demo screenshots to be inserted here: untrained rover falling into a crater, DQN on an "
            "unseen hazard map, Q-learning vs SARSA head-to-head]")
heading("C. Sample outputs", size=12, before=4)
para("Environment self-check and an excerpt of the training log:", align=WD_ALIGN_PARAGRAPH.LEFT)
code("$ python -m pragyan.env\n"
     "env self-check passed\n\n"
     "$ python train.py\n"
     f"[   147s] done: Value Iteration      ({VI['n_reachable_states']:,} reachable states, {VI['sweeps']} sweeps)\n"
     "[   831s] done: Q-learning\n[   831s] done: SARSA\n"
     "[  1003s] done: n-step SARSA (n=4)\n[  1003s] done: Monte Carlo\n[  1003s] done: Dyna-Q (k=10)\n"
     "  ...\n"
     "  dqn step 1800000 ep  44652 eps 0.05 ret   -16.2 succ 0.10 (795s)\n"
     "[  2018s] done: DQN\nfinished in 2018s")
para("Key events of the first greedy Q-learning evaluation mission (from the simulator\'s mission log):",
     align=WD_ALIGN_PARAGRAPH.LEFT)
code("T+005 LIBS     LIBS laser fired: Sulphur (S) detected           +9\n"
     "T+007 TRANSMIT Data relayed to Earth via Vikram                +24\n"
     "T+008 RECHARGE Solar panel charging                             -1\n"
     "T+020 APXS     APXS spectrum: Mg, Al, Si, Ca, Ti, Fe             +9\n"
     "T+022 S        Wheel slip on loose regolith - veered off course  -1\n"
     "T+023 TRANSMIT Data relayed to Earth via Vikram                +24\n"
     "T+035 APXS     APXS spectrum: Mg, Al, Si, Ca, Ti, Fe             +9\n"
     "T+041 TRANSMIT Data relayed to Earth via Vikram                +24\n"
     "T+055 LIBS     LIBS laser fired: Sulphur (S) detected           +9\n"
     "T+068 TRANSMIT Data relayed to Earth via Vikram                +24\n"
     "T+084 SLEEP    Parked in sunlight, payloads off - sleep mode    +79\n"
     "MISSION COMPLETE - all science relayed, score 135")
heading("D. Important code snippets", size=12, before=4)
para("Transition model (excerpt of outcomes(), pragyan/env.py):", align=WD_ALIGN_PARAGRAPH.LEFT)
code("if a < 8:   # drive: intended direction 90%, veer 45 degrees left/right 5% each\n"
     "    for d, p in ((a, 1 - SLIP_P), ((a - 1) % 8, SLIP_P / 2), ((a + 1) % 8, SLIP_P / 2)):\n"
     "        nr, nc, bumped = _move_target(m, r, c, d)\n"
     "        if m.terrain[nr, nc] == CRATER:\n"
     "            base.append((p, None, R_STEP + R_CRATER, True, \"crater\")); continue\n"
     "        ...\n"
     "# the sun moves on: phase advances with PHASE_P; after dusk comes the night\n"
     "out.append((p * (1 - PHASE_P), ns, rew, False, ev))\n"
     "out.append((p * PHASE_P, ns, rew + R_NIGHT, True, \"night\") if ph == 2 else ...)")
para("Q-learning and SARSA updates (pragyan/tabular.py):", align=WD_ALIGN_PARAGRAPH.LEFT)
code("a = egreedy(Q, s, eps, rng)                       # random valid action w.p. eps\n"
     "s2, r, term, trunc, info = env.step(a)\n"
     "target = r if term else r + gamma * Q[s2].max()   # Q-learning: off-policy max\n"
     "Q[s, a] += alpha * (target - Q[s, a])\n\n"
     "a2 = egreedy(Q, s2, e, rng)\n"
     "target = r if term else r + gamma * Q[s2, a2]     # SARSA: on-policy next action\n"
     "Q[s, a] += alpha * (target - Q[s, a])")
para("Double-DQN target with n-step returns and action masking (pragyan/dqn.py):", align=WD_ALIGN_PARAGRAPH.LEFT)
code("bs, ba, br, bs2, bd, bm2, bg = buf.sample(rng, batch)\n"
     "with torch.no_grad():\n"
     "    a2 = q(bs2).masked_fill(~bm2, -1e9).argmax(1, keepdim=True)   # online net selects\n"
     "    q2 = tgt(bs2).gather(1, a2).squeeze(1)                          # target net evaluates\n"
     "    y = br + bg * (1 - bd) * q2                                     # bg = gamma ** n\n"
     "loss = smooth_l1_loss(q(bs).gather(1, ba[:, None]).squeeze(1), y)")

out = os.path.join(HERE, "Pragyan_RL_Report.docx")
doc.save(out)
print("wrote", out)
