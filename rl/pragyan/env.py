"""PragyanEnv: a lunar-rover mission MDP inspired by Chandrayaan-3's Pragyan rover.

The rover starts next to the Vikram lander near the lunar south pole and has one
lunar day to: sample science sites with the right instrument (LIBS or APXS),
relay the data back through Vikram (only inside its radio range), keep its
battery alive (solar recharge does not work in shadow), avoid craters (wheels
slip on loose regolith), and park safely in sunlight before lunar night.

All dynamics live in one function, `outcomes(state, action)`, which returns the
full transition distribution. `step()` samples from it and Value Iteration reads
it directly, so the model the planner sees is exactly the environment.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---- terrain codes -----------------------------------------------------------
REGOLITH, SLOPE, BOULDER, CRATER, SHADOW, VIKRAM = range(6)
TERRAIN_NAMES = ["regolith", "slope", "boulder", "crater", "shadow", "vikram"]

# ---- actions ------------------------------------------------------------------
DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW",
                "LIBS", "APXS", "TRANSMIT", "RECHARGE", "SLEEP"]
LIBS, APXS, TRANSMIT, RECHARGE, SLEEP = 8, 9, 10, 11, 12
N_ACTIONS = 13

# ---- site types / status -------------------------------------------------------
SITE_LIBS, SITE_APXS = 0, 1
UNSAMPLED, HELD, SENT = 0, 1, 2

# ---- dynamics parameters ---------------------------------------------------------
MAX_BATTERY = 5
SLIP_P = 0.10             # wheel slip on regolith: veer 45 degrees left/right
DRAIN_P = 0.15            # chance a move costs 1 battery unit
DRAIN_P_SLOPE = 0.50      # ... when climbing onto a slope
PHASE_P = 0.025           # chance per step the sun moves to the next phase
PHASES = ["morning", "noon", "dusk"]   # after dusk comes lunar night
COMM_RADIUS = 3           # Chebyshev distance to Vikram for relay
MAX_STEPS = 200           # safety truncation

# ---- rewards -------------------------------------------------------------------------
R_STEP = -1.0
R_SLOPE = -1.0
R_BUMP = -1.0
R_SAMPLE = 10.0
R_SENT = 25.0
R_WRONG = -2.0
R_CRATER = -100.0          # losing the rover is catastrophic
R_DEAD = -100.0
R_SLEEP_SAFE = 30.0
R_MISSION = 50.0          # bonus: safe sleep with every sample relayed
R_SLEEP_UNSAFE = -50.0     # hibernating early, in shadow or on low battery = mission lost
R_NIGHT = -30.0

# Fixed landing-zone map (inspired by, not a replica of, the Shiv Shakti site).
#  . regolith  ^ slope  # boulder  O crater  s shadow  V Vikram  R rover start
#  L / l  LIBS site (l = inside shadow)      A / a  APXS site (a = inside shadow)
LANDING_ZONE = [
    "sss..^^...L.",
    "sas..^#.O...",
    "sss.....O...",
    "....O.....^.",
    "..#....O.O..",
    "............",
    "..VR...#....",
    ".....^...O..",
    ".#..OO....A.",
    ".......^^...",
    "L....#....O.",
    "............",
]


class LunarMap:
    """Terrain grid + science sites + Vikram position."""

    def __init__(self, terrain, sites, vikram, start):
        self.terrain = np.asarray(terrain, dtype=np.int8)
        self.H, self.W = self.terrain.shape
        self.sites = list(sites)            # [(r, c, type), ...]
        self.vikram = tuple(vikram)
        self.start = tuple(start)
        self.site_at = {(r, c): i for i, (r, c, _) in enumerate(self.sites)}

    @classmethod
    def from_strings(cls, rows):
        code = {".": REGOLITH, "^": SLOPE, "#": BOULDER, "O": CRATER, "s": SHADOW,
                "V": VIKRAM, "R": REGOLITH, "L": REGOLITH, "A": REGOLITH,
                "l": SHADOW, "a": SHADOW}
        terrain, libs, apxs, vikram, start = [], [], [], None, None
        for r, row in enumerate(rows):
            terrain.append([code[ch] for ch in row])
            for c, ch in enumerate(row):
                if ch in "Ll":
                    libs.append((r, c, SITE_LIBS))
                elif ch in "Aa":
                    apxs.append((r, c, SITE_APXS))
                elif ch == "V":
                    vikram = (r, c)
                elif ch == "R":
                    start = (r, c)
        return cls(terrain, libs + apxs, vikram, start)

    def passable(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W and self.terrain[r, c] not in (BOULDER, VIKRAM)

    def in_range(self, r, c):
        return max(abs(r - self.vikram[0]), abs(c - self.vikram[1])) <= COMM_RADIUS

    def to_dict(self):
        return {"terrain": self.terrain.tolist(), "sites": [list(s) for s in self.sites],
                "vikram": list(self.vikram), "start": list(self.start)}


def random_map(rng, H=12, W=12):
    """Random lunar terrain with guaranteed reachable sites (used for DQN)."""
    while True:
        t = np.full((H, W), REGOLITH, dtype=np.int8)
        # long south-pole shadow: a rectangle hugging one edge
        sh, sw = rng.integers(2, 5), rng.integers(3, 6)
        r0 = rng.choice([0, H - sh]); c0 = rng.integers(0, W - sw + 1)
        t[r0:r0 + sh, c0:c0 + sw] = SHADOW
        for _ in range(rng.integers(2, 5)):            # slope patches
            r, c = rng.integers(0, H - 1), rng.integers(0, W - 1)
            t[r:r + 2, c:c + rng.integers(1, 3)] = SLOPE
        free = lambda: (int(rng.integers(0, H)), int(rng.integers(0, W)))
        vr, vc = int(rng.integers(3, H - 3)), int(rng.integers(2, W - 2))
        t[vr, vc] = VIKRAM
        start = (vr, vc + 1)
        t[start] = REGOLITH
        reserved = {(vr, vc), start}
        for code, n in ((CRATER, rng.integers(5, 9)), (BOULDER, rng.integers(4, 8))):
            k = 0
            while k < n:
                r, c = free()
                if (r, c) in reserved or max(abs(r - vr), abs(c - vc)) <= 1:
                    continue
                t[r, c] = code; reserved.add((r, c)); k += 1
        sites = []
        while len(sites) < 4:
            r, c = free()
            if (r, c) in reserved or t[r, c] in (CRATER, BOULDER, VIKRAM):
                continue
            if max(abs(r - vr), abs(c - vc)) <= 2:
                continue
            sites.append((r, c, SITE_LIBS if len(sites) < 2 else SITE_APXS))
            reserved.add((r, c))
        m = LunarMap(t, sites, (vr, vc), start)
        reach = bfs_dist(m, start)
        if all((r, c) in reach for r, c, _ in sites):
            return m


def bfs_dist(m, src):
    """Shortest 8-connected path lengths avoiding craters (used by greedy + map checks)."""
    from collections import deque
    dist = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in dist and m.passable(nr, nc) and m.terrain[nr, nc] != CRATER:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


# ---- the transition model ------------------------------------------------------------
# state = (r, c, battery, sites_status_tuple, phase)

def _move_target(m, r, c, d):
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    if not m.passable(nr, nc):
        return r, c, True
    return nr, nc, False


def outcomes(m, state, a):
    """Full distribution over (prob, next_state, reward, terminated, event)."""
    r, c, bat, sites, phase = state
    base = []   # outcomes of the action itself, before the sun moves

    if a < 8:
        for d, p in ((a, 1 - SLIP_P), ((a - 1) % 8, SLIP_P / 2), ((a + 1) % 8, SLIP_P / 2)):
            nr, nc, bumped = _move_target(m, r, c, d)
            ev = "slip" if d != a else "move"
            if bumped:
                base.append((p, (r, c, bat, sites, phase), R_STEP + R_BUMP, False, "blocked"))
                continue
            if m.terrain[nr, nc] == CRATER:
                base.append((p, None, R_STEP + R_CRATER, True, "crater"))
                continue
            slope = m.terrain[nr, nc] == SLOPE
            rew = R_STEP + (R_SLOPE if slope else 0.0)
            q = DRAIN_P_SLOPE if slope else DRAIN_P
            base.append((p * (1 - q), (nr, nc, bat, sites, phase), rew, False, ev))
            if bat - 1 <= 0:
                base.append((p * q, None, rew + R_DEAD, True, "battery_dead"))
            else:
                base.append((p * q, (nr, nc, bat - 1, sites, phase), rew, False, ev))

    elif a in (LIBS, APXS):
        want = SITE_LIBS if a == LIBS else SITE_APXS
        i = m.site_at.get((r, c))
        ok = i is not None and m.sites[i][2] == want and sites[i] == UNSAMPLED
        if bat - 1 <= 0:
            base.append((1.0, None, R_STEP + R_DEAD, True, "battery_dead"))
        elif ok:
            ns = sites[:i] + (HELD,) + sites[i + 1:]
            base.append((1.0, (r, c, bat - 1, ns, phase), R_STEP + R_SAMPLE, False,
                         "libs_ok" if a == LIBS else "apxs_ok"))
        else:
            base.append((1.0, (r, c, bat - 1, sites, phase), R_STEP + R_WRONG, False, "wrong_instrument"))

    elif a == TRANSMIT:
        held = sum(1 for s in sites if s == HELD)
        if bat - 1 <= 0:
            base.append((1.0, None, R_STEP + R_DEAD, True, "battery_dead"))
        elif held and m.in_range(r, c):
            ns = tuple(SENT if s == HELD else s for s in sites)
            base.append((1.0, (r, c, bat - 1, ns, phase), R_STEP + R_SENT * held, False, "tx_ok"))
        else:
            base.append((1.0, (r, c, bat - 1, sites, phase), R_STEP + R_WRONG, False, "tx_fail"))

    elif a == RECHARGE:
        if m.terrain[r, c] == SHADOW:
            base.append((1.0, state, R_STEP, False, "recharge_shadow"))
        else:
            nb = min(MAX_BATTERY, bat + (2 if phase == 1 else 1))
            base.append((1.0, (r, c, nb, sites, phase), R_STEP, False, "recharge"))

    else:  # SLEEP
        # safe hibernation = at dusk (end of the lunar day), in sunlight, battery >= 3
        safe = phase == 2 and m.terrain[r, c] != SHADOW and bat >= 3
        bonus = R_MISSION if safe and all(s == SENT for s in sites) else 0.0
        base.append((1.0, None, R_STEP + bonus + (R_SLEEP_SAFE if safe else R_SLEEP_UNSAFE), True,
                     "sleep_safe" if safe else "sleep_unsafe"))

    # the sun moves on: phase advances with PHASE_P; after dusk comes the night
    out = []
    for p, ns, rew, term, ev in base:
        if term:
            out.append((p, ns, rew, term, ev))
            continue
        out.append((p * (1 - PHASE_P), ns, rew, False, ev))
        nr, nc, nb, nsites, ph = ns
        if ph == 2:
            out.append((p * PHASE_P, ns, rew + R_NIGHT, True, "night"))
        else:
            out.append((p * PHASE_P, (nr, nc, nb, nsites, ph + 1), rew, False, ev))
    return out


def valid_actions(m, state):
    """Invalid-action masking: the rover's onboard software refuses commands that
    cannot execute (driving into a boulder, LIBS off-site, transmitting with no data
    or out of range, recharging in shadow or at full charge, hibernating before dusk).
    Craters stay drivable: avoiding them is what the agent must learn."""
    r, c, bat, sites, phase = state
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for d, (dr, dc) in enumerate(DIRS):
        mask[d] = m.passable(r + dr, c + dc)
    i = m.site_at.get((r, c))
    if i is not None and sites[i] == UNSAMPLED:
        mask[LIBS if m.sites[i][2] == SITE_LIBS else APXS] = True
    mask[TRANSMIT] = HELD in sites and m.in_range(r, c)
    mask[RECHARGE] = m.terrain[r, c] != SHADOW and bat < MAX_BATTERY
    mask[SLEEP] = phase == 2
    if not mask.any():
        mask[SLEEP] = True
    return mask


def perturbed_map(rng, base=None):
    """The landing zone with fresh hazards: 2-4 new craters and boulders appear,
    0-2 old ones vanish, science sites jitter by up to 1 cell (used to train DQN)."""
    base = base or LunarMap.from_strings(LANDING_ZONE)
    while True:
        t = base.terrain.copy()
        vr, vc = base.vikram
        sites = []
        for r, c, typ in base.sites:
            for _ in range(10):
                nr, nc = r + int(rng.integers(-1, 2)), c + int(rng.integers(-1, 2))
                if 0 <= nr < base.H and 0 <= nc < base.W and t[nr, nc] in (REGOLITH, SHADOW, SLOPE) \
                        and (nr, nc) not in [(x[0], x[1]) for x in sites] and (nr, nc) != base.start:
                    break
            else:
                nr, nc = r, c
            sites.append((nr, nc, typ))
        hazards = [(r, c) for r in range(base.H) for c in range(base.W) if t[r, c] in (CRATER, BOULDER)]
        for j in rng.permutation(len(hazards))[:int(rng.integers(0, 3))]:
            t[hazards[j]] = REGOLITH
        reserved = {(x[0], x[1]) for x in sites} | {base.start}
        for code in (CRATER, BOULDER):
            k, n = 0, int(rng.integers(2, 5))
            while k < n:
                r, c = int(rng.integers(0, base.H)), int(rng.integers(0, base.W))
                if (r, c) in reserved or t[r, c] in (CRATER, BOULDER, VIKRAM) or max(abs(r - vr), abs(c - vc)) <= 1:
                    continue
                t[r, c] = code; reserved.add((r, c)); k += 1
        m = LunarMap(t, sites, base.vikram, base.start)
        reach = bfs_dist(m, m.start)
        if all((r, c) in reach for r, c, _ in sites):
            return m


# ---- state encodings ---------------------------------------------------------------------

def state_index(m, state):
    """Unique integer for tabular methods."""
    r, c, bat, sites, phase = state
    code = 0
    for s in reversed(sites):
        code = code * 3 + s
    return (((r * m.W + c) * (MAX_BATTERY + 1) + bat) * 3 ** len(sites) + code) * 3 + phase


def n_states(m):
    return m.H * m.W * (MAX_BATTERY + 1) * 3 ** len(m.sites) * 3


VIEW = 2                                  # 5x5 local camera window
FEATURE_DIM = 4 * (2 * VIEW + 1) ** 2 + 1 + 3 + 4 * 5 + 3 + 2 + 1


def features(m, state):
    """Observation vector for DQN: local NavCam view + mission status + relative vectors."""
    r, c, bat, sites, phase = state
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    k = 0
    for dr in range(-VIEW, VIEW + 1):
        for dc in range(-VIEW, VIEW + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < m.H and 0 <= cc < m.W):
                f[k] = 1.0                     # off-map counts as blocked
            else:
                t = m.terrain[rr, cc]
                if t in (BOULDER, VIKRAM): f[k] = 1.0
                elif t == CRATER: f[k + 1] = 1.0
                elif t == SLOPE: f[k + 2] = 1.0
                elif t == SHADOW: f[k + 3] = 1.0
            k += 4
    f[k] = bat / MAX_BATTERY; k += 1
    f[k + phase] = 1.0; k += 3
    for i, (sr, sc, _) in enumerate(m.sites):
        f[k + sites[i]] = 1.0
        f[k + 3] = (sr - r) / m.H
        f[k + 4] = (sc - c) / m.W
        k += 5
    f[k] = (m.vikram[0] - r) / m.H
    f[k + 1] = (m.vikram[1] - c) / m.W
    f[k + 2] = 1.0 if m.in_range(r, c) else 0.0
    k += 3
    i = m.site_at.get((r, c))
    if i is not None and sites[i] == UNSAMPLED:
        f[k + m.sites[i][2]] = 1.0
    k += 2
    f[k] = 1.0 if m.terrain[r, c] == SHADOW else 0.0
    return f


class PragyanEnv(gym.Env):
    """Gymnasium environment.

    obs_mode="index": observation is the tabular state index (fixed map).
    obs_mode="features": observation is the DQN feature vector.
    randomize: False (fixed landing zone), "perturb" (landing zone with new
    hazards each episode) or "random" (fully random lunar map each episode).
    """

    def __init__(self, lunar_map=None, randomize=False, obs_mode="index", seed=None):
        self.randomize = randomize
        self.obs_mode = obs_mode
        self.rng = np.random.default_rng(seed)
        self.map = lunar_map or LunarMap.from_strings(LANDING_ZONE)
        self.action_space = spaces.Discrete(N_ACTIONS)
        if obs_mode == "index":
            self.observation_space = spaces.Discrete(n_states(self.map))
        else:
            self.observation_space = spaces.Box(-1.0, 1.0, (FEATURE_DIM,), np.float32)
        self.state = None

    def _obs(self):
        if self.obs_mode == "index":
            return state_index(self.map, self.state)
        return features(self.map, self.state)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.randomize == "perturb":
            self.map = perturbed_map(self.rng)
        elif self.randomize:
            self.map = random_map(self.rng)
        r, c = self.map.start
        self.state = (r, c, MAX_BATTERY, (UNSAMPLED,) * len(self.map.sites), 0)
        self.t = 0
        return self._obs(), {}

    def action_mask(self):
        return valid_actions(self.map, self.state)

    def step(self, a):
        outs = outcomes(self.map, self.state, int(a))
        u = self.rng.random()
        for p, ns, rew, term, ev in outs:
            u -= p
            if u < 0:
                break
        self.t += 1
        info = {"event": ev, "success": False}
        if ns is not None:          # crater / dead / sleep keep the last position
            self.state = ns
        if term:
            info["success"] = ev == "sleep_safe" and all(s == SENT for s in self.state[3])
            info["sent"] = sum(s == SENT for s in self.state[3])
        trunc = (not term) and self.t >= MAX_STEPS
        if trunc:
            info["sent"] = sum(s == SENT for s in self.state[3])
        return self._obs(), rew, term, trunc, info


def _self_check():
    m = LunarMap.from_strings(LANDING_ZONE)
    assert m.vikram == (6, 2) and m.start == (6, 3) and len(m.sites) == 4
    s0 = (6, 3, MAX_BATTERY, (0, 0, 0, 0), 0)
    for a in range(N_ACTIONS):
        assert abs(sum(o[0] for o in outcomes(m, s0, a)) - 1) < 1e-9, a
    # crater ends the episode: rover just south of crater (3,4), moving N
    outs = outcomes(m, (4, 4, 5, (0, 0, 0, 0), 0), 0)
    assert any(o[3] and o[4] == "crater" for o in outs)
    # transmit outside range fails; inside range with held data succeeds
    far = outcomes(m, (11, 11, 5, (1, 0, 0, 0), 0), TRANSMIT)
    assert all(o[4] == "tx_fail" for o in far)
    near = outcomes(m, (6, 3, 5, (1, 0, 0, 0), 0), TRANSMIT)
    assert all(o[4] == "tx_ok" for o in near if not o[3])
    # recharge fails in shadow
    assert all(o[4] == "recharge_shadow" for o in outcomes(m, (0, 0, 2, (0, 0, 0, 0), 0), RECHARGE))
    # wrong instrument penalised
    lr, lc, _ = m.sites[0]
    assert outcomes(m, (lr, lc, 5, (0, 0, 0, 0), 0), APXS)[0][4] == "wrong_instrument"
    # state index is unique
    assert len({state_index(m, (r, c, b, (s, 0, 0, 0), p)) for r in range(12) for c in range(12)
                for b in range(6) for s in range(3) for p in range(3)}) == 12 * 12 * 6 * 3 * 3
    # gym API on both modes
    # masks: no LIBS off-site, no sleep before dusk, no transmit without data
    mk = valid_actions(m, s0)
    assert not mk[LIBS] and not mk[SLEEP] and not mk[TRANSMIT] and not mk[6]   # W is Vikram
    assert valid_actions(m, (lr, lc, 5, (0, 0, 0, 0), 2))[LIBS]
    for mode, rnd in (("index", False), ("features", "random"), ("features", "perturb")):
        env = PragyanEnv(randomize=rnd, obs_mode=mode, seed=0)
        obs, _ = env.reset(seed=1)
        if mode == "features":
            assert obs.shape == (FEATURE_DIM,)
        for _ in range(300):
            obs, rew, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        assert term or trunc
    print("env self-check passed")


if __name__ == "__main__":
    _self_check()
