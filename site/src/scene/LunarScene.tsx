import { useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, Line, OrbitControls, QuadraticBezierLine, Stars } from "@react-three/drei";
import * as THREE from "three";
import { BOULDER, COMM_RADIUS, HELD, SENT, VIEW, type LunarMap } from "../sim/env";
import type { Player } from "../sim/player";
import { buildTerrain, cellToWorld, fbm, heightAt, worldToCell, type Hazards } from "./terrain";

const GOLD = "#c9a13b";

export interface SceneOptions {
  labels?: boolean;
  range?: boolean;
  navcam?: boolean;
  camera?: "hero" | "control";
  /** hazards dropped by the user in live mode: drawn as separate meshes, terrain is never rebuilt */
  added?: AddedHazard[];
  onCell?: (r: number, c: number, shift: boolean) => void;
}

export interface AddedHazard { r: number; c: number; kind: "crater" | "boulder"; t0: number }

function usePlayer(p: Player) {
  useSyncExternalStore(p.subscribe, p.getVersion);
}

// ---------------------------------------------------------------------------------------------
export function LunarScene({ player, opts }: { player: Player; opts: SceneOptions }) {
  usePlayer(player);
  const m = player.map;
  const { geometry, hz } = useMemo(() => buildTerrain(m), [m]);
  const h = (x: number, z: number) => heightAt(m, hz, x, z);

  const onDown = (e: ThreeEvent<PointerEvent>) => {
    if (!opts.onCell) return;
    e.stopPropagation();
    const [r, c] = worldToCell(m, e.point.x, e.point.z);
    if (r >= 0 && c >= 0 && r < m.H && c < m.W) opts.onCell(r, c, e.shiftKey);
  };

  return (
    <>
      <color attach="background" args={["#000"]} />
      <fog attach="fog" args={["#000", opts.camera === "hero" ? 22 : 30, opts.camera === "hero" ? 46 : 60]} />
      <Sun player={player} />
      <ambientLight intensity={0.07} />
      <hemisphereLight args={["#9fb4d8", "#000", 0.08]} />
      <Stars radius={90} depth={40} count={3500} factor={3} saturation={0} fade speed={0.2} />
      <Earth />
      <mesh geometry={geometry} receiveShadow castShadow onPointerDown={onDown}>
        <meshStandardMaterial vertexColors roughness={1} metalness={0} />
      </mesh>
      <Boulders m={m} hz={hz} />
      {opts.added && <AddedHazards m={m} h={h} added={opts.added} />}
      <Vikram m={m} h={h} />
      <Sites player={player} h={h} labels={!!opts.labels} />
      {opts.range && <RangeOutline m={m} h={h} />}
      <Path player={player} h={h} />
      <Rover player={player} h={h} />
      {opts.navcam && <NavCam player={player} h={h} />}
      <Effects player={player} h={h} />
      <CameraRig mode={opts.camera ?? "control"} player={player} />
    </>
  );
}

// ---- sun: low raking light of the lunar south pole, moving with the mission phase ------------
const SUN = [
  { az: 1.05, el: 0.2, color: "#fff1dc", i: 3.1 },   // morning, from the east
  { az: 0.0, el: 0.36, color: "#ffffff", i: 3.4 },   // noon, from the north
  { az: -1.05, el: 0.16, color: "#ffd9b0", i: 2.8 }, // dusk, from the west
];
function Sun({ player }: { player: Player }) {
  const light = useRef<THREE.DirectionalLight>(null!);
  const target = useMemo(() => new THREE.Object3D(), []);
  const cur = useRef(new THREE.Vector3(10, 4, -8));
  const col = useRef(new THREE.Color());
  useFrame((_, dt) => {
    const night = player.done && player.endEv === "night";
    const s = SUN[Math.min(2, player.state[4])];
    const d = 30;
    const want = new THREE.Vector3(Math.sin(s.az) * Math.cos(s.el) * d, Math.sin(s.el) * d, -Math.cos(s.az) * Math.cos(s.el) * d);
    cur.current.lerp(want, 1 - Math.exp(-dt * 1.5));
    light.current.position.copy(cur.current);
    col.current.lerp(new THREE.Color(s.color), 1 - Math.exp(-dt * 1.5));
    light.current.color.copy(col.current);
    const wantI = night ? 0.15 : s.i;
    light.current.intensity += (wantI - light.current.intensity) * (1 - Math.exp(-dt * 2));
  });
  return (
    <>
      <primitive object={target} />
      <directionalLight ref={light} target={target} castShadow intensity={3} position={[10, 4, -8]}
        shadow-mapSize={[2048, 2048]} shadow-bias={-0.0004} shadow-normalBias={0.06}
        shadow-camera-left={-10} shadow-camera-right={10} shadow-camera-top={10} shadow-camera-bottom={-10}
        shadow-camera-near={1} shadow-camera-far={70} />
    </>
  );
}

function Earth() {
  return (
    <mesh position={[-38, 16, -60]}>
      <sphereGeometry args={[3.2, 48, 48]} />
      <meshStandardMaterial color="#5d84c4" roughness={0.6} emissive="#0b1a33" emissiveIntensity={0.6} />
    </mesh>
  );
}

// ---- boulders ---------------------------------------------------------------------------------
function rockGeometry(k: number) {
  const geo = new THREE.IcosahedronGeometry(1, 2);
  const p = geo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3(p.getX(i), p.getY(i), p.getZ(i));
    v.multiplyScalar(0.72 + 0.5 * fbm(v.x * 1.7 + k * 3, v.y * 1.7 + v.z * 1.3 + k, 3));
    v.y *= 0.68;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

function Boulders({ m, hz }: { m: LunarMap; hz: Hazards }) {
  const rocks = useMemo(() => {
    const out: { pos: [number, number, number]; geo: THREE.BufferGeometry; rot: number; s: number }[] = [];
    let k = 0;
    for (let r = 0; r < m.H; r++) for (let c = 0; c < m.W; c++) if (m.terrain[r][c] === BOULDER) {
      const [x, z] = cellToWorld(m, r, c);
      const geo = rockGeometry(k);
      out.push({ pos: [x, heightAt(m, hz, x, z) + 0.12, z], geo, rot: k * 1.7, s: 0.36 + 0.06 * ((k * 7) % 3) });
      k++;
    }
    return out;
  }, [m, hz]);
  return (
    <>
      {rocks.map((b, i) => (
        <mesh key={i} geometry={b.geo} position={b.pos} rotation={[0, b.rot, 0]} scale={b.s} castShadow receiveShadow>
          <meshStandardMaterial color="#6f6c66" roughness={0.95} flatShading />
        </mesh>
      ))}
    </>
  );
}

// ---- hazards dropped live: cheap meshes that pop in, no terrain rebuild -------------------------
const ADDED_ROCK = rockGeometry(7);
function AddedHazards({ m, h, added }: { m: LunarMap; h: (x: number, z: number) => number; added: AddedHazard[] }) {
  const refs = useRef<(THREE.Group | null)[]>([]);
  useFrame(() => {
    const now = performance.now();
    added.forEach((a, i) => {
      const g = refs.current[i];
      if (!g) return;
      const k = Math.min(1, (now - a.t0) / 350);
      g.scale.setScalar(1 - (1 - k) ** 3);
    });
  });
  return (
    <>
      {added.map((a, i) => {
        const [x, z] = cellToWorld(m, a.r, a.c), y = h(x, z);
        return (
          <group key={`${a.r}-${a.c}`} ref={(el) => { refs.current[i] = el; }} position={[x, y, z]}>
            {a.kind === "crater" ? (
              <>
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.012, 0]}>
                  <circleGeometry args={[0.4, 40]} />
                  <meshStandardMaterial color="#0b0a09" roughness={1} />
                </mesh>
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]} scale={[1, 1, 0.35]}>
                  <torusGeometry args={[0.42, 0.06, 10, 40]} />
                  <meshStandardMaterial color="#a6a29a" roughness={1} />
                </mesh>
              </>
            ) : (
              <mesh geometry={ADDED_ROCK} position={[0, 0.12, 0]} scale={0.38} castShadow>
                <meshStandardMaterial color="#6f6c66" roughness={0.95} flatShading />
              </mesh>
            )}
          </group>
        );
      })}
    </>
  );
}

// ---- Vikram lander ------------------------------------------------------------------------------
function Vikram({ m, h }: { m: LunarMap; h: (x: number, z: number) => number }) {
  const [x, z] = cellToWorld(m, m.vikram[0], m.vikram[1]);
  const y = h(x, z);
  const legs: [number, number][] = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  return (
    <group position={[x, y, z]}>
      <mesh position={[0, 0.42, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.62, 0.42, 0.62]} />
        <meshStandardMaterial color={GOLD} metalness={0.85} roughness={0.32} />
      </mesh>
      <mesh position={[0, 0.7, 0]} castShadow>
        <boxGeometry args={[0.5, 0.12, 0.5]} />
        <meshStandardMaterial color="#d8d6d0" metalness={0.4} roughness={0.5} />
      </mesh>
      <mesh position={[0.36, 0.5, 0]} rotation={[0, 0, 0.25]} castShadow>
        <boxGeometry args={[0.04, 0.38, 0.56]} />
        <meshStandardMaterial color="#1d2f5a" metalness={0.3} roughness={0.35} />
      </mesh>
      <mesh position={[0, 0.88, 0]} castShadow>
        <cylinderGeometry args={[0.012, 0.012, 0.3]} />
        <meshStandardMaterial color="#ccc" />
      </mesh>
      {legs.map(([a, b], i) => (
        <group key={i}>
          <mesh position={[a * 0.36, 0.2, b * 0.36]} rotation={[b * 0.45, 0, -a * 0.45]} castShadow>
            <cylinderGeometry args={[0.022, 0.022, 0.5]} />
            <meshStandardMaterial color="#bdbab3" metalness={0.6} roughness={0.4} />
          </mesh>
          <mesh position={[a * 0.46, 0.02, b * 0.46]} castShadow>
            <cylinderGeometry args={[0.07, 0.08, 0.03, 16]} />
            <meshStandardMaterial color="#bdbab3" />
          </mesh>
        </group>
      ))}
    </group>
  );
}

// ---- Pragyan rover ------------------------------------------------------------------------------
function Rover({ player, h }: { player: Player; h: (x: number, z: number) => number }) {
  const g = useRef<THREE.Group>(null!);
  const wheels = useRef<THREE.Group>(null!);
  const panel = useRef<THREE.MeshStandardMaterial>(null!);
  const last = useRef<[number, number]>([0, 0]);
  useFrame(() => {
    const p = player.pose(), m = player.map;
    const [x, z] = [p.c - m.W / 2 + 0.5, p.r - m.H / 2 + 0.5];
    const dist = Math.hypot(x - last.current[0], z - last.current[1]);
    last.current = [x, z];
    g.current.position.set(x, h(x, z) + 0.1 - p.fall * 0.45, z);
    g.current.rotation.y += (((p.heading - g.current.rotation.y + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI) * 0.25;
    g.current.scale.setScalar(1 - p.fall * 0.3);
    wheels.current.children.forEach((w) => { w.rotation.z -= dist * 6; });
    const charging = player.cur?.ev === "recharge";
    const dead = player.done && (player.endEv === "battery_dead" || player.endEv === "night");
    panel.current.emissiveIntensity = charging ? 0.9 + 0.4 * Math.sin(performance.now() / 120) : 0;
    panel.current.color.set(dead ? "#333" : "#2f4f9a");
  });
  const wheelPos: [number, number][] = [[-0.2, -0.2], [0, -0.21], [0.2, -0.2], [-0.2, 0.2], [0, 0.21], [0.2, 0.2]];
  return (
    <group ref={g} scale={1}>
      <group scale={0.95}>
        <mesh position={[0, 0.1, 0]} castShadow>
          <boxGeometry args={[0.5, 0.12, 0.34]} />
          <meshStandardMaterial color="#d9d5ca" metalness={0.5} roughness={0.45} />
        </mesh>
        <mesh position={[-0.04, 0.18, 0]} rotation={[0, 0, -0.12]} castShadow>
          <boxGeometry args={[0.42, 0.02, 0.36]} />
          <meshStandardMaterial ref={panel} color="#2f4f9a" metalness={0.3} roughness={0.3} emissive={GOLD} emissiveIntensity={0} />
        </mesh>
        <mesh position={[0.2, 0.26, 0]} castShadow>
          <cylinderGeometry args={[0.012, 0.012, 0.22]} />
          <meshStandardMaterial color="#bbb" />
        </mesh>
        <mesh position={[0.22, 0.38, 0]} castShadow>
          <boxGeometry args={[0.07, 0.05, 0.12]} />
          <meshStandardMaterial color="#222" metalness={0.5} roughness={0.4} />
        </mesh>
        <group ref={wheels}>
          {wheelPos.map(([wx, wz], i) => (
            <mesh key={i} position={[wx, 0.06, wz]} rotation={[Math.PI / 2, 0, 0]} castShadow>
              <cylinderGeometry args={[0.065, 0.065, 0.05, 14]} />
              <meshStandardMaterial color="#3a3a3a" roughness={0.8} />
            </mesh>
          ))}
        </group>
      </group>
    </group>
  );
}

// ---- science sites, path, range, navcam ---------------------------------------------------------
function Sites({ player, h, labels }: { player: Player; h: (x: number, z: number) => number; labels: boolean }) {
  const m = player.map, sites = player.state[3];
  return (
    <>
      {m.sites.map(([r, c, typ], i) => {
        const [x, z] = cellToWorld(m, r, c), y = h(x, z) + 0.02, st = sites[i];
        const color = st === SENT ? GOLD : st === HELD ? "#e9d39b" : "#ffffff";
        return (
          <group key={i} position={[x, y, z]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.3, 0.36, 48]} />
              <meshBasicMaterial color={color} transparent opacity={st === 0 ? 0.85 : 1} />
            </mesh>
            {st === SENT && (
              <mesh rotation={[-Math.PI / 2, 0, 0]}>
                <circleGeometry args={[0.24, 32]} />
                <meshBasicMaterial color={GOLD} transparent opacity={0.45} />
              </mesh>
            )}
            <mesh position={[0, 0.45, 0]}>
              <cylinderGeometry args={[0.008, 0.008, 0.9]} />
              <meshBasicMaterial color={color} transparent opacity={0.6} />
            </mesh>
            {labels && (
              <Html position={[0, 1.0, 0]} center distanceFactor={13} zIndexRange={[10, 0]}>
                <div className={`site-tag ${st === SENT ? "sent" : st === HELD ? "held" : ""}`}>
                  <b>{typ === 0 ? "Sulphur site" : "Element site"}</b>
                  <span>{typ === 0 ? "Needs the LIBS laser" : "Needs the APXS scan"}</span>
                  <span>{["Not sampled yet", "Sampled, not yet sent", "Sent to Earth"][st]}</span>
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </>
  );
}

function groundLine(h: (x: number, z: number) => number, pts: [number, number][], lift = 0.05) {
  const out: THREE.Vector3[] = [];
  for (let k = 0; k < pts.length - 1; k++) {
    const [x0, z0] = pts[k], [x1, z1] = pts[k + 1];
    const n = Math.max(2, Math.ceil(Math.hypot(x1 - x0, z1 - z0) * 6));
    for (let j = 0; j < n; j++) {
      const x = x0 + ((x1 - x0) * j) / n, z = z0 + ((z1 - z0) * j) / n;
      out.push(new THREE.Vector3(x, h(x, z) + lift, z));
    }
  }
  const [xl, zl] = pts[pts.length - 1];
  out.push(new THREE.Vector3(xl, h(xl, zl) + lift, zl));
  return out;
}

function Path({ player, h }: { player: Player; h: (x: number, z: number) => number }) {
  const m = player.map;
  const pts = useMemo(() => {
    const cells = player.track.map(([r, c]) => cellToWorld(m, r, c));
    return cells.length > 1 ? groundLine(h, cells, 0.04) : null;
  }, [player.track.length, m, player.version]); // eslint-disable-line
  if (!pts) return null;
  return <Line points={pts} color={GOLD} lineWidth={3} transparent opacity={0.95} />;
}

function RangeOutline({ m, h }: { m: LunarMap; h: (x: number, z: number) => number }) {
  const [vx, vz] = cellToWorld(m, m.vikram[0], m.vikram[1]);
  const R = COMM_RADIUS + 0.5;
  const lo = (v: number, lim: number) => Math.max(-lim, v - R), hi = (v: number, lim: number) => Math.min(lim, v + R);
  const L = m.W / 2;
  const corners: [number, number][] = [[lo(vx, L), lo(vz, L)], [hi(vx, L), lo(vz, L)], [hi(vx, L), hi(vz, L)], [lo(vx, L), hi(vz, L)], [lo(vx, L), lo(vz, L)]];
  const pts = useMemo(() => groundLine(h, corners, 0.03), [m]); // eslint-disable-line
  return <Line points={pts} color="#e9e7e2" lineWidth={1.2} dashed dashSize={0.18} gapSize={0.12} transparent opacity={0.55} />;
}

function NavCam({ player, h }: { player: Player; h: (x: number, z: number) => number }) {
  const ref = useRef<THREE.Group>(null!);
  const m = player.map;
  const pts = useMemo(() => {
    const R = VIEW + 0.5;
    return [[-R, -R], [R, -R], [R, R], [-R, R], [-R, -R]].map(([a, b]) => new THREE.Vector3(a, 0, b));
  }, []);
  useFrame(() => {
    const [r, c] = player.cur ? player.cur.prev : player.state;
    const [x, z] = cellToWorld(m, r, c);
    ref.current.position.set(x, h(x, z) + 0.35, z);
  });
  return (
    <group ref={ref}>
      <Line points={pts} color="#8fd3e8" lineWidth={1.5} dashed dashSize={0.25} gapSize={0.15} />
    </group>
  );
}

// ---- action effects -----------------------------------------------------------------------------
function Effects({ player, h }: { player: Player; h: (x: number, z: number) => number }) {
  const s = player.cur, m = player.map;
  const ring = useRef<THREE.Mesh>(null!);
  useFrame(() => {
    if (ring.current && player.cur) {
      const k = player.t;
      ring.current.scale.setScalar(0.3 + k * 1.6);
      (ring.current.material as THREE.MeshBasicMaterial).opacity = 1 - k;
    }
  });
  if (!s) return null;
  const [x, z] = cellToWorld(m, s.prev[0], s.prev[1]);
  const y = h(x, z);
  if (s.ev === "libs_ok" || (s.ev === "wrong_instrument" && s.a === 8)) {
    return (
      <group position={[x, y, z]}>
        <Line points={[[0.2, 0.42, 0], [0.32, 0.02, 0.05]]} color="#ff3b2a" lineWidth={3} />
        <pointLight position={[0.32, 0.08, 0.05]} color="#ff6a3a" intensity={2.5} distance={1.6} />
      </group>
    );
  }
  if (s.ev === "apxs_ok" || (s.ev === "wrong_instrument" && s.a === 9)) {
    return (
      <mesh ref={ring} position={[x, y + 0.04, z]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.28, 0.33, 48]} />
        <meshBasicMaterial color="#8fb4ff" transparent />
      </mesh>
    );
  }
  if (s.ev === "tx_ok" || s.ev === "tx_fail") {
    const [vx, vz] = cellToWorld(m, m.vikram[0], m.vikram[1]);
    const ok = s.ev === "tx_ok";
    const end: [number, number, number] = ok ? [vx, h(vx, vz) + 0.9, vz] : [x + (vx - x) * 0.4, y + 1.2, z + (vz - z) * 0.4];
    return (
      <QuadraticBezierLine start={[x, y + 0.45, z]} end={end} mid={[(x + end[0]) / 2, y + 2.2, (z + end[2]) / 2]}
        color={ok ? GOLD : "#d8452e"} lineWidth={2} dashed dashScale={8} />
    );
  }
  return null;
}

// ---- cameras --------------------------------------------------------------------------------------
function CameraRig({ mode, player }: { mode: "hero" | "control"; player: Player }) {
  const { camera, size } = useThree();
  const focus = useRef(new THREE.Vector3());
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    if (mode === "control") { cam.clearViewOffset(); camera.position.set(0, 16.5, 11.5); camera.lookAt(0, 0, 0.6); }
    else if (size.width > 900) cam.setViewOffset(size.width, size.height, -size.width * 0.2, 0, size.width, size.height);
    else cam.clearViewOffset();
  }, [mode, camera, size]);
  useFrame(({ clock }, dt) => {
    if (mode !== "hero") return;
    // hero: a slow orbit that keeps the driving rover in frame
    const p = player.pose(), m = player.map;
    const want = new THREE.Vector3(p.c - m.W / 2 + 0.5, 0, p.r - m.H / 2 + 0.5);
    focus.current.lerp(want, 1 - Math.exp(-dt * 0.8));
    const t = clock.elapsedTime * 0.05;
    camera.position.set(focus.current.x + Math.cos(t + 0.6) * 6.5, 2.6 + Math.sin(t * 0.7) * 0.3, focus.current.z + Math.sin(t + 0.6) * 6.5);
    camera.lookAt(focus.current.x, 0.1, focus.current.z);
  });
  return mode === "control" ? (
    <OrbitControls makeDefault target={[0, 0, 0.6]} minDistance={6} maxDistance={24} maxPolarAngle={1.25} enableDamping />
  ) : null;
}
