/** Procedural lunar terrain built from the 12x12 mission grid. 1 world unit = 1 grid cell (~1 m). */
import * as THREE from "three";
import { CRATER, SHADOW, SLOPE, type LunarMap } from "../sim/env";

export const cellToWorld = (m: LunarMap, r: number, c: number): [number, number] =>
  [c - m.W / 2 + 0.5, r - m.H / 2 + 0.5];
export const worldToCell = (m: LunarMap, x: number, z: number): [number, number] =>
  [Math.floor(z + m.H / 2), Math.floor(x + m.W / 2)];

// ---- value noise ---------------------------------------------------------------
function hash(x: number, y: number) {
  let h = (x * 374761393 + y * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function noise(x: number, y: number) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
export function fbm(x: number, y: number, oct = 4) {
  let s = 0, amp = 0.5, f = 1;
  for (let i = 0; i < oct; i++) { s += amp * noise(x * f, y * f); amp *= 0.5; f *= 2.03; }
  return s;
}
const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

// decorative craters outside the mission grid (scenery only)
const SCENIC: [number, number, number, number][] = [
  [-15, -9, 5.5, 1.2], [17, 5, 7, 1.5], [3, -21, 6, 1.3], [-20, 14, 8, 1.6], [12, 19, 4.5, 1],
  [-9, 22, 3.5, 0.8], [24, -14, 6, 1.2], [-26, -4, 4, 0.9],
];

function bowl(d: number, R: number, depth: number) {
  let h = 0;
  if (d < R) h -= depth * (1 - (d / R) ** 2);
  h += depth * 0.22 * Math.exp(-(((d - R) / (R * 0.22)) ** 2));
  return h;
}

export interface Hazards { craters: [number, number][]; slopes: [number, number][]; shadows: [number, number][] }

export function hazards(m: LunarMap): Hazards {
  const hz: Hazards = { craters: [], slopes: [], shadows: [] };
  for (let r = 0; r < m.H; r++) for (let c = 0; c < m.W; c++) {
    const t = m.terrain[r][c], p = cellToWorld(m, r, c);
    if (t === CRATER) hz.craters.push(p);
    else if (t === SLOPE) hz.slopes.push(p);
    else if (t === SHADOW) hz.shadows.push(p);
  }
  return hz;
}

/** shadow-basin weight 0..1 (smooth union of the permanently shadowed cells) */
function shadowWeight(hz: Hazards, x: number, z: number) {
  let d = 9;   // distance to the union of the shadowed cells (0 inside any of them)
  for (const [sx, sz] of hz.shadows) d = Math.min(d, Math.max(Math.abs(x - sx) - 0.5, Math.abs(z - sz) - 0.5, 0));
  return smooth(0.45, 0, d);
}

export function heightAt(m: LunarMap, hz: Hazards, x: number, z: number) {
  const half = m.W / 2;
  const edge = Math.max(Math.abs(x), Math.abs(z)) - half;          // >0 outside the mission grid
  const out = smooth(0, 10, edge);
  let h = (fbm(x * 0.9 + 3, z * 0.9 - 7) - 0.5) * 0.08;              // fine regolith undulation
  h += out * (fbm(x * 0.07, z * 0.07, 5) - 0.42) * 7;               // distant highlands
  for (const [cx, cz, R, depth] of SCENIC) h += bowl(Math.hypot(x - cx, z - cz), R, depth);
  for (const [cx, cz] of hz.craters) h += bowl(Math.hypot(x - cx, z - cz), 0.47, 0.42);
  for (const [cx, cz] of hz.slopes) h += 0.24 * Math.exp(-((x - cx) ** 2 + (z - cz) ** 2) / (2 * 0.33 ** 2));
  h -= 0.2 * shadowWeight(hz, x, z);
  return h;
}

export function buildTerrain(m: LunarMap, size = 56, segs = 448) {
  const hz = hazards(m);
  const g = new THREE.PlaneGeometry(size, size, segs, segs);
  g.rotateX(-Math.PI / 2);
  const pos = g.attributes.position as THREE.BufferAttribute;
  const col = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    pos.setY(i, heightAt(m, hz, x, z));
    const n = fbm(x * 2.1, z * 2.1, 3);
    let g0 = 0.5 + (n - 0.5) * 0.16 + (fbm(x * 0.35, z * 0.35, 3) - 0.5) * 0.1;
    const sw = shadowWeight(hz, x, z);
    g0 *= 1 - 0.55 * sw;
    col[i * 3] = g0 * (1 - 0.05 * sw);
    col[i * 3 + 1] = g0 * (1 - 0.02 * sw);
    col[i * 3 + 2] = g0 * (1 + 0.1 * sw);
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.computeVertexNormals();
  return { geometry: g, hz };
}
