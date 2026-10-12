export interface MapData {
  terrain: number[][];
  sites: [number, number, number][]; // row, col, type (0 = LIBS, 1 = APXS)
  vikram: [number, number];
  start: [number, number];
}

/** One recorded step: state AFTER the step, then action, reward, event. */
export type Frame = [number, number, number, number[], number, number, number, string];

export interface Episode {
  label: string;
  agent: string;
  stage: string;
  frames: Frame[];
  map: MapData | null;
}

export interface EvalRow {
  return_mean: number;
  return_std_seeds: number;
  success_rate: number;
  sent_mean: number;
  crater_rate: number;
  battery_rate: number;
  night_rate: number;
  steps_mean: number;
  train_time_s?: number;
  episodes?: number;
}

export interface Curve { x: number[]; y: number[] }

export interface PragyanData {
  map: MapData;
  consts: Record<string, number>;
  actions: string[];
  episodes: Record<string, Episode>;
  dqn: {
    layers: { W: number[][]; b: number[] }[];
    checks: { state: [number, number, number, number[], number]; action: number; features: number[] }[];
  };
  table: Record<string, EvalRow>;
  generalization: Record<string, Record<"fixed" | "perturbed" | "random", EvalRow>>;
  sweeps: Record<string, { return_mean: number; success_rate: number; train_time_s: number }>;
  valueIteration: { n_reachable_states: number; sweeps: number; V_start: number; plan_time: number };
  curves: Record<string, Record<"returns" | "success" | "steps" | "sent", Curve>>;
}
