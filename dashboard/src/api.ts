const API_BASE = '/api';

export interface StatusResponse {
  is_running: boolean;
  current_task: string | null;
  progress: number;
  error: string | null;
  has_results: boolean;
}

export interface StepMetric {
  step: number;
  sim_time: number;
  reward: number;
  total_reward: number;
  system_total_stopped?: number;
  system_total_waiting_time?: number;
  system_mean_waiting_time?: number;
  system_mean_speed?: number;
}

export interface SimulationResults {
  controller: string;
  total_reward: number;
  num_steps: number;
  final_info: Record<string, number>;
  step_data: StepMetric[];
}

export interface TrainingResults {
  controller: string;
  total_timesteps: number;
  training_time_seconds: number;
  episode_rewards: number[];
  step_metrics: Array<{
    timestep: number;
    reward: number;
    system_total_stopped?: number;
    system_mean_speed?: number;
  }>;
}

export interface ComparisonResults {
  fixed_time: SimulationResults;
  train: TrainingResults;
  rl_dqn: SimulationResults;
  comparison: {
    fixed_time: {
      total_reward: number;
      avg_reward_per_step: number;
      final_metrics: Record<string, number>;
    };
    rl_dqn: {
      total_reward: number;
      avg_reward_per_step: number;
      final_metrics: Record<string, number>;
      training_time: number;
      training_episodes: number;
    };
    improvement: {
      reward_pct: number;
    };
  };
}

export interface StreamEvent {
  is_running: boolean;
  progress: number;
  current_task: string | null;
  error: string | null;
  new_metrics: StepMetric[];
  done?: boolean;
}

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getStatus(): Promise<StatusResponse> {
  return fetchJSON(`${API_BASE}/status`);
}

export async function startFixedTime(numSeconds = 3600): Promise<void> {
  await fetchJSON(`${API_BASE}/fixed-time`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ num_seconds: numSeconds }),
  });
}

export async function startTraining(
  totalTimesteps = 20000,
  numSeconds = 3600
): Promise<void> {
  await fetchJSON(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ total_timesteps: totalTimesteps, num_seconds: numSeconds }),
  });
}

export async function startEvaluation(numSeconds = 3600): Promise<void> {
  await fetchJSON(`${API_BASE}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ num_seconds: numSeconds }),
  });
}

export async function startComparison(
  totalTimesteps = 20000,
  numSeconds = 3600
): Promise<void> {
  await fetchJSON(`${API_BASE}/comparison`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ total_timesteps: totalTimesteps, num_seconds: numSeconds }),
  });
}

export async function resetState(): Promise<void> {
  await fetchJSON(`${API_BASE}/reset`, { method: 'POST' });
}

export async function getResults(): Promise<ComparisonResults | SimulationResults> {
  return fetchJSON(`${API_BASE}/results`);
}

export function getCSVExportURL(controller: string): string {
  return `${API_BASE}/export/csv/${controller}`;
}

export function getReportURL(): string {
  return `${API_BASE}/export/report`;
}

export function subscribeToStream(
  onEvent: (event: StreamEvent) => void,
  onError?: (error: Event) => void
): EventSource {
  const es = new EventSource(`${API_BASE}/stream`);
  es.onmessage = (e) => {
    try {
      const data: StreamEvent = JSON.parse(e.data);
      onEvent(data);
      if (data.done) {
        es.close();
      }
    } catch {
      // ignore parse errors
    }
  };
  es.onerror = (e) => {
    onError?.(e);
    es.close();
  };
  return es;
}
