import { useCallback, useEffect, useRef, useState } from 'react';
import './index.css';
import {
  type ComparisonResults,
  type StatusResponse,
  type StepMetric,
  type StreamEvent,
  getCSVExportURL,
  getReportURL,
  getResults,
  getStatus,
  resetState,
  startComparison,
  startEvaluation,
  startFixedTime,
  startTraining,
  subscribeToStream,
} from './api';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
function cn(...classes: (string | false | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}

function formatNumber(n: number | undefined, decimals = 2) {
  if (n === undefined || n === null) return '—';
  return n.toFixed(decimals);
}

const TASK_LABELS: Record<string, string> = {
  fixed_time: 'Fixed-Time Baseline',
  training: 'RL Training (DQN)',
  evaluation: 'RL Evaluation',
  comparison: 'Full Comparison',
};

/* ------------------------------------------------------------------ */
/* Main App                                                            */
/* ------------------------------------------------------------------ */
export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<StepMetric[]>([]);
  const [results, setResults] = useState<ComparisonResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'results'>('dashboard');
  const esRef = useRef<EventSource | null>(null);

  // Poll status
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const s = await getStatus();
        setStatus(s);
        if (s.error) setError(s.error);
        if (s.has_results && !s.is_running) {
          try {
            const r = await getResults();
            if ('comparison' in r) setResults(r as ComparisonResults);
          } catch {
            /* no results yet */
          }
        }
      } catch {
        /* offline */
      }
    }, 2000);
    return () => clearInterval(poll);
  }, []);

  // Stream handler
  const startStream = useCallback(() => {
    esRef.current?.close();
    setLiveMetrics([]);
    const es = subscribeToStream(
      (event: StreamEvent) => {
        if (event.new_metrics?.length) {
          setLiveMetrics((prev) => [...prev, ...event.new_metrics].slice(-500));
        }
      },
      () => setError('Stream connection lost')
    );
    esRef.current = es;
  }, []);

  const handleAction = useCallback(
    async (action: () => Promise<void>) => {
      try {
        setError(null);
        setResults(null);
        await action();
        startStream();
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unknown error');
      }
    },
    [startStream]
  );

  const handleReset = useCallback(async () => {
    esRef.current?.close();
    setLiveMetrics([]);
    setResults(null);
    setError(null);
    await resetState();
    setStatus(null);
  }, []);

  const isRunning = status?.is_running ?? false;

  return (
    <div className="min-h-screen bg-surface">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-surface/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-accent-bright shadow-lg shadow-accent/20">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <polygon points="10 8 16 12 10 16 10 8"/>
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-text-primary">
                Smart Traffic Control
              </h1>
              <p className="text-xs text-text-muted">
                Reinforcement Learning × SUMO Simulation
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Status indicator */}
            <div className={cn(
              'flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium',
              isRunning
                ? 'bg-success-dim text-success'
                : 'bg-surface-card text-text-secondary'
            )}>
              <span className={cn(
                'h-2 w-2 rounded-full',
                isRunning ? 'bg-success animate-pulse' : 'bg-text-muted'
              )} />
              {isRunning ? TASK_LABELS[status?.current_task ?? ''] || 'Running' : 'Idle'}
            </div>

            {/* Tab buttons */}
            <div className="flex rounded-lg bg-surface-card p-1">
              {(['dashboard', 'results'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={cn(
                    'rounded-md px-3 py-1.5 text-xs font-medium transition-all',
                    activeTab === tab
                      ? 'bg-accent text-white shadow-sm'
                      : 'text-text-secondary hover:text-text-primary'
                  )}
                >
                  {tab === 'dashboard' ? 'Dashboard' : 'Results'}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        {/* Error banner */}
        {error && (
          <div className="mb-6 rounded-xl border border-danger/30 bg-danger-dim px-4 py-3 text-sm text-danger">
            <span className="font-medium">Error:</span> {error}
          </div>
        )}

        {activeTab === 'dashboard' ? (
          <DashboardView
            status={status}
            liveMetrics={liveMetrics}
            isRunning={isRunning}
            onAction={handleAction}
            onReset={handleReset}
          />
        ) : (
          <ResultsView results={results} />
        )}
      </main>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Dashboard View                                                      */
/* ------------------------------------------------------------------ */
interface DashboardProps {
  status: StatusResponse | null;
  liveMetrics: StepMetric[];
  isRunning: boolean;
  onAction: (action: () => Promise<void>) => void;
  onReset: () => void;
}

function DashboardView({ status, liveMetrics, isRunning, onAction, onReset }: DashboardProps) {
  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <section className="rounded-2xl border border-border bg-surface-card p-6">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-text-muted">
          Simulation Controls
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <ActionButton
            label="Fixed-Time"
            sublabel="Baseline"
            color="info"
            disabled={isRunning}
            onClick={() => onAction(() => startFixedTime(3600))}
          />
          <ActionButton
            label="Train RL"
            sublabel="DQN Agent"
            color="accent"
            disabled={isRunning}
            onClick={() => onAction(() => startTraining(20000, 3600))}
          />
          <ActionButton
            label="Evaluate"
            sublabel="Trained Model"
            color="success"
            disabled={isRunning}
            onClick={() => onAction(() => startEvaluation(3600))}
          />
          <ActionButton
            label="Compare"
            sublabel="Full Pipeline"
            color="warning"
            disabled={isRunning}
            onClick={() => onAction(() => startComparison(20000, 3600))}
          />
          <ActionButton
            label="Reset"
            sublabel="Clear State"
            color="danger"
            disabled={isRunning}
            onClick={onReset}
          />
          <ActionButton
            label="Export CSV"
            sublabel="Download"
            color="accent"
            disabled={false}
            onClick={() => window.open(getCSVExportURL('comparison'), '_blank')}
          />
        </div>
      </section>

      {/* Progress */}
      {isRunning && status && (
        <section className="rounded-2xl border border-border bg-surface-card p-6">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted">
              Progress
            </h2>
            <span className="text-sm font-mono text-accent-bright">
              {status.progress.toFixed(1)}%
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface">
            <div
              className="h-full rounded-full bg-gradient-to-r from-accent to-accent-bright transition-all duration-500"
              style={{ width: `${status.progress}%` }}
            />
          </div>
          <p className="mt-2 text-xs text-text-secondary">
            {TASK_LABELS[status.current_task ?? ''] || status.current_task}
          </p>
        </section>
      )}

      {/* Live Charts */}
      {liveMetrics.length > 0 && (
        <div className="grid gap-6 lg:grid-cols-2">
          <ChartCard title="Cumulative Reward">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={liveMetrics}>
                <defs>
                  <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6c5ce7" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#6c5ce7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d45" />
                <XAxis dataKey="sim_time" stroke="#5d6180" tick={{ fontSize: 11 }} />
                <YAxis stroke="#5d6180" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#1a1d2e', border: '1px solid #2a2d45', borderRadius: 8 }}
                  labelStyle={{ color: '#8b8fa8' }}
                />
                <Area
                  type="monotone"
                  dataKey="total_reward"
                  stroke="#6c5ce7"
                  fill="url(#rewardGrad)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Mean Speed (m/s)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={liveMetrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d45" />
                <XAxis dataKey="sim_time" stroke="#5d6180" tick={{ fontSize: 11 }} />
                <YAxis stroke="#5d6180" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#1a1d2e', border: '1px solid #2a2d45', borderRadius: 8 }}
                  labelStyle={{ color: '#8b8fa8' }}
                />
                <Line
                  type="monotone"
                  dataKey="system_mean_speed"
                  stroke="#00cec9"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Waiting Time">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={liveMetrics}>
                <defs>
                  <linearGradient id="waitGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ff6b6b" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#ff6b6b" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d45" />
                <XAxis dataKey="sim_time" stroke="#5d6180" tick={{ fontSize: 11 }} />
                <YAxis stroke="#5d6180" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#1a1d2e', border: '1px solid #2a2d45', borderRadius: 8 }}
                  labelStyle={{ color: '#8b8fa8' }}
                />
                <Area
                  type="monotone"
                  dataKey="system_mean_waiting_time"
                  stroke="#ff6b6b"
                  fill="url(#waitGrad)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Stopped Vehicles">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={liveMetrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2d45" />
                <XAxis dataKey="sim_time" stroke="#5d6180" tick={{ fontSize: 11 }} />
                <YAxis stroke="#5d6180" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#1a1d2e', border: '1px solid #2a2d45', borderRadius: 8 }}
                  labelStyle={{ color: '#8b8fa8' }}
                />
                <Line
                  type="monotone"
                  dataKey="system_total_stopped"
                  stroke="#fdcb6e"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      {/* Empty state */}
      {!isRunning && liveMetrics.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border py-20 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-surface-card">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6c5ce7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="7" height="7"/>
              <rect x="14" y="3" width="7" height="7"/>
              <rect x="14" y="14" width="7" height="7"/>
              <rect x="3" y="14" width="7" height="7"/>
            </svg>
          </div>
          <h3 className="mb-2 text-lg font-semibold text-text-primary">
            Ready to Simulate
          </h3>
          <p className="max-w-md text-sm text-text-secondary">
            Start a simulation to see live traffic metrics. Use <strong>Compare</strong> for a
            full Fixed-Time vs RL evaluation pipeline with exportable results.
          </p>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Results View                                                        */
/* ------------------------------------------------------------------ */
function ResultsView({ results }: { results: ComparisonResults | null }) {
  if (!results) {
    return (
      <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border py-20 text-center">
        <h3 className="mb-2 text-lg font-semibold text-text-primary">No Results Yet</h3>
        <p className="max-w-md text-sm text-text-secondary">
          Run a <strong>Compare</strong> simulation from the Dashboard to see detailed results here.
        </p>
      </div>
    );
  }

  const comp = results.comparison;
  const improvementColor = comp.improvement.reward_pct > 0 ? 'text-success' : 'text-danger';

  // Prepare comparison chart data
  const metricNames = ['total_reward', 'avg_reward_per_step'] as const;
  const barData = metricNames.map((m) => ({
    name: m.replace(/_/g, ' '),
    'Fixed-Time': comp.fixed_time[m],
    'RL (DQN)': comp.rl_dqn[m],
  }));

  // Per-step overlay data (downsample to 100 points)
  const fixedSteps = results.fixed_time.step_data;
  const rlSteps = results.rl_dqn.step_data;
  const overlayData: Array<Record<string, number>> = [];
  const maxLen = Math.max(fixedSteps.length, rlSteps.length);
  const step = Math.max(1, Math.floor(maxLen / 100));
  for (let i = 0; i < maxLen; i += step) {
    const row: Record<string, number> = { sim_time: (i + 1) * 5 };
    if (i < fixedSteps.length) {
      row['Fixed-Time Reward'] = fixedSteps[i].total_reward;
      row['Fixed-Time Wait'] = fixedSteps[i].system_mean_waiting_time ?? 0;
    }
    if (i < rlSteps.length) {
      row['RL Reward'] = rlSteps[i].total_reward;
      row['RL Wait'] = rlSteps[i].system_mean_waiting_time ?? 0;
    }
    overlayData.push(row);
  }

  // Training curve
  const trainCurve = results.train.episode_rewards.map((r, i) => ({
    episode: i + 1,
    reward: r,
  }));

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="RL Total Reward"
          value={formatNumber(comp.rl_dqn.total_reward, 1)}
          sub="DQN Agent"
          color="accent"
        />
        <StatCard
          label="Fixed Total Reward"
          value={formatNumber(comp.fixed_time.total_reward, 1)}
          sub="Baseline"
          color="info"
        />
        <StatCard
          label="Improvement"
          value={`${comp.improvement.reward_pct > 0 ? '+' : ''}${comp.improvement.reward_pct}%`}
          sub="Reward Change"
          color={comp.improvement.reward_pct > 0 ? 'success' : 'danger'}
        />
        <StatCard
          label="Training Time"
          value={`${comp.rl_dqn.training_time}s`}
          sub={`${comp.rl_dqn.training_episodes} episodes`}
          color="warning"
        />
      </div>

      {/* Detailed final metrics comparison */}
      <section className="rounded-2xl border border-border bg-surface-card p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted">
            Final Metrics Comparison
          </h2>
          <div className="flex gap-2">
            <a
              href={getCSVExportURL('comparison')}
              className="rounded-lg bg-surface-hover px-3 py-1.5 text-xs font-medium text-accent-bright hover:bg-accent hover:text-white transition-colors"
              download
            >
              ↓ CSV
            </a>
            <a
              href={getReportURL()}
              className="rounded-lg bg-surface-hover px-3 py-1.5 text-xs font-medium text-accent-bright hover:bg-accent hover:text-white transition-colors"
              download
            >
              ↓ Report
            </a>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-text-muted">
                <th className="pb-3 pr-4 font-medium">Metric</th>
                <th className="pb-3 pr-4 font-medium">Fixed-Time</th>
                <th className="pb-3 pr-4 font-medium">RL (DQN)</th>
                <th className="pb-3 font-medium">Δ</th>
              </tr>
            </thead>
            <tbody className="text-text-primary">
              {Object.keys(comp.fixed_time.final_metrics).map((key) => {
                const fv = comp.fixed_time.final_metrics[key];
                const rv = comp.rl_dqn.final_metrics[key] ?? 0;
                const delta = rv - fv;
                const isSpeedMetric = key.includes('speed');
                const betterColor = isSpeedMetric
                  ? delta > 0 ? 'text-success' : 'text-danger'
                  : delta < 0 ? 'text-success' : 'text-danger';
                return (
                  <tr key={key} className="border-b border-border/50">
                    <td className="py-2.5 pr-4 text-text-secondary">
                      {key.replace(/system_/g, '').replace(/_/g, ' ')}
                    </td>
                    <td className="py-2.5 pr-4 font-mono">{formatNumber(fv)}</td>
                    <td className="py-2.5 pr-4 font-mono">{formatNumber(rv)}</td>
                    <td className={cn('py-2.5 font-mono', betterColor)}>
                      {delta > 0 ? '+' : ''}{formatNumber(delta)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Comparison Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="Cumulative Reward — Overlay">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={overlayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d45" />
              <XAxis dataKey="sim_time" stroke="#5d6180" tick={{ fontSize: 11 }} />
              <YAxis stroke="#5d6180" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1a1d2e', border: '1px solid #2a2d45', borderRadius: 8 }}
                labelStyle={{ color: '#8b8fa8' }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="Fixed-Time Reward" stroke="#74b9ff" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="RL Reward" stroke="#6c5ce7" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Mean Waiting Time — Overlay">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={overlayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d45" />
              <XAxis dataKey="sim_time" stroke="#5d6180" tick={{ fontSize: 11 }} />
              <YAxis stroke="#5d6180" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1a1d2e', border: '1px solid #2a2d45', borderRadius: 8 }}
                labelStyle={{ color: '#8b8fa8' }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="Fixed-Time Wait" stroke="#74b9ff" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="RL Wait" stroke="#ff6b6b" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Training Reward Curve">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainCurve}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d45" />
              <XAxis dataKey="episode" stroke="#5d6180" tick={{ fontSize: 11 }} />
              <YAxis stroke="#5d6180" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1a1d2e', border: '1px solid #2a2d45', borderRadius: 8 }}
                labelStyle={{ color: '#8b8fa8' }}
              />
              <Line type="monotone" dataKey="reward" stroke="#a29bfe" strokeWidth={2} dot={{ r: 3, fill: '#a29bfe' }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Reward Summary">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d45" />
              <XAxis dataKey="name" stroke="#5d6180" tick={{ fontSize: 11 }} />
              <YAxis stroke="#5d6180" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1a1d2e', border: '1px solid #2a2d45', borderRadius: 8 }}
                labelStyle={{ color: '#8b8fa8' }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="Fixed-Time" fill="#74b9ff" radius={[4, 4, 0, 0]} />
              <Bar dataKey="RL (DQN)" fill="#6c5ce7" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Reusable Components                                                 */
/* ------------------------------------------------------------------ */
type ColorName = 'accent' | 'success' | 'warning' | 'danger' | 'info';

const COLOR_MAP: Record<ColorName, { bg: string; text: string; hover: string }> = {
  accent: { bg: 'bg-accent/10', text: 'text-accent-bright', hover: 'hover:bg-accent/20' },
  success: { bg: 'bg-success/10', text: 'text-success', hover: 'hover:bg-success/20' },
  warning: { bg: 'bg-warning/10', text: 'text-warning', hover: 'hover:bg-warning/20' },
  danger: { bg: 'bg-danger/10', text: 'text-danger', hover: 'hover:bg-danger/20' },
  info: { bg: 'bg-info/10', text: 'text-info', hover: 'hover:bg-info/20' },
};

function ActionButton({
  label,
  sublabel,
  color,
  disabled,
  onClick,
}: {
  label: string;
  sublabel: string;
  color: ColorName;
  disabled: boolean;
  onClick: () => void;
}) {
  const c = COLOR_MAP[color];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'rounded-xl border border-border px-4 py-3 text-left transition-all',
        c.bg,
        c.hover,
        disabled && 'cursor-not-allowed opacity-40'
      )}
    >
      <div className={cn('text-sm font-semibold', c.text)}>{label}</div>
      <div className="text-xs text-text-muted">{sublabel}</div>
    </button>
  );
}

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub: string;
  color: ColorName;
}) {
  const c = COLOR_MAP[color];
  return (
    <div className="rounded-2xl border border-border bg-surface-card p-5">
      <p className="text-xs font-medium uppercase tracking-wider text-text-muted">{label}</p>
      <p className={cn('mt-2 text-2xl font-bold', c.text)}>{value}</p>
      <p className="mt-1 text-xs text-text-secondary">{sub}</p>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-border bg-surface-card p-5">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-text-muted">
        {title}
      </h3>
      {children}
    </div>
  );
}
