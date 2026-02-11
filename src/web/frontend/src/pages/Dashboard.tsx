import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import {
  Cpu,
  HardDrive,
  MemoryStick,
  Wifi,
  WifiOff,
  Loader2,
  Bot,
  Calendar,
  CheckCircle2,
  XCircle,
  Activity,
  Zap,
  Server,
  Database,
  Plug,
  DollarSign,
  Download,
} from 'lucide-react';
import { hudApi, systemApi, usageApi, exportApi } from '../services/api';

function VitalsCard({
  icon: Icon,
  label,
  value,
  unit,
  sublabel,
  status,
}: {
  icon: React.ElementType;
  label: string;
  value: number;
  unit?: string;
  sublabel?: string;
  status?: 'good' | 'warn' | 'critical';
}) {
  const statusColors = {
    good: { bar: 'bg-emerald-500', text: 'text-emerald-400', glow: 'shadow-emerald-500/20' },
    warn: { bar: 'bg-amber-500', text: 'text-amber-400', glow: 'shadow-amber-500/20' },
    critical: { bar: 'bg-rose-500', text: 'text-rose-400', glow: 'shadow-rose-500/20' },
  };
  const s = status ?? (value > 80 ? 'critical' : value > 60 ? 'warn' : 'good');
  const colors = statusColors[s];

  return (
    <div className="group relative overflow-hidden rounded-2xl dashboard-card p-5 transition-all duration-300">
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" style={{ opacity: 0.5 }} />
      <div className="relative flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2.5 rounded-xl bg-theme-muted ${s === 'good' ? 'group-hover:bg-emerald-500/10' : ''}`}>
            <Icon className={`w-5 h-5 ${colors.text}`} />
          </div>
          <div>
            <p className="text-xs font-medium text-theme-secondary uppercase tracking-wider">{label}</p>
            <p className={`text-2xl font-bold tracking-tight ${colors.text}`}>
              {value}{unit ?? '%'}
            </p>
            {sublabel && <p className="text-xs text-theme-secondary mt-0.5">{sublabel}</p>}
          </div>
        </div>
      </div>
      {unit === '%' && (
        <div className="relative mt-4 h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'color-mix(in srgb, var(--bg-card) 70%, transparent)' }}>
          <div
            className={`h-full rounded-full ${colors.bar} transition-all duration-700 ease-out`}
            style={{ width: `${Math.min(value, 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

function SystemVitals() {
  const { data: vitals, isLoading } = useQuery({
    queryKey: ['hud-vitals'],
    queryFn: async () => {
      const res = await hudApi.getVitals();
      return res.data;
    },
    refetchInterval: 5000,
  });

  if (isLoading || !vitals) {
    return (
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="rounded-2xl card-theme p-5 animate-pulse border-theme"
          >
            <div className="h-4 w-16 rounded mb-3 bg-theme-muted" />
            <div className="h-8 w-20 rounded bg-theme-muted" />
          </div>
        ))}
      </div>
    );
  }

  const cpuVal = typeof vitals.cpu_percent === 'number' ? vitals.cpu_percent : 0;
  const memVal = typeof vitals.memory_percent === 'number' ? vitals.memory_percent : 0;
  const diskVal = typeof vitals.disk_percent === 'number' ? vitals.disk_percent : 0;
  const llm = vitals.llm ?? {};
  const anyLlmAvailable = llm.local === true || llm.cloud === true || llm.gemini === true || llm.openrouter === true || llm.nvidia === true;
  const allLlmUnknown = [llm.local, llm.cloud, llm.gemini, llm.openrouter, llm.nvidia].every((v) => v === undefined || v === null);
  const llmLabel = allLlmUnknown
    ? 'Checking…'
    : (llm.local && llm.cloud) ? 'Hybrid'
    : llm.local ? 'Local' : llm.cloud ? 'Cloud'
    : llm.gemini ? 'Gemini' : llm.openrouter ? 'OpenRouter'
    : llm.nvidia ? 'NVIDIA' : 'Offline';
  const llmSublabel = allLlmUnknown
    ? '—'
    : [llm.local && 'Local', llm.cloud && 'Cloud', llm.gemini && 'Gemini', llm.openrouter && 'OpenRouter', llm.nvidia && 'NVIDIA']
        .filter(Boolean)
        .map((l, i) => (i > 0 ? ` · ${l} ✓` : `${l} ✓`))
        .join('') || 'Offline';

  return (
    <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
      <VitalsCard
        icon={Cpu}
        label="CPU"
        value={cpuVal}
        status={cpuVal > 80 ? 'critical' : cpuVal > 60 ? 'warn' : 'good'}
      />
      <VitalsCard
        icon={MemoryStick}
        label="Memory"
        value={memVal}
        sublabel={vitals.memory_total_gb != null ? `${vitals.memory_used_gb ?? 0} / ${vitals.memory_total_gb} GB` : undefined}
        status={memVal > 80 ? 'critical' : memVal > 60 ? 'warn' : 'good'}
      />
      <VitalsCard
        icon={HardDrive}
        label="Disk"
        value={diskVal}
        sublabel={vitals.disk_total_gb != null ? `${vitals.disk_used_gb ?? 0} / ${vitals.disk_total_gb} GB` : undefined}
        status={diskVal > 90 ? 'critical' : diskVal > 75 ? 'warn' : 'good'}
      />
      <div className="rounded-2xl dashboard-card p-5 transition-all duration-300 ">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-theme-muted">
            {anyLlmAvailable ? (
              <Wifi className="w-5 h-5 text-emerald-400" />
            ) : (
              <WifiOff className="w-5 h-5 text-amber-400" />
            )}
          </div>
          <div>
            <p className="text-xs font-medium text-theme-secondary uppercase tracking-wider">LLM</p>
            <p className="text-sm font-semibold text-theme-primary mt-0.5">{llmLabel}</p>
            <p className="text-xs text-theme-secondary mt-0.5">{llmSublabel}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function AgentPanel() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['hud-agents-full'],
    queryFn: async () => {
      const res = await hudApi.getAllAgentsFull();
      return res.data;
    },
    refetchInterval: 1500,
  });

  const agents = Array.isArray(data?.agents) ? data.agents : [];

  if (isError) {
    return (
      <div className="rounded-2xl dashboard-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-theme-muted">
            <Bot className="w-5 h-5 text-theme-secondary" />
          </div>
          <div>
            <h3 className="font-semibold text-theme-primary">Active Agents</h3>
            <p className="text-xs text-theme-secondary">Real-time bot status</p>
          </div>
        </div>
        <div className="rounded-xl border border-theme card-theme p-6 text-center">
          <p className="text-sm text-theme-secondary">Unable to load agents</p>
        </div>
      </div>
    );
  }

  if (isLoading && agents.length === 0) {
    return (
      <div className="rounded-2xl dashboard-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-theme-muted">
            <Bot className="w-5 h-5 text-theme-secondary" />
          </div>
          <div>
            <h3 className="font-semibold text-theme-primary">Active Agents</h3>
            <p className="text-xs text-theme-secondary">Real-time bot status</p>
          </div>
        </div>
        <div className="flex items-center justify-center py-12 gap-2 text-theme-secondary">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading agents...
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl dashboard-card p-6 transition-all duration-300 ">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-theme-muted">
            <Bot className="w-5 h-5 text-theme-secondary" />
          </div>
          <div>
            <h3 className="font-semibold text-theme-primary">Active Agents</h3>
            <p className="text-xs text-theme-secondary">{agents.length} {agents.length === 1 ? 'agent' : 'agents'} running</p>
          </div>
        </div>
        {agents.length > 0 && (
          <span className="flex items-center gap-1.5 text-xs text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            Live
          </span>
        )}
      </div>

      {agents.length === 0 ? (
        <div className="rounded-xl border border-dashed border-theme card-theme p-8 text-center">
          <Bot className="w-10 h-10 text-theme-secondary mx-auto mb-3" />
          <p className="text-sm font-medium text-theme-secondary">No agents running</p>
          <p className="text-xs text-theme-secondary mt-1 max-w-[240px] mx-auto">
            Agents will appear here when tasks are running
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {agents.slice(0, 5).map((a: {
            id: string;
            description: string;
            status: string;
            bots?: { id: string; name: string; source: string; status: string; output?: string; error?: string }[];
          }) => (
            <div
              key={a.id}
              className="rounded-xl border border-theme card-theme p-4 transition-colors hover:bg-theme-muted"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-theme-primary truncate max-w-[220px]">{a.description}</span>
                <span className={`text-xs font-medium px-2.5 py-1 rounded-lg flex items-center gap-1.5 shrink-0 ${
                  a.status === 'running' ? 'bg-blue-500/15 text-blue-400 border border-blue-500/20' :
                  a.status === 'completed' ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20' :
                  a.status === 'failed' ? 'bg-rose-500/15 text-rose-400 border border-rose-500/20' :
                  'bg-slate-500/15 text-theme-secondary border border-slate-500/20'
                }`}>
                  {a.status === 'running' && <Activity className="w-3.5 h-3.5 animate-pulse" />}
                  {a.status === 'completed' && <CheckCircle2 className="w-3.5 h-3.5" />}
                  {a.status === 'failed' && <XCircle className="w-3.5 h-3.5" />}
                  {a.status}
                </span>
              </div>
              {a.bots && a.bots.length > 0 && (
                <div className="space-y-2 mt-3 pt-3 border-t border-theme">
                  {a.bots.map((b: { id: string; name: string; source: string; status: string; output?: string; error?: string }) => (
                    <div key={b.id} className="flex items-center gap-3 text-xs">
                      <span className={`w-2 h-2 rounded-full shrink-0 ${
                        b.status === 'running' ? 'bg-blue-400 animate-pulse' :
                        b.status === 'completed' ? 'bg-emerald-400' :
                        b.status === 'failed' ? 'bg-rose-400' : 'bg-slate-500'
                      }`} />
                      <span className="text-theme-secondary w-20 shrink-0">{b.name}</span>
                      <span className={`flex-1 truncate min-w-0 ${
                        b.status === 'completed' ? 'text-theme-secondary' : 'text-theme-secondary'
                      }`}>
                        {b.status === 'running' ? 'Searching...' : (b.output || b.error || '—')}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function UsageCostWidget() {
  const { data, isLoading } = useQuery({
    queryKey: ['usage'],
    queryFn: async () => {
      const res = await usageApi.getStats();
      return res.data;
    },
    refetchInterval: 30000,
  });

  if (isLoading || !data) {
    return (
      <div className="rounded-2xl dashboard-card p-5 animate-pulse">
        <div className="h-4 w-24 bg-theme-muted rounded mb-3" />
        <div className="h-6 w-16 bg-theme-muted rounded" />
      </div>
    );
  }

  const totalCalls = Number(data?.total_calls) || 0;
  const totalTokens = Number(data?.total_tokens) || 0;
  const cost = Number(data?.cost_estimate_usd) || 0;
  const latency = Number(data?.avg_latency_ms) || 0;

  return (
    <div className="rounded-2xl dashboard-card p-5 transition-all duration-300 ">
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 rounded-lg bg-theme-muted">
          <DollarSign className="w-5 h-5 text-emerald-400" />
        </div>
        <div>
          <h3 className="font-semibold text-theme-primary">Usage & Cost</h3>
          <p className="text-xs text-theme-secondary">LLM usage tracking</p>
        </div>
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-theme-secondary">API calls</span>
          <span className="font-medium text-theme-primary tabular-nums">{totalCalls}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-theme-secondary">Tokens</span>
          <span className="font-medium text-theme-primary tabular-nums">{totalTokens.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-theme-secondary">Est. cost</span>
          <span className="font-medium text-emerald-400 tabular-nums">${cost.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-theme-secondary">Avg latency</span>
          <span className="font-medium text-theme-primary tabular-nums">{Math.round(latency)}ms</span>
        </div>
      </div>
    </div>
  );
}

function ExportButton() {
  const [downloading, setDownloading] = useState(false);
  const handleExport = async (type: 'all' | 'conversations' | 'audit') => {
    setDownloading(true);
    try {
      const res = await exportApi.export(type);
      const blob = res.data instanceof Blob ? res.data : new Blob([JSON.stringify(res.data)]);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `aria-export-${type}-${Date.now()}.${type === 'audit' ? 'log' : 'json'}`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="rounded-2xl dashboard-card p-5 transition-all duration-300 ">
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 rounded-lg bg-theme-muted">
          <Download className="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <h3 className="font-semibold text-theme-primary">Data Export</h3>
          <p className="text-xs text-theme-secondary">Export conversations & audit</p>
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        {(['all', 'conversations', 'audit'] as const).map((t) => (
          <button
            key={t}
            onClick={() => handleExport(t)}
            disabled={downloading}
            className="px-3 py-1.5 rounded-lg bg-theme-muted hover:bg-theme-muted text-sm text-theme-primary transition-all disabled:opacity-50"
          >
            {downloading ? '...' : t === 'all' ? 'Full' : t === 'conversations' ? 'Chats' : 'Audit'}
          </button>
        ))}
      </div>
    </div>
  );
}

function QuickStats() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['system-status'],
    queryFn: async () => {
      const res = await systemApi.getStatus();
      return res.data;
    },
    refetchInterval: 15000,
  });

  const skills = data?.skills;
  const stats = [
    { icon: Zap, label: 'Skills', value: typeof skills?.enabled_skills === 'number' ? skills.enabled_skills : 0 },
    { icon: Calendar, label: 'Scheduled', value: typeof data?.scheduled_jobs === 'number' ? data.scheduled_jobs : 0 },
    { icon: Database, label: 'Vector docs', value: typeof data?.vector_memory?.document_count === 'number' ? data.vector_memory.document_count : 0 },
    { icon: Plug, label: 'Plugins', value: typeof data?.plugins === 'number' ? data.plugins : 0 },
  ];

  return (
    <div className="rounded-2xl dashboard-card p-6 transition-all duration-300 ">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-theme-muted">
          <Server className="w-5 h-5 text-theme-secondary" />
        </div>
        <div>
          <h3 className="font-semibold text-theme-primary">Quick Stats</h3>
          <p className="text-xs text-theme-secondary">System overview</p>
        </div>
      </div>
      {isError ? (
        <p className="text-sm text-theme-secondary">Unable to load stats</p>
      ) : isLoading && !data ? (
        <div className="flex items-center gap-2 text-theme-secondary">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-sm">Loading…</span>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          {stats.map(({ icon: Icon, label, value }) => (
            <div key={label} className="flex items-center gap-3 p-3 rounded-xl card-theme">
              <Icon className="w-4 h-4 text-theme-secondary" />
              <div>
                <p className="text-xs text-theme-secondary">{label}</p>
                <p className="text-lg font-semibold text-theme-primary tabular-nums">{value}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ConversationTimeline() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['hud-timeline'],
    queryFn: async () => {
      const res = await hudApi.getTimeline();
      return res.data;
    },
    refetchInterval: 10000,
  });

  const events = Array.isArray(data?.events) ? data.events : [];

  if (isError) {
    return (
      <div className="rounded-2xl dashboard-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-theme-muted">
            <Calendar className="w-5 h-5 text-theme-secondary" />
          </div>
          <div>
            <h3 className="font-semibold text-theme-primary">Today's Activity</h3>
            <p className="text-xs text-theme-secondary">Conversation timeline</p>
          </div>
        </div>
        <div className="rounded-xl border border-theme card-theme p-6 text-center">
          <p className="text-sm text-theme-secondary">Unable to load activity</p>
        </div>
      </div>
    );
  }

  if (isLoading && events.length === 0) {
    return (
      <div className="rounded-2xl dashboard-card p-6 animate-pulse">
        <div className="h-5 w-32 bg-theme-muted rounded mb-4" />
        <div className="space-y-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-10 bg-theme-muted rounded" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl dashboard-card p-6 transition-all duration-300 ">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-theme-muted">
          <Calendar className="w-5 h-5 text-theme-secondary" />
        </div>
        <div>
          <h3 className="font-semibold text-theme-primary">Today's Activity</h3>
          <p className="text-xs text-theme-secondary">Conversation timeline</p>
        </div>
      </div>
      {events.length === 0 ? (
        <div className="rounded-xl border border-dashed border-theme card-theme p-6 text-center">
          <Calendar className="w-8 h-8 text-theme-secondary mx-auto mb-2" />
          <p className="text-sm text-theme-secondary">No activity today</p>
        </div>
      ) : (
        <ul className="space-y-2">
          {events.slice(0, 10).map((e: { channel: string; user_id: string; updated_at: string; message_count: number }, i: number) => (
            <li
              key={i}
              className="flex items-center justify-between py-2.5 px-3 rounded-lg card-theme hover:bg-theme-muted transition-colors"
            >
              <span className="text-sm text-theme-primary font-medium">{e.channel} / {e.user_id}</span>
              <span className="text-xs text-theme-secondary">{e.message_count} msgs</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function Dashboard() {
  return (
    <div className="min-h-screen relative">
      {/* Subtle grid background */}
      <div
        className="absolute inset-0 opacity-[0.02] pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '48px 48px',
        }}
      />
      {/* Hero header */}
      <div className="relative overflow-hidden border-b border-theme z-10" style={{ background: 'linear-gradient(to bottom, color-mix(in srgb, var(--bg-card) 50%, transparent), transparent)' }}>
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(59,130,246,0.08),transparent)]" />
        <div className="relative px-8 py-10 z-10">
          <h1 className="text-3xl font-bold text-theme-primary tracking-tight">
            Dashboard
          </h1>
          <p className="text-theme-secondary mt-1 max-w-2xl">
            Real-time system vitals, active agents and usage tracking.
          </p>
        </div>
      </div>

      <div className="relative p-8 space-y-8 z-10">
        {/* System Vitals */}
        <section>
          <h2 className="text-sm font-semibold text-theme-secondary uppercase tracking-wider mb-4">System Vitals</h2>
          <SystemVitals />
        </section>

        {/* Agent panel + Overview */}
        <section className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <h2 className="text-sm font-semibold text-theme-secondary uppercase tracking-wider">Active Agents</h2>
            <AgentPanel />
          </div>
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-theme-secondary uppercase tracking-wider">Overview</h2>
            <UsageCostWidget />
            <ExportButton />
            <QuickStats />
          </div>
        </section>

        {/* Activity timeline */}
        <section>
          <h2 className="text-sm font-semibold text-theme-secondary uppercase tracking-wider mb-4">Activity</h2>
          <ConversationTimeline />
        </section>
      </div>
    </div>
  );
}
