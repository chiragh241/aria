import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
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
  Search,
  MapPin,
  Zap,
  Server,
  Database,
  Plug,
  DollarSign,
  Download,
} from 'lucide-react';
import { hudApi, systemApi, agentsApi, usageApi, exportApi } from '../services/api';

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
    <div className="group relative overflow-hidden rounded-2xl bg-slate-800/50 border border-white/[0.06] p-5 transition-all duration-300 hover:border-white/[0.1] hover:bg-slate-800/70 hover:shadow-xl hover:shadow-slate-900/50">
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" />
      <div className="relative flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2.5 rounded-xl bg-slate-700/50 ${s === 'good' ? 'group-hover:bg-emerald-500/10' : ''}`}>
            <Icon className={`w-5 h-5 ${colors.text}`} />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">{label}</p>
            <p className={`text-2xl font-bold tracking-tight ${colors.text}`}>
              {value}{unit ?? '%'}
            </p>
            {sublabel && <p className="text-xs text-slate-600 mt-0.5">{sublabel}</p>}
          </div>
        </div>
      </div>
      {unit === '%' && (
        <div className="relative mt-4 h-1.5 rounded-full bg-slate-700/80 overflow-hidden">
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
            className="rounded-2xl bg-slate-800/30 border border-white/[0.04] p-5 animate-pulse"
          >
            <div className="h-4 w-16 bg-slate-700/50 rounded mb-3" />
            <div className="h-8 w-20 bg-slate-700/50 rounded" />
          </div>
        ))}
      </div>
    );
  }

  const cpuVal = vitals.cpu_percent ?? 0;
  const memVal = vitals.memory_percent ?? 0;
  const diskVal = vitals.disk_percent ?? 0;

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
        sublabel={`${vitals.memory_used_gb ?? 0} / ${vitals.memory_total_gb ?? 0} GB`}
        status={memVal > 80 ? 'critical' : memVal > 60 ? 'warn' : 'good'}
      />
      <VitalsCard
        icon={HardDrive}
        label="Disk"
        value={diskVal}
        status={diskVal > 90 ? 'critical' : diskVal > 75 ? 'warn' : 'good'}
      />
      <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-5 transition-all duration-300 hover:border-white/[0.1] hover:bg-slate-800/70">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-slate-700/50">
            {vitals.llm?.local || vitals.llm?.cloud ? (
              <Wifi className="w-5 h-5 text-emerald-400" />
            ) : (
              <WifiOff className="w-5 h-5 text-amber-400" />
            )}
          </div>
          <div>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">LLM</p>
            <p className="text-sm font-semibold text-slate-200 mt-0.5">
              {vitals.llm?.local && vitals.llm?.cloud ? 'Hybrid' : vitals.llm?.local ? 'Local' : vitals.llm?.cloud ? 'Cloud' : 'Offline'}
            </p>
            <p className="text-xs text-slate-600 mt-0.5">
              Local {vitals.llm?.local ? '✓' : '✗'} · Cloud {vitals.llm?.cloud ? '✓' : '✗'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function QuickActionCard({
  icon: Icon,
  label,
  placeholder,
  onRun,
  iconColor,
  buttonClass,
}: {
  icon: React.ElementType;
  label: string;
  placeholder: string;
  onRun: (q: string) => Promise<unknown>;
  iconColor: string;
  buttonClass: string;
}) {
  const [query, setQuery] = useState('');
  const runMutation = useMutation({
    mutationFn: (q: string) => onRun(q),
  });
  const handleSubmit = () => {
    if (!query.trim()) return;
    runMutation.mutate(query);
    setQuery('');
  };

  return (
    <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-4 transition-all duration-300 hover:border-white/[0.1]">
      <div className="flex items-center gap-2 mb-3">
        <Icon className={`w-4 h-4 ${iconColor}`} />
        <span className="text-sm font-semibold text-slate-300">{label}</span>
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          className="flex-1 px-4 py-2.5 rounded-xl bg-slate-900/60 border border-white/[0.06] text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500/40 focus:ring-1 focus:ring-blue-500/20 transition-all"
        />
        <button
          onClick={handleSubmit}
          disabled={!query.trim() || runMutation.isPending}
          className={`px-4 py-2.5 rounded-xl font-medium text-sm flex items-center gap-2 transition-all disabled:opacity-50 ${buttonClass}`}
        >
          {runMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          Run
        </button>
      </div>
    </div>
  );
}

function ResearchLauncher() {
  const queryClient = useQueryClient();
  return (
    <QuickActionCard
      icon={Search}
      label="Research"
      placeholder="Research a topic (Reddit, X, Web)..."
      iconColor="text-blue-400"
      buttonClass="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400"
      onRun={async (q) => {
        const res = await agentsApi.run(q, 'research');
        queryClient.invalidateQueries({ queryKey: ['hud-agents-full'] });
        return res.data;
      }}
    />
  );
}

function ItineraryLauncher() {
  const queryClient = useQueryClient();
  return (
    <QuickActionCard
      icon={MapPin}
      label="Multi-City Itinerary"
      placeholder="e.g. Paris, London, Tokyo"
      iconColor="text-emerald-400"
      buttonClass="bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400"
      onRun={async (q) => {
        const res = await agentsApi.run(q, 'itinerary');
        queryClient.invalidateQueries({ queryKey: ['hud-agents-full'] });
        return res.data;
      }}
    />
  );
}

function AgentPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['hud-agents-full'],
    queryFn: async () => {
      const res = await hudApi.getAllAgentsFull();
      return res.data;
    },
    refetchInterval: 1500,
  });

  const agents = data?.agents ?? [];

  if (isLoading) {
    return (
      <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-slate-700/50">
            <Bot className="w-5 h-5 text-slate-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">Active Agents</h3>
            <p className="text-xs text-slate-500">Real-time bot status</p>
          </div>
        </div>
        <div className="flex items-center justify-center py-12 gap-2 text-slate-500">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading agents...
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-6 transition-all duration-300 hover:border-white/[0.08]">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-slate-700/50">
            <Bot className="w-5 h-5 text-slate-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">Active Agents</h3>
            <p className="text-xs text-slate-500">{agents.length} {agents.length === 1 ? 'agent' : 'agents'} running</p>
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
        <div className="rounded-xl border border-dashed border-white/[0.08] bg-slate-900/30 p-8 text-center">
          <Bot className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <p className="text-sm font-medium text-slate-400">No agents running</p>
          <p className="text-xs text-slate-600 mt-1 max-w-[240px] mx-auto">
            Launch a research or itinerary task above to see parallel bots in action
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
              className="rounded-xl border border-white/[0.06] bg-slate-900/40 p-4 transition-colors hover:bg-slate-900/50"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-200 truncate max-w-[220px]">{a.description}</span>
                <span className={`text-xs font-medium px-2.5 py-1 rounded-lg flex items-center gap-1.5 shrink-0 ${
                  a.status === 'running' ? 'bg-blue-500/15 text-blue-400 border border-blue-500/20' :
                  a.status === 'completed' ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20' :
                  a.status === 'failed' ? 'bg-rose-500/15 text-rose-400 border border-rose-500/20' :
                  'bg-slate-500/15 text-slate-400 border border-slate-500/20'
                }`}>
                  {a.status === 'running' && <Activity className="w-3.5 h-3.5 animate-pulse" />}
                  {a.status === 'completed' && <CheckCircle2 className="w-3.5 h-3.5" />}
                  {a.status === 'failed' && <XCircle className="w-3.5 h-3.5" />}
                  {a.status}
                </span>
              </div>
              {a.bots && a.bots.length > 0 && (
                <div className="space-y-2 mt-3 pt-3 border-t border-white/[0.04]">
                  {a.bots.map((b: { id: string; name: string; source: string; status: string; output?: string; error?: string }) => (
                    <div key={b.id} className="flex items-center gap-3 text-xs">
                      <span className={`w-2 h-2 rounded-full shrink-0 ${
                        b.status === 'running' ? 'bg-blue-400 animate-pulse' :
                        b.status === 'completed' ? 'bg-emerald-400' :
                        b.status === 'failed' ? 'bg-rose-400' : 'bg-slate-500'
                      }`} />
                      <span className="text-slate-500 w-20 shrink-0">{b.name}</span>
                      <span className={`flex-1 truncate min-w-0 ${
                        b.status === 'completed' ? 'text-slate-400' : 'text-slate-500'
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
      <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-5 animate-pulse">
        <div className="h-4 w-24 bg-slate-700/50 rounded mb-3" />
        <div className="h-6 w-16 bg-slate-700/50 rounded" />
      </div>
    );
  }

  const total = data.total_calls ?? 0;
  const cost = data.cost_estimate_usd ?? 0;
  const latency = data.avg_latency_ms ?? 0;

  return (
    <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-5 transition-all duration-300 hover:border-white/[0.08]">
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 rounded-lg bg-slate-700/50">
          <DollarSign className="w-5 h-5 text-emerald-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Usage & Cost</h3>
          <p className="text-xs text-slate-500">LLM usage tracking</p>
        </div>
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-slate-500">API calls</span>
          <span className="font-medium text-white tabular-nums">{total}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">Est. cost</span>
          <span className="font-medium text-emerald-400 tabular-nums">${cost.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">Avg latency</span>
          <span className="font-medium text-slate-300 tabular-nums">{Math.round(latency)}ms</span>
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
    <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-5 transition-all duration-300 hover:border-white/[0.08]">
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 rounded-lg bg-slate-700/50">
          <Download className="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Data Export</h3>
          <p className="text-xs text-slate-500">Export conversations & audit</p>
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        {(['all', 'conversations', 'audit'] as const).map((t) => (
          <button
            key={t}
            onClick={() => handleExport(t)}
            disabled={downloading}
            className="px-3 py-1.5 rounded-lg bg-slate-700/50 hover:bg-slate-600/50 text-sm text-slate-300 transition-all disabled:opacity-50"
          >
            {downloading ? '...' : t === 'all' ? 'Full' : t === 'conversations' ? 'Chats' : 'Audit'}
          </button>
        ))}
      </div>
    </div>
  );
}

function QuickStats() {
  const { data } = useQuery({
    queryKey: ['system-status'],
    queryFn: async () => {
      const res = await systemApi.getStatus();
      return res.data;
    },
    refetchInterval: 15000,
  });

  const stats = [
    { icon: Zap, label: 'Skills', value: data?.skills?.enabled_skills ?? 0 },
    { icon: Calendar, label: 'Scheduled', value: data?.scheduled_jobs ?? 0 },
    { icon: Database, label: 'Vector docs', value: data?.vector_memory?.document_count ?? 0 },
    { icon: Plug, label: 'Plugins', value: data?.plugins ?? 0 },
  ];

  return (
    <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-6 transition-all duration-300 hover:border-white/[0.08]">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-slate-700/50">
          <Server className="w-5 h-5 text-slate-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Quick Stats</h3>
          <p className="text-xs text-slate-500">System overview</p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        {stats.map(({ icon: Icon, label, value }) => (
          <div key={label} className="flex items-center gap-3 p-3 rounded-xl bg-slate-900/40">
            <Icon className="w-4 h-4 text-slate-500" />
            <div>
              <p className="text-xs text-slate-500">{label}</p>
              <p className="text-lg font-semibold text-white tabular-nums">{value}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ConversationTimeline() {
  const { data, isLoading } = useQuery({
    queryKey: ['hud-timeline'],
    queryFn: async () => {
      const res = await hudApi.getTimeline();
      return res.data;
    },
    refetchInterval: 10000,
  });

  const events = data?.events ?? [];

  if (isLoading) {
    return (
      <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-6 animate-pulse">
        <div className="h-5 w-32 bg-slate-700/50 rounded mb-4" />
        <div className="space-y-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-10 bg-slate-700/30 rounded" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-slate-800/50 border border-white/[0.06] p-6 transition-all duration-300 hover:border-white/[0.08]">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-slate-700/50">
          <Calendar className="w-5 h-5 text-slate-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Today's Activity</h3>
          <p className="text-xs text-slate-500">Conversation timeline</p>
        </div>
      </div>
      {events.length === 0 ? (
        <div className="rounded-xl border border-dashed border-white/[0.08] bg-slate-900/30 p-6 text-center">
          <Calendar className="w-8 h-8 text-slate-600 mx-auto mb-2" />
          <p className="text-sm text-slate-500">No activity today</p>
        </div>
      ) : (
        <ul className="space-y-2">
          {events.slice(0, 10).map((e: { channel: string; user_id: string; updated_at: string; message_count: number }, i: number) => (
            <li
              key={i}
              className="flex items-center justify-between py-2.5 px-3 rounded-lg bg-slate-900/40 hover:bg-slate-900/50 transition-colors"
            >
              <span className="text-sm text-slate-300 font-medium">{e.channel} / {e.user_id}</span>
              <span className="text-xs text-slate-500">{e.message_count} msgs</span>
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
      <div className="relative overflow-hidden border-b border-white/[0.06] bg-gradient-to-b from-slate-800/40 to-transparent z-10">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(59,130,246,0.08),transparent)]" />
        <div className="relative px-8 py-10 z-10">
          <h1 className="text-3xl font-bold text-white tracking-tight">
            Dashboard
          </h1>
          <p className="text-slate-400 mt-1 max-w-2xl">
            Real-time system vitals, active agents, and JARVIS HUD. Monitor and launch research or itinerary tasks.
          </p>
        </div>
      </div>

      <div className="relative p-8 space-y-8 z-10">
        {/* System Vitals */}
        <section>
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">System Vitals</h2>
          <SystemVitals />
        </section>

        {/* Quick actions + Agent panel */}
        <section className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <div className="grid sm:grid-cols-2 gap-4">
              <ResearchLauncher />
              <ItineraryLauncher />
            </div>
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider pt-2">Active Agents</h2>
            <AgentPanel />
          </div>
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Overview</h2>
            <UsageCostWidget />
            <ExportButton />
            <QuickStats />
          </div>
        </section>

        {/* Activity timeline */}
        <section>
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">Activity</h2>
          <ConversationTimeline />
        </section>
      </div>
    </div>
  );
}
