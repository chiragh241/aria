import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Cpu, HardDrive, MemoryStick, Wifi, WifiOff, Loader2, Bot, Calendar, CheckCircle2, XCircle, Activity, Search, Send, MapPin } from 'lucide-react';
import { hudApi, systemApi, agentsApi } from '../services/api';

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
      <div className="glass-card p-4 flex items-center justify-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
        <span className="text-sm text-slate-500">Loading vitals...</span>
      </div>
    );
  }

  const cpuColor = (vitals.cpu_percent ?? 0) > 80 ? 'text-red-400' : (vitals.cpu_percent ?? 0) > 60 ? 'text-amber-400' : 'text-green-400';
  const memColor = (vitals.memory_percent ?? 0) > 80 ? 'text-red-400' : (vitals.memory_percent ?? 0) > 60 ? 'text-amber-400' : 'text-green-400';

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div className="glass-card p-4">
        <div className="flex items-center gap-2 text-slate-500 mb-1">
          <Cpu className="w-4 h-4" />
          <span className="text-xs font-medium">CPU</span>
        </div>
        <p className={`text-xl font-bold ${cpuColor}`}>{vitals.cpu_percent ?? 0}%</p>
      </div>
      <div className="glass-card p-4">
        <div className="flex items-center gap-2 text-slate-500 mb-1">
          <MemoryStick className="w-4 h-4" />
          <span className="text-xs font-medium">Memory</span>
        </div>
        <p className={`text-xl font-bold ${memColor}`}>{vitals.memory_percent ?? 0}%</p>
        <p className="text-xs text-slate-600 mt-0.5">
          {vitals.memory_used_gb ?? 0} / {vitals.memory_total_gb ?? 0} GB
        </p>
      </div>
      <div className="glass-card p-4">
        <div className="flex items-center gap-2 text-slate-500 mb-1">
          <HardDrive className="w-4 h-4" />
          <span className="text-xs font-medium">Disk</span>
        </div>
        <p className="text-xl font-bold text-slate-200">{vitals.disk_percent ?? 0}%</p>
      </div>
      <div className="glass-card p-4">
        <div className="flex items-center gap-2 text-slate-500 mb-1">
          {vitals.llm?.local || vitals.llm?.cloud ? (
            <Wifi className="w-4 h-4 text-green-400" />
          ) : (
            <WifiOff className="w-4 h-4 text-amber-400" />
          )}
          <span className="text-xs font-medium">LLM</span>
        </div>
        <p className="text-sm text-slate-300">
          Local: {vitals.llm?.local ? '✓' : '✗'} | Cloud: {vitals.llm?.cloud ? '✓' : '✗'}
        </p>
      </div>
    </div>
  );
}

function ResearchLauncher() {
  const [query, setQuery] = useState('');
  const queryClient = useQueryClient();
  const runMutation = useMutation({
    mutationFn: async (q: string) => {
      const res = await agentsApi.run(q, 'research');
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['hud-agents-full'] });
    },
  });
  return (
    <div className="glass-card p-3 mb-3">
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Research a topic (Reddit, X, Web)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && (runMutation.mutate(query), setQuery(''))}
          className="flex-1 px-3 py-2 rounded-lg bg-slate-800/50 border border-white/10 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500/50"
        />
        <button
          onClick={() => (runMutation.mutate(query), setQuery(''))}
          disabled={!query.trim() || runMutation.isPending}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm flex items-center gap-2"
        >
          {runMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          Research
        </button>
      </div>
    </div>
  );
}

function ItineraryLauncher() {
  const [query, setQuery] = useState('');
  const queryClient = useQueryClient();
  const runMutation = useMutation({
    mutationFn: async (q: string) => {
      const res = await agentsApi.run(q, 'itinerary');
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['hud-agents-full'] });
    },
  });
  return (
    <div className="glass-card p-3 mb-3">
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Multi-city itinerary (e.g. Paris, London, Tokyo)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && (runMutation.mutate(query), setQuery(''))}
          className="flex-1 px-3 py-2 rounded-lg bg-slate-800/50 border border-white/10 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500/50"
        />
        <button
          onClick={() => (runMutation.mutate(query), setQuery(''))}
          disabled={!query.trim() || runMutation.isPending}
          className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm flex items-center gap-2"
        >
          {runMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <MapPin className="w-4 h-4" />}
          Itinerary
        </button>
      </div>
    </div>
  );
}

function AgentPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['hud-agents-full'],
    queryFn: async () => {
      const res = await hudApi.getAllAgentsFull();
      return res.data;
    },
    refetchInterval: 1500, // Real-time: poll every 1.5s for live bot updates
  });

  const agents = data?.agents ?? [];

  if (isLoading) {
    return (
      <div className="glass-card p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <Bot className="w-4 h-4" />
          Active Agents & Bots
        </h3>
        <div className="flex items-center gap-2 text-slate-500">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading...
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <Bot className="w-4 h-4" />
        Active Agents & Bots
      </h3>
      {agents.length === 0 ? (
        <p className="text-sm text-slate-500">No agents running. Ask Aria to research a topic to see multiple bots in action.</p>
      ) : (
        <div className="space-y-4">
          {agents.slice(0, 5).map((a: {
            id: string;
            description: string;
            status: string;
            bots?: { id: string; name: string; source: string; status: string; output?: string; error?: string }[];
            final_result?: string;
          }) => (
            <div key={a.id} className="border border-white/5 rounded-lg p-3 bg-slate-900/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-300 text-sm font-medium truncate max-w-[180px]">{a.description}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full flex items-center gap-1 ${
                  a.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                  a.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                  a.status === 'failed' ? 'bg-red-500/20 text-red-400' : 'bg-slate-500/20 text-slate-400'
                }`}>
                  {a.status === 'running' && <Activity className="w-3 h-3 animate-pulse" />}
                  {a.status === 'completed' && <CheckCircle2 className="w-3 h-3" />}
                  {a.status === 'failed' && <XCircle className="w-3 h-3" />}
                  {a.status}
                </span>
              </div>
              {a.bots && a.bots.length > 0 && (
                <div className="space-y-1.5 mt-2">
                  {a.bots.map((b: { id: string; name: string; source: string; status: string; output?: string; error?: string }) => (
                    <div key={b.id} className="flex items-center gap-2 text-xs">
                      <span className={`w-2 h-2 rounded-full ${
                        b.status === 'running' ? 'bg-blue-400 animate-pulse' :
                        b.status === 'completed' ? 'bg-green-400' :
                        b.status === 'failed' ? 'bg-red-400' : 'bg-slate-500'
                      }`} />
                      <span className="text-slate-500 w-20">{b.name}</span>
                      <span className={`flex-1 truncate ${
                        b.status === 'completed' ? 'text-slate-400' : 'text-slate-500'
                      }`}>
                        {b.status === 'running' ? 'Searching...' : (b.output || b.error || '-')}
                      </span>
                      <span className="text-slate-600">{b.status}</span>
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
      <div className="glass-card p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <Calendar className="w-4 h-4" />
          Today's Activity
        </h3>
        <div className="flex items-center gap-2 text-slate-500">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading...
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <Calendar className="w-4 h-4" />
        Today's Activity
      </h3>
      {events.length === 0 ? (
        <p className="text-sm text-slate-500">No activity today</p>
      ) : (
        <ul className="space-y-2">
          {events.slice(0, 10).map((e: { channel: string; user_id: string; updated_at: string; message_count: number }, i: number) => (
            <li key={i} className="flex items-center justify-between text-sm">
              <span className="text-slate-300">{e.channel} / {e.user_id}</span>
              <span className="text-slate-500">{e.message_count} msgs</span>
            </li>
          ))}
        </ul>
      )}
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

  return (
    <div className="glass-card p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">Quick Stats</h3>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-slate-500">Skills:</span>
          <span className="ml-2 text-slate-300">{data?.skills?.enabled_skills ?? 0}</span>
        </div>
        <div>
          <span className="text-slate-500">Scheduled:</span>
          <span className="ml-2 text-slate-300">{data?.scheduled_jobs ?? 0}</span>
        </div>
        <div>
          <span className="text-slate-500">Vector docs:</span>
          <span className="ml-2 text-slate-300">{data?.vector_memory?.document_count ?? 0}</span>
        </div>
        <div>
          <span className="text-slate-500">Plugins:</span>
          <span className="ml-2 text-slate-300">{data?.plugins ?? 0}</span>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Dashboard</h1>
        <p className="text-slate-500 text-sm mt-1">Real-time system status and JARVIS HUD</p>
      </div>

      <div className="space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">System Vitals</h2>
          <SystemVitals />
        </div>

        <div className="grid md:grid-cols-2 gap-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Active Agents & Bots</h2>
          <ResearchLauncher />
          <AgentPanel />
        </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Quick Stats</h2>
            <QuickStats />
          </div>
        </div>

        <div>
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Today's Activity</h2>
          <ConversationTimeline />
        </div>
      </div>
    </div>
  );
}
