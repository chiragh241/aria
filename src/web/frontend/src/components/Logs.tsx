import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import { FileText, Loader2, RefreshCw, Filter, AlertCircle, Clock, ChevronDown, Activity, ShieldAlert, XCircle, CheckCircle2 } from 'lucide-react';
import { useState } from 'react';
import { auditApi } from '../services/api';

interface AuditEntry {
  id: string;
  timestamp: string;
  event: string;
  action_type: string;
  user_id: string | null;
  channel: string | null;
  status: string;
  details: Record<string, any>;
}

const eventColors: Record<string, string> = {
  action_requested: 'bg-blue-500/10 text-blue-400 border border-blue-500/20',
  action_approved: 'bg-green-500/10 text-green-400 border border-green-500/20',
  action_denied: 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20',
  action_executed: 'bg-green-500/10 text-green-400 border border-green-500/20',
  action_failed: 'bg-red-500/10 text-red-400 border border-red-500/20',
  security_violation: 'bg-red-500/10 text-red-400 border border-red-500/20',
};

const statusColors: Record<string, string> = {
  info: 'text-blue-400',
  warning: 'text-yellow-400',
  error: 'text-red-400',
};

export default function Logs() {
  const [filter, setFilter] = useState<string>('');
  const [limit, setLimit] = useState(50);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['auditLog', filter, limit],
    queryFn: async () => {
      const response = await auditApi.getLog(limit, 0, filter || undefined);
      return response.data.entries as AuditEntry[];
    },
    refetchInterval: 30000,
    retry: 2,
  });

  const { data: stats } = useQuery({
    queryKey: ['auditStats'],
    queryFn: async () => {
      try {
        const response = await auditApi.getStats();
        return response.data;
      } catch {
        return null;
      }
    },
    retry: 1,
  });

  const eventTypes = [
    { value: '', label: 'All Events' },
    { value: 'action_requested', label: 'Requested' },
    { value: 'action_approved', label: 'Approved' },
    { value: 'action_denied', label: 'Denied' },
    { value: 'action_executed', label: 'Executed' },
    { value: 'action_failed', label: 'Failed' },
  ];

  const statCards = [
    {
      label: 'Total Events',
      value: stats?.total_entries ?? 0,
      icon: Activity,
      color: 'text-theme-primary',
      accent: 'from-blue-500',
    },
    {
      label: 'Executed',
      value: stats?.by_event?.action_executed ?? 0,
      icon: CheckCircle2,
      color: 'text-green-400',
      accent: 'from-green-500',
    },
    {
      label: 'Denied',
      value: stats?.by_event?.action_denied ?? 0,
      icon: ShieldAlert,
      color: 'text-yellow-400',
      accent: 'from-yellow-500',
    },
    {
      label: 'Failed',
      value: stats?.by_event?.action_failed ?? 0,
      icon: XCircle,
      color: 'text-red-400',
      accent: 'from-red-500',
    },
  ];

  return (
    <div className="p-6 lg:p-8">
      <div className="page-header flex items-center justify-between">
        <div>
          <h1 className="page-title flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/10 border border-emerald-500/20 flex items-center justify-center">
              <FileText className="w-5 h-5 text-emerald-400" />
            </div>
            Audit Logs
          </h1>
          <p className="page-subtitle">
            Track all actions and security events
          </p>
        </div>

        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="btn-ghost flex items-center gap-2 text-sm"
        >
          <RefreshCw className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`} />
          <span className="hidden sm:inline">Refresh</span>
        </button>
      </div>

      {/* Error state */}
      {isError && (
        <div className="glass-card p-8 text-center mb-6">
          <AlertCircle className="w-10 h-10 mx-auto text-red-400/60 mb-3" />
          <p className="text-red-400 font-medium mb-1">Failed to load audit logs</p>
          <p className="text-sm text-slate-500 mb-4">
            {(error as any)?.response?.status === 401
              ? 'Session expired. Please log in again.'
              : (error as any)?.message || 'Could not connect to the server.'}
          </p>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors text-sm"
          >
            Try Again
          </button>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {statCards.map((card) => (
          <div key={card.label} className="stat-card" style={{ '--stat-accent': card.accent } as React.CSSProperties}>
            <div className="flex items-start justify-between">
              <div>
                <p className={`text-2xl font-bold ${card.color} tracking-tight`}>
                  {card.value}
                </p>
                <p className="text-xs text-slate-500 mt-1">{card.label}</p>
              </div>
              <card.icon className={`w-4 h-4 ${card.color} opacity-40`} />
            </div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-6">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-500" />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="px-3 py-2 glass-input text-sm"
          >
            {eventTypes.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>

        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="px-3 py-2 glass-input text-sm"
        >
          <option value={25}>25 entries</option>
          <option value={50}>50 entries</option>
          <option value={100}>100 entries</option>
        </select>

        {isFetching && !isLoading && (
          <Loader2 className="w-4 h-4 animate-spin text-theme-secondary" />
        )}
      </div>

      {/* Log entries */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 animate-spin text-theme-secondary" />
        </div>
      ) : !isError && data?.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <div className="w-16 h-16 rounded-2xl card-theme border border-theme flex items-center justify-center mx-auto mb-5">
            <Clock className="w-8 h-8 text-theme-secondary" />
          </div>
          <p className="text-lg font-medium text-slate-300 mb-1">No log entries yet</p>
          <p className="text-sm text-theme-secondary max-w-sm mx-auto">
            {filter
              ? `No "${filter.replace(/_/g, ' ')}" events found. Try changing the filter.`
              : 'Audit logs will appear here as Aria processes requests.'}
          </p>
          {filter && (
            <button
              onClick={() => setFilter('')}
              className="mt-4 px-4 py-2 btn-theme-secondary hover:bg-theme-muted rounded-lg transition-colors text-sm border border-theme"
            >
              Clear Filter
            </button>
          )}
        </div>
      ) : !isError && data && data.length > 0 ? (
        <div className="glass-card overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-theme">
                <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Time
                </th>
                <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Event
                </th>
                <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Action
                </th>
                <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  User
                </th>
                <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Channel
                </th>
                <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="w-10"></th>
              </tr>
            </thead>
            <tbody>
              {data.map((entry) => (
                <>
                  <tr
                    key={entry.id}
                    onClick={() =>
                      setExpandedRow(expandedRow === entry.id ? null : entry.id)
                    }
                    className={`table-row-hover border-b border-white/[0.03] ${
                      expandedRow === entry.id ? 'bg-white/[0.02]' : ''
                    }`}
                  >
                    <td className="px-4 py-3 text-xs text-theme-secondary font-mono">
                      {format(new Date(entry.timestamp), 'MMM d, HH:mm:ss')}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                          eventColors[entry.event] || 'card-theme text-theme-secondary border border-theme'
                        }`}
                      >
                        {entry.event.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-theme-secondary">
                      {entry.action_type?.replace(/_/g, ' ') || '-'}
                    </td>
                    <td className="px-4 py-3 text-xs text-theme-secondary">
                      {entry.user_id || '-'}
                    </td>
                    <td className="px-4 py-3 text-xs text-theme-secondary">
                      {entry.channel || '-'}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-xs ${statusColors[entry.status] || 'text-theme-secondary'}`}>
                        {entry.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <ChevronDown
                        className={`w-3.5 h-3.5 text-theme-secondary transition-transform ${
                          expandedRow === entry.id ? 'rotate-180' : ''
                        }`}
                      />
                    </td>
                  </tr>
                  {expandedRow === entry.id && entry.details && Object.keys(entry.details).length > 0 && (
                    <tr key={`${entry.id}-details`}>
                      <td colSpan={7} className="px-4 py-3 card-theme border-b border-white/[0.03]">
                        <pre className="text-[11px] text-slate-500 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
                          {JSON.stringify(entry.details, null, 2)}
                        </pre>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
