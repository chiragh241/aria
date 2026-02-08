import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { Shield, Check, X, Loader2, Clock, AlertCircle, RefreshCw } from 'lucide-react';
import { approvalsApi } from '../services/api';

interface Approval {
  id: string;
  action_type: string;
  description: string;
  created_at: string;
  timeout: number;
}

const actionConfig: Record<string, { icon: string; color: string; gradient: string }> = {
  read_files:      { icon: '📁', color: 'blue',   gradient: 'from-blue-500/10 to-blue-500/5' },
  write_files:     { icon: '✏️', color: 'amber',  gradient: 'from-amber-500/10 to-amber-500/5' },
  delete_files:    { icon: '🗑️', color: 'red',    gradient: 'from-red-500/10 to-red-500/5' },
  shell_commands:  { icon: '💻', color: 'purple', gradient: 'from-purple-500/10 to-purple-500/5' },
  web_requests:    { icon: '🌐', color: 'cyan',   gradient: 'from-cyan-500/10 to-cyan-500/5' },
  send_emails:     { icon: '📧', color: 'green',  gradient: 'from-green-500/10 to-green-500/5' },
  send_messages:   { icon: '💬', color: 'blue',   gradient: 'from-blue-500/10 to-blue-500/5' },
  calendar_read:   { icon: '📅', color: 'teal',   gradient: 'from-teal-500/10 to-teal-500/5' },
  calendar_write:  { icon: '📅', color: 'teal',   gradient: 'from-teal-500/10 to-teal-500/5' },
};

const defaultAction = { icon: '⚡', color: 'blue', gradient: 'from-blue-500/10 to-blue-500/5' };

export default function Approvals() {
  const queryClient = useQueryClient();

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['approvals'],
    queryFn: async () => {
      const response = await approvalsApi.getPending();
      return response.data.approvals as Approval[];
    },
    refetchInterval: 5000,
    retry: 2,
  });

  const respondMutation = useMutation({
    mutationFn: async ({ id, approved }: { id: string; approved: boolean }) => {
      await approvalsApi.respond(id, approved);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['approvals'] });
    },
  });

  return (
    <div className="p-6 lg:p-8 max-w-4xl">
      <div className="page-header flex items-start justify-between">
        <div>
          <h1 className="page-title flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/10 border border-amber-500/20 flex items-center justify-center">
              <Shield className="w-5 h-5 text-amber-400" />
            </div>
            Pending Approvals
          </h1>
          <p className="page-subtitle">
            Review and approve actions that require your permission
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="btn-ghost"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Error state */}
      {isError && (
        <div className="glass-card p-8 text-center mb-6">
          <AlertCircle className="w-10 h-10 mx-auto text-red-400/60 mb-3" />
          <p className="text-red-400 font-medium mb-1">Failed to load approvals</p>
          <p className="text-sm text-slate-500 mb-4">
            {(error as any)?.message || 'Could not connect to the server.'}
          </p>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors text-sm"
          >
            Try Again
          </button>
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 animate-spin text-slate-600" />
        </div>
      ) : !isError && data?.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <div className="w-16 h-16 rounded-2xl bg-slate-800/50 border border-white/[0.06] flex items-center justify-center mx-auto mb-5">
            <Shield className="w-8 h-8 text-slate-600" />
          </div>
          <p className="text-lg font-medium text-slate-300 mb-1">All clear</p>
          <p className="text-sm text-slate-600">
            Actions requiring your approval will appear here
          </p>
        </div>
      ) : !isError && (
        <div className="space-y-3">
          {data?.map((approval) => {
            const config = actionConfig[approval.action_type] || defaultAction;
            return (
              <div
                key={approval.id}
                className="glass-card-hover p-5 animate-slide-up"
              >
                <div className="flex items-start gap-4">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${config.gradient} border border-white/[0.06] flex items-center justify-center text-2xl flex-shrink-0`}>
                    {config.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-white text-[15px]">
                      {approval.action_type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                    </h3>
                    <p className="text-sm text-slate-400 mt-0.5 line-clamp-2">{approval.description}</p>
                    <div className="flex items-center gap-4 mt-3 text-xs text-slate-500">
                      <span className="flex items-center gap-1.5">
                        <Clock className="w-3 h-3" />
                        {format(new Date(approval.created_at), 'HH:mm:ss')}
                      </span>
                      <span className="flex items-center gap-1.5">
                        Timeout: {approval.timeout}s
                      </span>
                    </div>
                  </div>

                  <div className="flex gap-2 flex-shrink-0">
                    <button
                      onClick={() =>
                        respondMutation.mutate({ id: approval.id, approved: true })
                      }
                      disabled={respondMutation.isPending}
                      className="px-4 py-2 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 text-green-400 rounded-lg font-medium transition-all text-sm flex items-center gap-2"
                    >
                      <Check className="w-4 h-4" />
                      Approve
                    </button>
                    <button
                      onClick={() =>
                        respondMutation.mutate({ id: approval.id, approved: false })
                      }
                      disabled={respondMutation.isPending}
                      className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 text-red-400 rounded-lg font-medium transition-all text-sm flex items-center gap-2"
                    >
                      <X className="w-4 h-4" />
                      Deny
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
