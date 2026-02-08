import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Puzzle, Power, Loader2, ChevronRight, AlertCircle, RefreshCw, Search, Zap, Info, Save, CheckCircle, XCircle, Eye, EyeOff, Play } from 'lucide-react';
import { useState, useEffect } from 'react';
import { skillsApi } from '../services/api';

interface CredentialField {
  key: string;
  label: string;
  secret: boolean;
  is_set: boolean;
  value: string;
}

interface Skill {
  name: string;
  description: string;
  version: string;
  enabled: boolean;
  initialized?: boolean;
  capabilities: {
    name: string;
    description: string;
  }[];
}

const SKILLS_NEEDING_SETUP = ['calendar', 'email', 'sms', 'browser'];

const SKILL_SETUP_HELP: Record<string, string> = {
  calendar: 'Set up Google Calendar API credentials. Download credentials.json from Google Cloud Console.',
  email: 'Configure your email server for sending and receiving. Gmail users: use an App Password.',
  sms: 'Set up Twilio for SMS messaging. Get credentials from twilio.com/console.',
  browser: 'Browser works out of the box with Playwright. Optionally add Brave Search API key for web search.',
  video: 'Requires FFmpeg installed on your system. Run: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)',
  stt: 'Uses OpenAI Whisper locally. Downloads model (~1.5GB) on first use. No API key needed.',
  tts: 'Uses Microsoft Edge TTS. Free, works immediately. No setup needed.',
};

const skillIcons: Record<string, { icon: string; gradient: string }> = {
  filesystem:  { icon: '📁', gradient: 'from-blue-500/15 to-blue-500/5' },
  shell:       { icon: '💻', gradient: 'from-purple-500/15 to-purple-500/5' },
  browser:     { icon: '🌐', gradient: 'from-cyan-500/15 to-cyan-500/5' },
  calendar:    { icon: '📅', gradient: 'from-green-500/15 to-green-500/5' },
  email:       { icon: '📧', gradient: 'from-amber-500/15 to-amber-500/5' },
  sms:         { icon: '💬', gradient: 'from-teal-500/15 to-teal-500/5' },
  tts:         { icon: '🔊', gradient: 'from-orange-500/15 to-orange-500/5' },
  stt:         { icon: '🎤', gradient: 'from-pink-500/15 to-pink-500/5' },
  image:       { icon: '🖼️', gradient: 'from-violet-500/15 to-violet-500/5' },
  video:       { icon: '🎬', gradient: 'from-red-500/15 to-red-500/5' },
  documents:   { icon: '📄', gradient: 'from-emerald-500/15 to-emerald-500/5' },
  memory:      { icon: '🧠', gradient: 'from-indigo-500/15 to-indigo-500/5' },
  weather:     { icon: '🌤️', gradient: 'from-sky-500/15 to-sky-500/5' },
  news:        { icon: '📰', gradient: 'from-amber-600/15 to-amber-600/5' },
  finance:     { icon: '📈', gradient: 'from-lime-500/15 to-lime-500/5' },
  contacts:    { icon: '👥', gradient: 'from-fuchsia-500/15 to-fuchsia-500/5' },
  tracking:    { icon: '📦', gradient: 'from-rose-500/15 to-rose-500/5' },
  home:        { icon: '🏠', gradient: 'from-amber-500/15 to-amber-500/5' },
  webhook:     { icon: '🔗', gradient: 'from-slate-500/15 to-slate-500/5' },
  agent:       { icon: '🤖', gradient: 'from-cyan-600/15 to-cyan-600/5' },
  research:    { icon: '🔍', gradient: 'from-blue-600/15 to-blue-600/5' },
  notion:      { icon: '📝', gradient: 'from-slate-600/15 to-slate-600/5' },
  todoist:     { icon: '✅', gradient: 'from-green-600/15 to-green-600/5' },
  linear:      { icon: '📋', gradient: 'from-violet-600/15 to-violet-600/5' },
  spotify:     { icon: '🎵', gradient: 'from-green-500/15 to-green-500/5' },
};

const defaultSkillIcon = { icon: '⚡', gradient: 'from-blue-500/15 to-blue-500/5' };

function SkillCredentialsForm({ skillName }: { skillName: string }) {
  const [credValues, setCredValues] = useState<Record<string, string>>({});
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [saveMsg, setSaveMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [testMsg, setTestMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const { data: credData, isLoading } = useQuery({
    queryKey: ['skillCredentials', skillName],
    queryFn: async () => {
      const resp = await skillsApi.getCredentials(skillName);
      return resp.data as { skill: string; fields: CredentialField[]; has_credentials: boolean };
    },
  });

  useEffect(() => {
    if (credData?.fields) {
      const vals: Record<string, string> = {};
      credData.fields.forEach((f) => {
        vals[f.key] = f.value || '';
      });
      setCredValues(vals);
    }
  }, [credData]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const nonEmpty: Record<string, string> = {};
      for (const [k, v] of Object.entries(credValues)) {
        if (v.trim()) nonEmpty[k] = v.trim();
      }
      return (await skillsApi.saveCredentials(skillName, nonEmpty)).data;
    },
    onSuccess: (data) => {
      setSaveMsg({ type: 'success', text: data.message || 'Credentials saved' });
      setTimeout(() => setSaveMsg(null), 3000);
    },
    onError: () => {
      setSaveMsg({ type: 'error', text: 'Failed to save credentials' });
      setTimeout(() => setSaveMsg(null), 3000);
    },
  });

  const testMutation = useMutation({
    mutationFn: async () => {
      return (await skillsApi.testConnection(skillName)).data;
    },
    onSuccess: (data) => {
      setTestMsg({ type: data.success ? 'success' : 'error', text: data.message });
      setTimeout(() => setTestMsg(null), 5000);
    },
    onError: () => {
      setTestMsg({ type: 'error', text: 'Test request failed' });
      setTimeout(() => setTestMsg(null), 5000);
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-2">
        <Loader2 className="w-3.5 h-3.5 animate-spin text-theme-secondary" />
        <span className="text-xs text-theme-secondary">Loading credentials...</span>
      </div>
    );
  }

  if (!credData?.fields?.length) {
    return null;
  }

  return (
    <div className="mt-4 space-y-3">
      <h4 className="text-[11px] font-semibold text-theme-secondary uppercase tracking-wider">Credentials</h4>

      {credData.fields.map((field) => (
        <div key={field.key}>
          <label className="block text-[11px] text-theme-secondary mb-1">
            {field.label}
            {field.is_set && (
              <span className="ml-2 text-green-400/80 text-[10px]">configured</span>
            )}
          </label>
          <div className="relative">
            <input
              type={field.secret && !showSecrets[field.key] ? 'password' : 'text'}
              value={credValues[field.key] || ''}
              onChange={(e) => setCredValues({ ...credValues, [field.key]: e.target.value })}
              placeholder={field.is_set ? '(already set - enter new value to change)' : `Enter ${field.label}`}
              className="w-full px-3 py-2 glass-input text-xs pr-8"
            />
            {field.secret && (
              <button
                type="button"
                onClick={() => setShowSecrets({ ...showSecrets, [field.key]: !showSecrets[field.key] })}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-theme-secondary hover:text-slate-300"
              >
                {showSecrets[field.key] ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              </button>
            )}
          </div>
        </div>
      ))}

      {saveMsg && (
        <div className={`flex items-center gap-1.5 text-xs ${saveMsg.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
          {saveMsg.type === 'success' ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
          {saveMsg.text}
        </div>
      )}

      {testMsg && (
        <div className={`flex items-center gap-1.5 text-xs ${testMsg.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
          {testMsg.type === 'success' ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
          {testMsg.text}
        </div>
      )}

      <div className="flex gap-2">
        <button
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 rounded-lg text-xs font-medium border border-blue-500/20 transition-colors"
        >
          {saveMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
          Save Credentials
        </button>
        <button
          onClick={() => testMutation.mutate()}
          disabled={testMutation.isPending}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 rounded-lg text-xs font-medium border border-emerald-500/20 transition-colors"
        >
          {testMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
          Test Connection
        </button>
      </div>
    </div>
  );
}

export default function Skills() {
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const queryClient = useQueryClient();

  const { data: skills, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['skills'],
    queryFn: async () => {
      const response = await skillsApi.list();
      return response.data.skills as Skill[];
    },
    retry: 2,
  });

  const toggleMutation = useMutation({
    mutationFn: async ({ name, enabled }: { name: string; enabled: boolean }) => {
      await skillsApi.toggle(name, enabled);
    },
    onMutate: async ({ name, enabled }) => {
      await queryClient.cancelQueries({ queryKey: ['skills'] });
      const previous = queryClient.getQueryData(['skills']);
      queryClient.setQueryData(['skills'], (old: Skill[] | undefined) =>
        old?.map((s) => (s.name === name ? { ...s, enabled } : s))
      );
      return { previous };
    },
    onError: (_err, _vars, context) => {
      queryClient.setQueryData(['skills'], context?.previous);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['skills'] });
    },
  });

  const filteredSkills = skills?.filter(
    (s) =>
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const selected = skills?.find((s) => s.name === selectedSkill);
  const enabledCount = skills?.filter((s) => s.enabled).length ?? 0;
  const totalCount = skills?.length ?? 0;
  const needsSetup = selected && SKILLS_NEEDING_SETUP.includes(selected.name);

  return (
    <div className="p-6 lg:p-8">
      <div className="page-header flex items-start justify-between">
        <div>
          <h1 className="page-title flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/10 border border-purple-500/20 flex items-center justify-center">
              <Puzzle className="w-5 h-5 text-purple-400" />
            </div>
            Skills
          </h1>
          <p className="page-subtitle">
            Manage Aria's capabilities and abilities
          </p>
        </div>
        {skills && (
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium px-2.5 py-1.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
              {enabledCount}/{totalCount} active
            </span>
            <button
              onClick={() => refetch()}
              className="btn-ghost p-2"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Error state */}
      {isError && (
        <div className="glass-card p-8 text-center mb-6">
          <AlertCircle className="w-10 h-10 mx-auto text-red-400/60 mb-3" />
          <p className="text-red-400 font-medium mb-1">Failed to load skills</p>
          <p className="text-sm text-theme-secondary mb-4">
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

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 animate-spin text-theme-secondary" />
        </div>
      )}

      {/* Main content */}
      {!isLoading && !isError && (
        <>
          {/* Search */}
          <div className="relative mb-6">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-theme-secondary" />
            <input
              type="text"
              placeholder="Search skills..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 glass-input"
            />
          </div>

          {/* Toggle error */}
          {toggleMutation.isError && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
              Failed to toggle skill. Please try again.
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Skills list */}
            <div className="lg:col-span-2 space-y-2">
              {filteredSkills?.length === 0 ? (
                <div className="glass-card p-10 text-center">
                  <Search className="w-10 h-10 mx-auto text-theme-secondary mb-3" />
                  <p className="text-theme-secondary">
                    {searchQuery ? 'No skills match your search' : 'No skills available'}
                  </p>
                </div>
              ) : (
                filteredSkills?.map((skill) => {
                  const iconCfg = skillIcons[skill.name] || defaultSkillIcon;
                  const hasSetup = SKILLS_NEEDING_SETUP.includes(skill.name);
                  return (
                    <div
                      key={skill.name}
                      onClick={() => setSelectedSkill(skill.name)}
                      className={`glass-card-hover p-4 cursor-pointer ${
                        selectedSkill === skill.name
                          ? 'border-blue-500/30 bg-blue-500/[0.04]'
                          : ''
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3.5">
                          <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${iconCfg.gradient} border border-theme flex items-center justify-center text-xl flex-shrink-0`}>
                            {iconCfg.icon}
                          </div>
                          <div>
                            <h3 className="font-medium text-theme-primary text-[14px] capitalize flex items-center gap-2">
                              {skill.name}
                              {skill.enabled && hasSetup && (
                                <span className="text-[10px] text-amber-400/80 font-normal px-1.5 py-0.5 bg-amber-500/10 rounded flex items-center gap-1 border border-amber-500/20">
                                  <Info className="w-2.5 h-2.5" /> configure
                                </span>
                              )}
                            </h3>
                            <p className="text-xs text-theme-secondary mt-0.5 line-clamp-1">{skill.description}</p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2.5">
                          <span
                            className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-1 rounded ${
                              skill.enabled
                                ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                                : 'card-theme text-theme-secondary border border-theme'
                            }`}
                          >
                            {skill.enabled ? 'On' : 'Off'}
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleMutation.mutate({
                                name: skill.name,
                                enabled: !skill.enabled,
                              });
                            }}
                            disabled={toggleMutation.isPending}
                            className={`p-1.5 rounded-lg transition-all ${
                              skill.enabled
                                ? 'text-green-400 hover:bg-green-500/10'
                                : 'text-theme-secondary hover:bg-theme-hover'
                            }`}
                            title={skill.enabled ? 'Disable skill' : 'Enable skill'}
                          >
                            <Power className="w-4 h-4" />
                          </button>
                          <ChevronRight className={`w-4 h-4 text-theme-secondary transition-transform ${selectedSkill === skill.name ? 'rotate-90' : ''}`} />
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Skill details */}
            <div className="lg:col-span-1">
              {selected ? (
                <div className="glass-card p-5 sticky top-6 animate-fade-in">
                  <div className="flex items-center gap-3 mb-5">
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${(skillIcons[selected.name] || defaultSkillIcon).gradient} border border-theme flex items-center justify-center text-2xl`}>
                      {(skillIcons[selected.name] || defaultSkillIcon).icon}
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-theme-primary capitalize">
                        {selected.name}
                      </h2>
                      <div className="flex items-center gap-2">
                        <p className="text-xs text-theme-secondary">v{selected.version}</p>
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            selected.enabled ? 'bg-green-400' : 'bg-theme-muted'
                          }`}
                        />
                      </div>
                    </div>
                  </div>

                  <p className="text-sm text-theme-secondary mb-5 leading-relaxed">{selected.description}</p>

                  {/* Setup help text */}
                  {SKILL_SETUP_HELP[selected.name] && (
                    <div className="mb-4 p-3 bg-blue-500/[0.06] border border-blue-500/20 rounded-lg">
                      <p className="text-[11px] text-blue-400 font-medium mb-1">Setup Info</p>
                      <p className="text-[11px] text-theme-secondary">{SKILL_SETUP_HELP[selected.name]}</p>
                    </div>
                  )}

                  {/* Credential form for skills that need API keys */}
                  {needsSetup && <SkillCredentialsForm skillName={selected.name} />}

                  {/* Capabilities */}
                  <div className="mt-5">
                    <div className="flex items-center gap-2 mb-3">
                      <Zap className="w-3.5 h-3.5 text-theme-secondary" />
                      <h3 className="text-[11px] font-semibold text-theme-secondary uppercase tracking-wider">
                        Capabilities ({selected.capabilities?.length || 0})
                      </h3>
                    </div>
                    <div className="space-y-1.5 max-h-48 overflow-y-auto">
                      {selected.capabilities?.length > 0 ? (
                        selected.capabilities.map((cap) => (
                          <div
                            key={cap.name}
                            className="card-theme border border-theme rounded-lg p-3"
                          >
                            <p className="font-medium text-theme-primary text-xs">{cap.name}</p>
                            <p className="text-[11px] text-theme-secondary mt-0.5">
                              {cap.description}
                            </p>
                          </div>
                        ))
                      ) : (
                        <p className="text-xs text-theme-secondary">No capabilities listed</p>
                      )}
                    </div>
                  </div>

                  <div className="mt-5 pt-4 border-t border-theme">
                    <button
                      onClick={() =>
                        toggleMutation.mutate({
                          name: selected.name,
                          enabled: !selected.enabled,
                        })
                      }
                      disabled={toggleMutation.isPending}
                      className={`w-full py-2.5 rounded-lg font-medium text-sm transition-all ${
                        selected.enabled
                          ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20'
                          : 'bg-green-500/10 text-green-400 hover:bg-green-500/20 border border-green-500/20'
                      }`}
                    >
                      {selected.enabled ? 'Disable Skill' : 'Enable Skill'}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="glass-card p-8 text-center">
                  <div className="w-14 h-14 rounded-2xl card-theme border border-theme flex items-center justify-center mx-auto mb-4">
                    <Puzzle className="w-7 h-7 text-theme-secondary" />
                  </div>
                  <p className="text-sm text-theme-secondary">Select a skill to view details</p>
                  <p className="text-xs text-theme-secondary mt-1.5">
                    Click any skill to configure and test it
                  </p>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
