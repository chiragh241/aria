import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Settings as SettingsIcon,
  Shield,
  Bot,
  Loader2,
  Save,
  MessageSquare,
  Container,
  Globe,
  Puzzle,
  Key,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  AlertCircle,
  Eye,
  EyeOff,
  RefreshCw,
  Brain,
  Smartphone,
  RotateCcw,
  Send,
  Play,
  Trash2,
} from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import QRCode from 'qrcode';
import { configApi, knowledgeApi, whatsappApi, channelsApi, restartApi, dockerApi, systemApi } from '../services/api';

// ── Types ────────────────────────────────────────────────────────────────────

interface FullConfig {
  llm: {
    provider: string;
    anthropic_model: string;
    anthropic_key_set: boolean;
    ollama_enabled: boolean;
    ollama_model: string;
    ollama_base_url: string;
    cloud_enabled: boolean;
  };
  channels: {
    web_enabled: boolean;
    slack_enabled: boolean;
    slack_bot_token_set: boolean;
    slack_app_token_set: boolean;
    whatsapp_enabled: boolean;
    whatsapp_bridge_port: number;
    whatsapp_allowed_numbers: string[];
  };
  environment: {
    deployment_mode: string;
    sandbox_mode: string;
    docker_memory: string;
    docker_cpu: number;
    trusted_paths: string[];
  };
  security: {
    active_profile: string;
    approval_timeout: number;
  };
  browser: {
    mode: string;
    brave_key_set: boolean;
  };
  skills: Record<string, boolean>;
  dashboard: {
    port: number;
  };
  memory?: {
    knowledge_graph_enabled: boolean;
    knowledge_graph_provider: string;
    knowledge_graph_auto_process_after_ingest?: boolean;
  };
}

interface Detection {
  ollama: { installed: boolean; running: boolean; models: string[] };
  docker: { installed: boolean; running: boolean; has_sandbox_image: boolean };
  ffmpeg: { installed: boolean };
  playwright: { installed: boolean; has_chromium: boolean };
  node: { installed: boolean; version: string };
  anthropic_key: { found: boolean; source: string };
  brave_key: { found: boolean };
}

// ── Tabs ─────────────────────────────────────────────────────────────────────

const TABS = [
  { id: 'llm', label: 'LLM', icon: Bot },
  { id: 'channels', label: 'Channels', icon: MessageSquare },
  { id: 'environment', label: 'Environment', icon: Container },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'browser', label: 'Browser', icon: Globe },
  { id: 'skills', label: 'Skills', icon: Puzzle },
  { id: 'memory', label: 'Memory', icon: Brain },
  { id: 'dashboard', label: 'Dashboard', icon: SettingsIcon },
] as const;

type TabId = (typeof TABS)[number]['id'];

// ── Skill labels ─────────────────────────────────────────────────────────────

const SKILL_META: Record<string, { label: string; group: string; setup?: string }> = {
  filesystem: { label: 'Filesystem Operations', group: 'Core' },
  shell: { label: 'Shell Commands', group: 'Core' },
  browser: { label: 'Web Browser', group: 'Web', setup: 'Requires Playwright or Brave API key. Configure in Browser tab.' },
  tts: { label: 'Text-to-Speech', group: 'Media', setup: 'Uses Edge-TTS (free, no API key needed). Works out of the box.' },
  stt: { label: 'Speech-to-Text', group: 'Media', setup: 'Uses OpenAI Whisper locally. First run downloads the model (~150MB).' },
  image: { label: 'Image Processing', group: 'Media' },
  video: { label: 'Video Processing', group: 'Media', setup: 'Requires FFmpeg installed on your system.' },
  documents: { label: 'Document Processing', group: 'Media' },
  calendar: { label: 'Calendar (Google)', group: 'Communication', setup: 'Requires Google Calendar API credentials. Set GOOGLE_CREDENTIALS_FILE in .env and complete OAuth flow.' },
  email: { label: 'Email (SMTP/IMAP)', group: 'Communication', setup: 'Requires SMTP/IMAP credentials. Set EMAIL_HOST, EMAIL_USER, EMAIL_PASSWORD in .env file.' },
  sms: { label: 'SMS (Twilio)', group: 'Communication', setup: 'Requires Twilio account. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER in .env file.' },
  memory: { label: 'Memory & Recall', group: 'JARVIS', setup: 'Remember facts, recall memories. No setup needed.' },
  weather: { label: 'Weather', group: 'JARVIS', setup: 'Set WEATHERAPI_KEY for WeatherAPI.com. Uses your IP for location when not specified.' },
  news: { label: 'News', group: 'JARVIS', setup: 'Set NEWS_API_KEY for headlines. RSS fallback available.' },
  finance: { label: 'Finance', group: 'JARVIS', setup: 'Stock/crypto prices via yfinance. No API key needed.' },
  contacts: { label: 'Contacts', group: 'JARVIS', setup: 'Store and search contacts. No setup needed.' },
  tracking: { label: 'Package Tracking', group: 'JARVIS', setup: 'Track packages. Add tracking numbers to the list.' },
  home: { label: 'Smart Home (HA)', group: 'JARVIS', setup: 'Set HA_URL and HA_TOKEN for Home Assistant.' },
  webhook: { label: 'Webhooks', group: 'JARVIS', setup: 'Send HTTP requests to URLs.' },
  agent: { label: 'Autonomous Agents', group: 'JARVIS', setup: 'Research, coding, and data analysis agents.' },
};

// ── Main Component ───────────────────────────────────────────────────────────

export default function Settings() {
  const [activeTab, setActiveTab] = useState<TabId>('llm');

  const { data: config, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['config'],
    queryFn: async () => (await configApi.get()).data as FullConfig,
    retry: 2,
  });

  const { data: detection } = useQuery({
    queryKey: ['detection'],
    queryFn: async () => {
      try {
        return (await configApi.detection()).data as Detection;
      } catch {
        return undefined;
      }
    },
    staleTime: 60000,
    retry: 1,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-6 h-6 animate-spin text-slate-600" />
      </div>
    );
  }

  if (isError || !config) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <div className="glass-card p-10 text-center max-w-md">
          <AlertCircle className="w-12 h-12 mx-auto text-red-400/60 mb-4" />
          <h2 className="text-lg font-semibold text-white mb-2">Failed to load configuration</h2>
          <p className="text-sm text-slate-500 mb-6">
            {(error as any)?.response?.status === 401
              ? 'Session expired. Please log in again.'
              : (error as any)?.message || 'Could not connect to the Aria server. Make sure it is running.'}
          </p>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors text-sm inline-flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Tab sidebar */}
      <div className="w-52 bg-slate-900/40 border-r border-white/[0.06] p-3 space-y-0.5 flex-shrink-0">
        <h2 className="text-[10px] font-semibold text-slate-600 uppercase tracking-widest px-3 py-2">
          Configuration
        </h2>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-gradient-to-r from-blue-600/20 to-blue-500/10 text-blue-400 border border-blue-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-6 lg:p-8">
        <div className="max-w-2xl">
          {activeTab === 'llm' && <LlmTab config={config} detection={detection} />}
          {activeTab === 'channels' && <ChannelsTab config={config} detection={detection} />}
          {activeTab === 'environment' && <EnvironmentTab config={config} detection={detection} />}
          {activeTab === 'security' && <SecurityTab config={config} />}
          {activeTab === 'browser' && <BrowserTab config={config} detection={detection} />}
          {activeTab === 'skills' && <SkillsTab config={config} />}
          {activeTab === 'memory' && <MemoryTab config={config} />}
          {activeTab === 'dashboard' && <DashboardTab config={config} />}
        </div>
      </div>
    </div>
  );
}

// ── Reusable components ──────────────────────────────────────────────────────

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="glass-card p-5 mb-5">
      <h3 className="text-sm font-semibold text-white mb-4">{title}</h3>
      {children}
    </div>
  );
}

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium ${
        ok ? 'bg-green-500/10 text-green-400 border border-green-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'
      }`}
    >
      {ok ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
      {label}
    </span>
  );
}

function RestartBanner({ show }: { show: boolean }) {
  const [restarting, setRestarting] = useState(false);
  if (!show) return null;

  const handleRestart = async () => {
    setRestarting(true);
    try {
      await restartApi.restart();
      setTimeout(() => window.location.reload(), 3000);
    } catch {
      setRestarting(false);
    }
  };

  return (
    <div className="mb-5 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg flex items-center justify-between text-yellow-400 text-sm">
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 flex-shrink-0" />
        Changes saved. Restart Aria for them to take full effect.
      </div>
      <button
        onClick={handleRestart}
        disabled={restarting}
        className="px-3 py-1.5 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-300 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all"
      >
        {restarting ? <Loader2 className="w-3 h-3 animate-spin" /> : <RotateCcw className="w-3 h-3" />}
        {restarting ? 'Restarting...' : 'Restart Now'}
      </button>
    </div>
  );
}

function SaveButton({
  onClick,
  loading,
  disabled,
  error,
}: {
  onClick: () => void;
  loading: boolean;
  disabled?: boolean;
  error?: string | null;
}) {
  return (
    <div className="flex items-center gap-3 mt-2">
      <button
        onClick={onClick}
        disabled={loading || disabled}
        className="btn-primary flex items-center gap-2"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
        Save Changes
      </button>
      {error && (
        <span className="text-sm text-red-400">Save failed. Please try again.</span>
      )}
    </div>
  );
}

function TabHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="mb-6">
      <h2 className="text-xl font-bold text-white tracking-tight">{title}</h2>
      <p className="text-sm text-slate-500 mt-1">{subtitle}</p>
    </div>
  );
}

// ── LLM Tab ──────────────────────────────────────────────────────────────────

function LlmTab({ config, detection }: { config: FullConfig; detection?: Detection }) {
  const queryClient = useQueryClient();
  const [cloudEnabled, setCloudEnabled] = useState(config.llm.cloud_enabled);
  const [ollamaEnabled, setOllamaEnabled] = useState(config.llm.ollama_enabled);
  const [anthropicModel, setAnthropicModel] = useState(config.llm.anthropic_model);
  const [ollamaModel, setOllamaModel] = useState(config.llm.ollama_model);
  const [ollamaUrl, setOllamaUrl] = useState(config.llm.ollama_base_url);
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [needsRestart, setNeedsRestart] = useState(false);

  // Sync state when config refreshes (fixes stale enabled state)
  useEffect(() => {
    setCloudEnabled(config.llm.cloud_enabled);
    setOllamaEnabled(config.llm.ollama_enabled);
    setAnthropicModel(config.llm.anthropic_model);
    setOllamaModel(config.llm.ollama_model);
    setOllamaUrl(config.llm.ollama_base_url);
  }, [config]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const data: Record<string, any> = {
        cloud_enabled: cloudEnabled,
        ollama_enabled: ollamaEnabled,
        anthropic_model: anthropicModel,
        ollama_model: ollamaModel,
        ollama_base_url: ollamaUrl,
      };
      if (apiKey) data.anthropic_api_key = apiKey;
      await configApi.updateLlm(data);
    },
    onSuccess: () => {
      setNeedsRestart(true);
      setApiKey('');
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });

  const validateMutation = useMutation({
    mutationFn: async () => (await configApi.validateKey('anthropic', apiKey)).data,
  });

  return (
    <div>
      <TabHeader title="LLM Providers" subtitle="Configure which language models Aria uses." />
      <RestartBanner show={needsRestart} />

      {detection && (
        <div className="flex gap-2 mb-6 flex-wrap">
          <StatusBadge
            ok={cloudEnabled && detection.anthropic_key.found}
            label={
              !cloudEnabled
                ? 'Claude disabled'
                : detection.anthropic_key.found
                  ? `Claude enabled (${detection.anthropic_key.source})`
                  : 'Claude enabled — no API key'
            }
          />
          <StatusBadge
            ok={ollamaEnabled && detection.ollama.running}
            label={
              !ollamaEnabled
                ? 'Ollama disabled'
                : detection.ollama.running
                  ? 'Ollama running'
                  : detection.ollama.installed
                    ? 'Ollama not running'
                    : 'Ollama not found'
            }
          />
        </div>
      )}

      <SectionCard title="Anthropic Claude">
        <div className="space-y-4">
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-sm text-slate-300">Enable Claude (cloud)</span>
            <input
              type="checkbox"
              checked={cloudEnabled}
              onChange={(e) => setCloudEnabled(e.target.checked)}
              className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-600 focus:ring-blue-500"
            />
          </label>

          {cloudEnabled && (
            <>
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">Model</label>
                <select
                  value={anthropicModel}
                  onChange={(e) => setAnthropicModel(e.target.value)}
                  className="w-full px-3 py-2.5 glass-input"
                >
                  <option value="claude-sonnet-4-20250514">Claude Sonnet 4</option>
                  <option value="claude-opus-4-5-20251101">Claude Opus 4.5</option>
                  <option value="claude-3-5-haiku-20241022">Claude Haiku 3.5</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">
                  API Key{' '}
                  {config.llm.anthropic_key_set && (
                    <span className="text-green-400 normal-case tracking-normal">(set)</span>
                  )}
                </label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <input
                      type={showKey ? 'text' : 'password'}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={
                        config.llm.anthropic_key_set
                          ? 'Enter new key to replace...'
                          : 'sk-ant-...'
                      }
                      className="w-full px-3 py-2.5 glass-input pr-10"
                    />
                    <button
                      onClick={() => setShowKey(!showKey)}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white transition-colors"
                    >
                      {showKey ? (
                        <EyeOff className="w-4 h-4" />
                      ) : (
                        <Eye className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                  <button
                    onClick={() => validateMutation.mutate()}
                    disabled={!apiKey || validateMutation.isPending}
                    className="px-3 py-2 bg-slate-800/60 hover:bg-slate-700/60 disabled:opacity-50 text-white rounded-lg text-sm flex items-center gap-1.5 border border-white/[0.06] transition-all"
                  >
                    {validateMutation.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Key className="w-4 h-4" />
                    )}
                    Validate
                  </button>
                </div>
                {validateMutation.data && (
                  <p
                    className={`text-xs mt-1.5 ${
                      (validateMutation.data as any).valid
                        ? 'text-green-400'
                        : 'text-red-400'
                    }`}
                  >
                    {(validateMutation.data as any).message}
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      </SectionCard>

      <SectionCard title="Ollama (Local)">
        <div className="space-y-4">
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-sm text-slate-300">Enable Ollama (local)</span>
            <input
              type="checkbox"
              checked={ollamaEnabled}
              onChange={(e) => setOllamaEnabled(e.target.checked)}
              className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-600 focus:ring-blue-500"
            />
          </label>

          {ollamaEnabled && (
            <>
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">Model</label>
                {detection?.ollama.models && detection.ollama.models.length > 0 ? (
                  <select
                    value={ollamaModel}
                    onChange={(e) => setOllamaModel(e.target.value)}
                    className="w-full px-3 py-2.5 glass-input"
                  >
                    {detection.ollama.models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                    {!detection.ollama.models.includes(ollamaModel) && (
                      <option value={ollamaModel}>{ollamaModel}</option>
                    )}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={ollamaModel}
                    onChange={(e) => setOllamaModel(e.target.value)}
                    className="w-full px-3 py-2.5 glass-input"
                  />
                )}
              </div>
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">Base URL</label>
                <input
                  type="text"
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  className="w-full px-3 py-2.5 glass-input"
                />
              </div>
            </>
          )}
        </div>
      </SectionCard>

      <SaveButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} />
    </div>
  );
}

// ── Channels Tab ─────────────────────────────────────────────────────────────

function ChannelsTab({
  config,
  detection,
}: {
  config: FullConfig;
  detection?: Detection;
}) {
  const queryClient = useQueryClient();
  const [slackEnabled, setSlackEnabled] = useState(config.channels.slack_enabled);
  const [whatsappEnabled, setWhatsappEnabled] = useState(config.channels.whatsapp_enabled);
  const [botToken, setBotToken] = useState('');
  const [appToken, setAppToken] = useState('');
  const [bridgePort, setBridgePort] = useState(String(config.channels.whatsapp_bridge_port));
  const [allowedNumbers, setAllowedNumbers] = useState(
    (config.channels.whatsapp_allowed_numbers || []).join(', ')
  );
  const [needsRestart, setNeedsRestart] = useState(false);

  useEffect(() => {
    setSlackEnabled(config.channels.slack_enabled);
    setWhatsappEnabled(config.channels.whatsapp_enabled);
    setBridgePort(String(config.channels.whatsapp_bridge_port));
    setAllowedNumbers((config.channels.whatsapp_allowed_numbers || []).join(', '));
  }, [config]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const data: Record<string, any> = {
        slack_enabled: slackEnabled,
        whatsapp_enabled: whatsappEnabled,
        whatsapp_bridge_port: Number(bridgePort),
        whatsapp_allowed_numbers: allowedNumbers
          .split(',')
          .map((n) => n.trim().replace(/^\+/, '').split('@')[0])
          .filter(Boolean),
      };
      if (botToken) data.slack_bot_token = botToken;
      if (appToken) data.slack_app_token = appToken;
      await configApi.updateChannels(data);
    },
    onSuccess: () => {
      setNeedsRestart(true);
      setBotToken('');
      setAppToken('');
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });

  const validateSlack = useMutation({
    mutationFn: async () =>
      (await configApi.validateKey('slack_bot', botToken, { app_token: appToken })).data,
  });

  const testSlack = useMutation({
    mutationFn: async () => (await channelsApi.testSlack()).data,
  });

  const testWhatsapp = useMutation({
    mutationFn: async () => (await channelsApi.testWhatsapp()).data,
  });

  return (
    <div>
      <TabHeader title="Channels" subtitle="Configure messaging channels for Aria." />
      <RestartBanner show={needsRestart} />

      <SectionCard title="Web UI">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-300">Web dashboard</span>
          <StatusBadge ok={true} label="Always enabled" />
        </div>
      </SectionCard>

      <SectionCard title="Slack">
        <div className="space-y-4">
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-sm text-slate-300">Enable Slack</span>
            <input
              type="checkbox"
              checked={slackEnabled}
              onChange={(e) => setSlackEnabled(e.target.checked)}
              className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-600 focus:ring-blue-500"
            />
          </label>

          {slackEnabled && (
            <>
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">
                  Bot Token{' '}
                  {config.channels.slack_bot_token_set && (
                    <span className="text-green-400 normal-case tracking-normal">(set)</span>
                  )}
                </label>
                <input
                  type="password"
                  value={botToken}
                  onChange={(e) => setBotToken(e.target.value)}
                  placeholder={
                    config.channels.slack_bot_token_set
                      ? 'Enter new token to replace...'
                      : 'xoxb-...'
                  }
                  className="w-full px-3 py-2.5 glass-input"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">
                  App Token{' '}
                  {config.channels.slack_app_token_set && (
                    <span className="text-green-400 normal-case tracking-normal">(set)</span>
                  )}
                </label>
                <input
                  type="password"
                  value={appToken}
                  onChange={(e) => setAppToken(e.target.value)}
                  placeholder={
                    config.channels.slack_app_token_set
                      ? 'Enter new token to replace...'
                      : 'xapp-...'
                  }
                  className="w-full px-3 py-2.5 glass-input"
                />
              </div>
              <div className="flex gap-2">
                {botToken && appToken && (
                  <button
                    onClick={() => validateSlack.mutate()}
                    disabled={validateSlack.isPending}
                    className="px-3 py-2 bg-slate-800/60 hover:bg-slate-700/60 text-white rounded-lg text-sm flex items-center gap-1.5 border border-white/[0.06] transition-all"
                  >
                    {validateSlack.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Key className="w-4 h-4" />
                    )}
                    Validate Tokens
                  </button>
                )}
                {config.channels.slack_bot_token_set && (
                  <button
                    onClick={() => testSlack.mutate()}
                    disabled={testSlack.isPending}
                    className="px-3 py-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 rounded-lg text-sm flex items-center gap-1.5 border border-blue-500/20 transition-all"
                  >
                    {testSlack.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Send className="w-4 h-4" />
                    )}
                    Test Connection
                  </button>
                )}
              </div>
              {validateSlack.data && (
                <p
                  className={`text-xs ${
                    (validateSlack.data as any).valid ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {(validateSlack.data as any).valid
                    ? `Connected to: ${(validateSlack.data as any).message}`
                    : (validateSlack.data as any).message}
                </p>
              )}
              {testSlack.data && (
                <p className={`text-xs ${(testSlack.data as any).success ? 'text-green-400' : 'text-red-400'}`}>
                  {(testSlack.data as any).message}
                </p>
              )}
            </>
          )}
        </div>
      </SectionCard>

      <SectionCard title="WhatsApp">
        <div className="space-y-4">
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-sm text-slate-300">Enable WhatsApp</span>
            <input
              type="checkbox"
              checked={whatsappEnabled}
              onChange={(e) => setWhatsappEnabled(e.target.checked)}
              className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-600 focus:ring-blue-500"
            />
          </label>
          {whatsappEnabled && (
            <>
              {detection && !detection.node.installed && (
                <p className="text-xs text-yellow-400 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" /> Node.js is required but not installed
                </p>
              )}
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">Bridge Port</label>
                <input
                  type="number"
                  value={bridgePort}
                  onChange={(e) => setBridgePort(e.target.value)}
                  className="w-full px-3 py-2.5 glass-input"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">
                  Allowed Phone Numbers
                </label>
                <input
                  type="text"
                  value={allowedNumbers}
                  onChange={(e) => setAllowedNumbers(e.target.value)}
                  placeholder="919882278774, 14155551234"
                  className="w-full px-3 py-2.5 glass-input"
                />
                <p className="text-[11px] text-slate-600 mt-1.5">
                  Comma-separated phone numbers with country code (no +). Only these numbers can message Aria. Leave empty to allow all.
                </p>
              </div>
              <WhatsAppQRSection />
              <button
                onClick={() => testWhatsapp.mutate()}
                disabled={testWhatsapp.isPending}
                className="px-3 py-2 bg-green-500/10 hover:bg-green-500/20 text-green-400 rounded-lg text-sm flex items-center gap-1.5 border border-green-500/20 transition-all"
              >
                {testWhatsapp.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                Test WhatsApp Connection
              </button>
              {testWhatsapp.data && (
                <p className={`text-xs ${(testWhatsapp.data as any).success ? 'text-green-400' : 'text-red-400'}`}>
                  {(testWhatsapp.data as any).message}
                </p>
              )}
            </>
          )}
        </div>
      </SectionCard>

      <SaveButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} />
    </div>
  );
}

// ── WhatsApp QR Section ──────────────────────────────────────────────────────

function WhatsAppQRSection() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { data: bridgeStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['whatsapp-status'],
    queryFn: async () => {
      try {
        return (await whatsappApi.getStatus()).data;
      } catch {
        return { ready: false, hasQR: false, bridge_running: false };
      }
    },
    refetchInterval: 5000,
    retry: 1,
  });

  const { data: qrData } = useQuery({
    queryKey: ['whatsapp-qr'],
    queryFn: async () => {
      try {
        return (await whatsappApi.getQR()).data;
      } catch {
        return { qr: null };
      }
    },
    enabled: bridgeStatus?.hasQR === true,
    refetchInterval: 10000,
    retry: 1,
  });

  const startBridge = useMutation({
    mutationFn: async () => (await whatsappApi.start()).data,
    onSuccess: () => {
      setTimeout(() => refetchStatus(), 3000);
    },
  });

  useEffect(() => {
    if (qrData?.qr && canvasRef.current) {
      QRCode.toCanvas(canvasRef.current, qrData.qr, {
        width: 256,
        margin: 2,
        color: { dark: '#000000', light: '#ffffff' },
      });
    }
  }, [qrData?.qr]);

  if (!bridgeStatus || bridgeStatus.bridge_running === false) {
    return (
      <div className="bg-slate-900/40 border border-white/[0.04] rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-slate-800/60 border border-white/[0.06] flex items-center justify-center">
              <Smartphone className="w-4 h-4 text-slate-500" />
            </div>
            <div>
              <p className="text-sm text-slate-400">Bridge not running</p>
              <p className="text-[11px] text-slate-600">
                Start the WhatsApp bridge to connect your phone.
              </p>
            </div>
          </div>
          <button
            onClick={() => startBridge.mutate()}
            disabled={startBridge.isPending}
            className="px-3 py-1.5 bg-green-500/10 hover:bg-green-500/20 text-green-400 rounded-lg text-xs font-medium flex items-center gap-1.5 border border-green-500/20 transition-all"
          >
            {startBridge.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
            {startBridge.isPending ? 'Starting...' : 'Start Bridge'}
          </button>
        </div>
        {startBridge.data && (
          <p className={`text-xs mt-2 ${(startBridge.data as any).success ? 'text-green-400' : 'text-red-400'}`}>
            {(startBridge.data as any).message}
          </p>
        )}
      </div>
    );
  }

  if (bridgeStatus.ready) {
    return (
      <div className="bg-green-500/[0.04] border border-green-500/20 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-green-500/10 border border-green-500/20 flex items-center justify-center">
            <CheckCircle2 className="w-4 h-4 text-green-400" />
          </div>
          <div>
            <p className="text-sm text-green-400 font-medium">WhatsApp Connected</p>
            <p className="text-[11px] text-slate-500">
              Your WhatsApp account is linked and ready.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (bridgeStatus.hasQR && qrData?.qr) {
    return (
      <div className="bg-slate-900/40 border border-white/[0.04] rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Smartphone className="w-4 h-4 text-blue-400" />
          <p className="text-sm text-white font-medium">Scan QR Code</p>
        </div>
        <p className="text-xs text-slate-500 mb-4">
          Open WhatsApp on your phone &gt; Settings &gt; Linked Devices &gt; Link a Device
        </p>
        <div className="flex justify-center">
          <div className="bg-white p-2 rounded-lg">
            <canvas ref={canvasRef} />
          </div>
        </div>
        <p className="text-[10px] text-slate-600 text-center mt-3">
          QR code refreshes automatically
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900/40 border border-white/[0.04] rounded-lg p-4">
      <div className="flex items-center gap-3">
        <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
        <p className="text-sm text-slate-400">Waiting for QR code from bridge...</p>
      </div>
    </div>
  );
}

// ── Docker Controls ──────────────────────────────────────────────────────────

function DockerControls() {
  const [msg, setMsg] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['dockerStatus'],
    queryFn: async () => (await dockerApi.status()).data as { running: boolean; containers: { name: string; status: string; ports: string }[] },
    refetchInterval: 10000,
  });

  const startMutation = useMutation({
    mutationFn: async () => (await dockerApi.start()).data,
    onSuccess: (data: any) => {
      setMsg({ type: data.success ? 'success' : 'error', text: data.message });
      if (data.success) setTimeout(() => refetchStatus(), 5000);
    },
    onError: () => setMsg({ type: 'error', text: 'Failed to start Docker' }),
  });

  const stopMutation = useMutation({
    mutationFn: async () => (await dockerApi.stop()).data,
    onSuccess: (data: any) => {
      setMsg({ type: data.success ? 'info' : 'error', text: data.message });
      setTimeout(() => refetchStatus(), 3000);
    },
    onError: () => setMsg({ type: 'error', text: 'Failed to stop Docker' }),
  });

  return (
    <SectionCard title="Docker Containers">
      <div className="space-y-3">
        {status?.containers && status.containers.length > 0 ? (
          <div className="space-y-1.5">
            {status.containers.map((c) => (
              <div key={c.name} className="flex items-center justify-between bg-slate-900/40 border border-white/[0.04] rounded-lg p-2.5">
                <div>
                  <p className="text-xs font-medium text-white">{c.name}</p>
                  <p className="text-[10px] text-slate-500">{c.status}</p>
                </div>
                <span className="text-[10px] text-slate-500 font-mono">{c.ports}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500">No Aria containers running</p>
        )}

        {msg && (
          <p className={`text-xs ${msg.type === 'success' ? 'text-green-400' : msg.type === 'error' ? 'text-red-400' : 'text-blue-400'}`}>
            {msg.text}
          </p>
        )}

        <div className="flex gap-2">
          <button
            onClick={() => startMutation.mutate()}
            disabled={startMutation.isPending}
            className="flex items-center gap-1.5 px-3 py-2 bg-green-500/10 text-green-400 hover:bg-green-500/20 rounded-lg text-xs font-medium border border-green-500/20 transition-colors"
          >
            {startMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
            Build & Start Containers
          </button>
          {status?.running && (
            <button
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
              className="flex items-center gap-1.5 px-3 py-2 bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded-lg text-xs font-medium border border-red-500/20 transition-colors"
            >
              {stopMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <XCircle className="w-3 h-3" />}
              Stop Containers
            </button>
          )}
          <button
            onClick={() => refetchStatus()}
            className="p-2 btn-ghost"
            title="Refresh status"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </SectionCard>
  );
}

// ── Environment Tab ──────────────────────────────────────────────────────────

function EnvironmentTab({
  config,
  detection,
}: {
  config: FullConfig;
  detection?: Detection;
}) {
  const queryClient = useQueryClient();
  const [deploymentMode, setDeploymentMode] = useState(config.environment.deployment_mode || 'local');
  const [mode, setMode] = useState(config.environment.sandbox_mode);
  const [dockerMem, setDockerMem] = useState(config.environment.docker_memory);
  const [dockerCpu, setDockerCpu] = useState(String(config.environment.docker_cpu));
  const [paths, setPaths] = useState(config.environment.trusted_paths.join(', '));
  const [needsRestart, setNeedsRestart] = useState(false);

  useEffect(() => {
    setDeploymentMode(config.environment.deployment_mode || 'local');
    setMode(config.environment.sandbox_mode);
    setDockerMem(config.environment.docker_memory);
    setDockerCpu(String(config.environment.docker_cpu));
    setPaths(config.environment.trusted_paths.join(', '));
  }, [config]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      await configApi.updateEnvironment({
        deployment_mode: deploymentMode,
        sandbox_mode: mode,
        docker_memory: dockerMem,
        docker_cpu: parseFloat(dockerCpu),
        trusted_paths: paths
          .split(',')
          .map((p) => p.trim())
          .filter(Boolean),
      });
    },
    onSuccess: () => {
      setNeedsRestart(true);
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });

  return (
    <div>
      <TabHeader title="Execution Environment" subtitle="Choose how Aria runs shell commands and code." />
      <RestartBanner show={needsRestart} />

      {detection && (
        <div className="flex gap-2 mb-6 flex-wrap">
          <StatusBadge
            ok={detection.docker.installed && detection.docker.running}
            label={
              detection.docker.running
                ? 'Docker running'
                : detection.docker.installed
                  ? 'Docker stopped'
                  : 'Docker not found'
            }
          />
        </div>
      )}

      <SectionCard title="Deployment Mode">
        <div className="space-y-2">
          <label
            className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer border transition-all ${
              deploymentMode === 'local'
                ? 'border-blue-500/30 bg-blue-500/[0.05]'
                : 'border-white/[0.06] hover:border-white/[0.1]'
            }`}
          >
            <input
              type="radio"
              name="deploy"
              checked={deploymentMode === 'local'}
              onChange={() => setDeploymentMode('local')}
              className="mt-1"
            />
            <div>
              <p className="font-medium text-sm text-white">Local</p>
              <p className="text-xs text-slate-500">
                Aria runs directly on your machine.
              </p>
            </div>
          </label>
          <label
            className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer border transition-all ${
              deploymentMode === 'docker'
                ? 'border-blue-500/30 bg-blue-500/[0.05]'
                : 'border-white/[0.06] hover:border-white/[0.1]'
            }`}
          >
            <input
              type="radio"
              name="deploy"
              checked={deploymentMode === 'docker'}
              onChange={() => setDeploymentMode('docker')}
              className="mt-1"
            />
            <div>
              <p className="font-medium text-sm text-white">Docker</p>
              <p className="text-xs text-slate-500">
                All services run in containers. Auto-starts on next launch.
              </p>
            </div>
          </label>
        </div>
        {deploymentMode === 'docker' && (
          <p className="text-[11px] text-yellow-400/80 mt-3 flex items-center gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
            Switching to Docker mode requires a restart. Aria will run via docker compose on next start.
          </p>
        )}
      </SectionCard>

      <SectionCard title="Sandbox Mode">
        <div className="space-y-2">
          <label
            className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer border transition-all ${
              mode === 'docker'
                ? 'border-blue-500/30 bg-blue-500/[0.05]'
                : 'border-white/[0.06] hover:border-white/[0.1]'
            }`}
          >
            <input
              type="radio"
              name="mode"
              checked={mode === 'docker'}
              onChange={() => setMode('docker')}
              className="mt-1"
            />
            <div>
              <p className="font-medium text-sm text-white">Docker (Sandboxed)</p>
              <p className="text-xs text-slate-500">
                Commands run in an isolated container. Safer but requires Docker.
              </p>
            </div>
          </label>
          <label
            className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer border transition-all ${
              mode === 'direct'
                ? 'border-blue-500/30 bg-blue-500/[0.05]'
                : 'border-white/[0.06] hover:border-white/[0.1]'
            }`}
          >
            <input
              type="radio"
              name="mode"
              checked={mode === 'direct'}
              onChange={() => setMode('direct')}
              className="mt-1"
            />
            <div>
              <p className="font-medium text-sm text-white">Local (Direct)</p>
              <p className="text-xs text-slate-500">
                Commands run directly on your machine. No extra setup needed.
              </p>
            </div>
          </label>
        </div>
      </SectionCard>

      {mode === 'docker' && (
        <>
          {detection && !detection.docker.installed && (
            <SectionCard title="Docker Setup Required">
              <div className="space-y-3">
                <p className="text-xs text-yellow-400 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" /> Docker is not installed on this system
                </p>
                <div className="bg-slate-900/40 border border-white/[0.04] rounded-lg p-4 text-xs text-slate-400 space-y-2">
                  <p className="font-medium text-white text-sm">Installation Steps:</p>
                  <ol className="list-decimal list-inside space-y-1.5 ml-1">
                    <li>Install Docker Desktop from <span className="text-blue-400">docker.com/get-started</span></li>
                    <li>Start Docker Desktop and wait for it to initialize</li>
                    <li>Build the Aria sandbox image:
                      <code className="block mt-1 bg-slate-800/80 px-2.5 py-1.5 rounded text-blue-300 font-mono">docker build -t aria-sandbox:latest -f docker/Dockerfile.sandbox .</code>
                    </li>
                    <li>Restart Aria to use Docker sandboxing</li>
                  </ol>
                  <p className="text-slate-500 mt-2">Alternatively, use <span className="text-white font-medium">docker-compose</span> to run the full stack:</p>
                  <code className="block bg-slate-800/80 px-2.5 py-1.5 rounded text-blue-300 font-mono">docker-compose -f docker/docker-compose.yaml up -d</code>
                </div>
              </div>
            </SectionCard>
          )}
          {detection && detection.docker.installed && !detection.docker.running && (
            <div className="mb-5 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg text-yellow-400 text-sm flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" /> Docker is installed but not running. Start Docker Desktop first.
            </div>
          )}
          <SectionCard title="Docker Settings">
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">Memory Limit</label>
                <input
                  type="text"
                  value={dockerMem}
                  onChange={(e) => setDockerMem(e.target.value)}
                  className="w-full px-3 py-2.5 glass-input"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">CPU Limit</label>
                <input
                  type="text"
                  value={dockerCpu}
                  onChange={(e) => setDockerCpu(e.target.value)}
                  className="w-full px-3 py-2.5 glass-input"
                />
              </div>
            </div>
          </SectionCard>
          <DockerControls />
        </>
      )}

      <SectionCard title="Trusted Paths">
        <div>
          <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">
            Comma-separated list of paths accessible without sandbox
          </label>
          <input
            type="text"
            value={paths}
            onChange={(e) => setPaths(e.target.value)}
            className="w-full px-3 py-2.5 glass-input"
          />
        </div>
      </SectionCard>

      <SaveButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} />
    </div>
  );
}

// ── Security Tab ─────────────────────────────────────────────────────────────

const PROFILE_INFO: Record<string, { desc: string; border: string }> = {
  paranoid: {
    desc: 'Maximum security - every action requires approval',
    border: 'border-red-500/30 bg-red-500/[0.04]',
  },
  balanced: {
    desc: 'Safe defaults with approval for destructive actions (recommended)',
    border: 'border-yellow-500/30 bg-yellow-500/[0.04]',
  },
  trusted: {
    desc: 'Minimal friction - only blocks dangerous operations',
    border: 'border-green-500/30 bg-green-500/[0.04]',
  },
};

const PROFILE_TABLE = [
  { action: 'Read files', paranoid: 'Approve', balanced: 'Auto*', trusted: 'Auto' },
  { action: 'Write files', paranoid: 'Approve', balanced: 'Notify', trusted: 'Auto' },
  { action: 'Delete files', paranoid: 'Approve', balanced: 'Approve', trusted: 'Notify' },
  { action: 'Shell commands', paranoid: 'Approve', balanced: 'Approve', trusted: 'Notify' },
  { action: 'Send messages', paranoid: 'Approve', balanced: 'Approve', trusted: 'Notify' },
  { action: 'Web requests', paranoid: 'Approve', balanced: 'Auto*', trusted: 'Auto' },
  { action: 'Calendar', paranoid: 'Approve', balanced: 'Auto/Notify', trusted: 'Auto' },
  { action: 'Create skills', paranoid: 'Approve', balanced: 'Approve', trusted: 'Notify' },
];

function actionColor(val: string) {
  if (val.startsWith('Approve')) return 'text-red-400';
  if (val.startsWith('Notify')) return 'text-yellow-400';
  return 'text-green-400';
}

function SecurityTab({ config }: { config: FullConfig }) {
  const queryClient = useQueryClient();
  const [profile, setProfile] = useState(config.security.active_profile);
  const [saved, setSaved] = useState(false);

  const saveMutation = useMutation({
    mutationFn: () => configApi.updateSecurity({ active_profile: profile }),
    onSuccess: () => {
      setSaved(true);
      queryClient.invalidateQueries({ queryKey: ['config'] });
      setTimeout(() => setSaved(false), 3000);
    },
  });

  return (
    <div>
      <TabHeader title="Security Profile" subtitle="Controls when Aria asks for your permission." />
      {saved && (
        <div className="mb-5 p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-green-400 text-sm flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" /> Profile updated.
        </div>
      )}

      <div className="space-y-2 mb-6">
        {(['paranoid', 'balanced', 'trusted'] as const).map((p) => (
          <label
            key={p}
            className={`flex items-start gap-3 p-4 rounded-lg cursor-pointer border transition-all ${
              profile === p ? PROFILE_INFO[p].border : 'border-white/[0.06] hover:border-white/[0.1]'
            }`}
          >
            <input
              type="radio"
              name="profile"
              checked={profile === p}
              onChange={() => setProfile(p)}
              className="mt-0.5"
            />
            <div>
              <p className="font-medium text-sm text-white capitalize">
                {p}{' '}
                {p === 'balanced' && (
                  <span className="text-[10px] text-yellow-400 ml-1 font-normal">(recommended)</span>
                )}
              </p>
              <p className="text-xs text-slate-500">{PROFILE_INFO[p].desc}</p>
            </div>
          </label>
        ))}
      </div>

      <SectionCard title="Profile Comparison">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-white/[0.06]">
                <th className="text-left py-2 pr-4 font-medium">Action</th>
                <th className="text-center py-2 px-3 text-red-400/70 font-medium">Paranoid</th>
                <th className="text-center py-2 px-3 text-yellow-400/70 font-medium">Balanced</th>
                <th className="text-center py-2 px-3 text-green-400/70 font-medium">Trusted</th>
              </tr>
            </thead>
            <tbody>
              {PROFILE_TABLE.map((row) => (
                <tr key={row.action} className="border-b border-white/[0.03]">
                  <td className="py-2 pr-4 text-slate-400">{row.action}</td>
                  <td className={`text-center py-2 px-3 ${actionColor(row.paranoid)}`}>
                    {row.paranoid}
                  </td>
                  <td className={`text-center py-2 px-3 ${actionColor(row.balanced)}`}>
                    {row.balanced}
                  </td>
                  <td className={`text-center py-2 px-3 ${actionColor(row.trusted)}`}>
                    {row.trusted}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[10px] text-slate-600 mt-3">
          * Auto actions are restricted to safe paths/domains
        </p>
      </SectionCard>

      <SaveButton
        onClick={() => saveMutation.mutate()}
        loading={saveMutation.isPending}
        disabled={profile === config.security.active_profile}
      />
    </div>
  );
}

// ── Browser Tab ──────────────────────────────────────────────────────────────

function BrowserTab({
  config,
  detection,
}: {
  config: FullConfig;
  detection?: Detection;
}) {
  const queryClient = useQueryClient();
  const [mode, setMode] = useState(config.browser.mode);
  const [braveKey, setBraveKey] = useState('');
  const [needsRestart, setNeedsRestart] = useState(false);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const data: Record<string, any> = { mode };
      if (braveKey) data.brave_api_key = braveKey;
      await configApi.updateBrowser(data);
    },
    onSuccess: () => {
      setNeedsRestart(true);
      setBraveKey('');
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });

  const validateBrave = useMutation({
    mutationFn: async () => (await configApi.validateKey('brave', braveKey)).data,
  });

  return (
    <div>
      <TabHeader title="Browser" subtitle="How Aria accesses the web for browsing and search." />
      <RestartBanner show={needsRestart} />

      {detection && (
        <div className="flex gap-2 mb-6 flex-wrap">
          <StatusBadge
            ok={detection.playwright.installed}
            label={
              detection.playwright.has_chromium
                ? 'Playwright + Chromium'
                : detection.playwright.installed
                  ? 'Playwright (no Chromium)'
                  : 'Playwright not installed'
            }
          />
          <StatusBadge
            ok={detection.brave_key.found || config.browser.brave_key_set}
            label={detection.brave_key.found ? 'Brave API key set' : 'No Brave key'}
          />
        </div>
      )}

      <SectionCard title="Browsing Mode">
        <div className="space-y-2">
          {[
            { value: 'playwright', label: 'Playwright', desc: 'Full browser automation - navigate pages, fill forms, take screenshots.' },
            { value: 'brave', label: 'Brave Search API', desc: 'Search results only, no page interaction. Requires API key.' },
            { value: 'none', label: 'None', desc: 'Disable web browsing entirely.' },
          ].map((opt) => (
            <label
              key={opt.value}
              className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer border transition-all ${
                mode === opt.value
                  ? 'border-blue-500/30 bg-blue-500/[0.05]'
                  : 'border-white/[0.06] hover:border-white/[0.1]'
              }`}
            >
              <input
                type="radio"
                name="browser"
                checked={mode === opt.value}
                onChange={() => setMode(opt.value)}
                className="mt-1"
              />
              <div>
                <p className="font-medium text-sm text-white">{opt.label}</p>
                <p className="text-xs text-slate-500">{opt.desc}</p>
              </div>
            </label>
          ))}
        </div>
      </SectionCard>

      {mode === 'brave' && (
        <SectionCard title="Brave Search API Key">
          <div className="space-y-3">
            <div className="flex gap-2">
              <input
                type="password"
                value={braveKey}
                onChange={(e) => setBraveKey(e.target.value)}
                placeholder={
                  config.browser.brave_key_set ? 'Enter new key to replace...' : 'Enter API key'
                }
                className="flex-1 px-3 py-2.5 glass-input"
              />
              <button
                onClick={() => validateBrave.mutate()}
                disabled={!braveKey || validateBrave.isPending}
                className="px-3 py-2 bg-slate-800/60 hover:bg-slate-700/60 disabled:opacity-50 text-white rounded-lg text-sm flex items-center gap-1.5 border border-white/[0.06] transition-all"
              >
                {validateBrave.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Key className="w-4 h-4" />
                )}
                Validate
              </button>
            </div>
            {validateBrave.data && (
              <p
                className={`text-xs ${
                  (validateBrave.data as any).valid ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {(validateBrave.data as any).message}
              </p>
            )}
          </div>
        </SectionCard>
      )}

      <SaveButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} />
    </div>
  );
}

// ── Skills Tab ───────────────────────────────────────────────────────────────

function SkillsTab({ config }: { config: FullConfig }) {
  const queryClient = useQueryClient();
  const [skills, setSkills] = useState<Record<string, boolean>>({ ...config.skills });
  const [needsRestart, setNeedsRestart] = useState(false);

  useEffect(() => {
    setSkills({ ...config.skills });
  }, [config]);

  const saveMutation = useMutation({
    mutationFn: () => configApi.updateSkills(skills),
    onSuccess: () => {
      setNeedsRestart(true);
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });

  const groups = new Map<string, string[]>();
  for (const [name, meta] of Object.entries(SKILL_META)) {
    if (!groups.has(meta.group)) groups.set(meta.group, []);
    groups.get(meta.group)!.push(name);
  }

  return (
    <div>
      <TabHeader title="Skills" subtitle="Enable or disable Aria's capabilities." />
      <RestartBanner show={needsRestart} />

      {Array.from(groups.entries()).map(([group, skillNames]) => (
        <SectionCard key={group} title={group}>
          <div className="space-y-3">
            {skillNames.map((name) => (
              <div key={name} className="py-1">
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm text-slate-300">{SKILL_META[name]?.label || name}</span>
                  <input
                    type="checkbox"
                    checked={skills[name] ?? false}
                    onChange={(e) =>
                      setSkills((prev) => ({ ...prev, [name]: e.target.checked }))
                    }
                    className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-600 focus:ring-blue-500"
                  />
                </label>
                {skills[name] && SKILL_META[name]?.setup && (
                  <p className="text-[11px] text-slate-500 mt-1 ml-0.5">
                    {SKILL_META[name].setup}
                  </p>
                )}
              </div>
            ))}
          </div>
        </SectionCard>
      ))}

      <SaveButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} />
    </div>
  );
}

// ── Memory Tab ──────────────────────────────────────────────────────────

function MemoryTab({ config }: { config: FullConfig }) {
  const queryClient = useQueryClient();
  const [processing, setProcessing] = useState(false);
  const [processResult, setProcessResult] = useState<string | null>(null);
  const [kgEnabled, setKgEnabled] = useState(config.memory?.knowledge_graph_enabled ?? false);
  const [kgAutoProcess, setKgAutoProcess] = useState(config.memory?.knowledge_graph_auto_process_after_ingest ?? false);
  const [needsRestart, setNeedsRestart] = useState(false);

  useEffect(() => {
    setKgEnabled(config.memory?.knowledge_graph_enabled ?? false);
    setKgAutoProcess(config.memory?.knowledge_graph_auto_process_after_ingest ?? false);
  }, [config]);

  const handleProcessGraph = async () => {
    setProcessing(true);
    setProcessResult(null);
    try {
      const res = await knowledgeApi.processGraph();
      setProcessResult(
        res.data.success
          ? 'Knowledge graph processed successfully.'
          : 'Processing completed but no new knowledge was added.'
      );
    } catch {
      setProcessResult('Failed to process knowledge graph. Make sure cognee is installed: pip install cognee');
    } finally {
      setProcessing(false);
    }
  };

  const saveMutation = useMutation({
    mutationFn: async () => {
      await configApi.updateMemory({
        knowledge_graph_enabled: kgEnabled,
        knowledge_graph_auto_process_after_ingest: kgAutoProcess,
      });
    },
    onSuccess: () => {
      setNeedsRestart(true);
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });

  const kgProvider = config.memory?.knowledge_graph_provider ?? 'cognee';

  return (
    <div>
      <TabHeader title="Memory & Knowledge" subtitle="Configure how Aria stores and retrieves information." />
      <RestartBanner show={needsRestart} />

      <SectionCard title="Vector Memory (ChromaDB)">
        <div className="space-y-2 text-sm">
          <div className="flex justify-between py-1">
            <span className="text-slate-500 text-xs">Provider</span>
            <StatusBadge ok={true} label="ChromaDB" />
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-500 text-xs">Status</span>
            <StatusBadge ok={true} label="Active" />
          </div>
          <p className="text-[11px] text-slate-600 mt-2">
            Vector memory provides semantic search over conversations, documents, and ingested knowledge.
          </p>
        </div>
      </SectionCard>

      <SectionCard title="Knowledge Graph (cognee)">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300 font-medium">Knowledge Graph</p>
              <p className="text-xs text-slate-500">
                Extracts entities, relationships, and structured knowledge using cognee.
              </p>
            </div>
            <label className="cursor-pointer">
              <input
                type="checkbox"
                checked={kgEnabled}
                onChange={(e) => setKgEnabled(e.target.checked)}
                className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-600 focus:ring-blue-500"
              />
            </label>
          </div>

          {!kgEnabled && (
            <div className="bg-slate-900/40 border border-white/[0.04] rounded-lg p-4 text-xs text-slate-500 space-y-1">
              <p>To use the knowledge graph, enable it above and install cognee:</p>
              <code className="block mt-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded text-blue-300 font-mono">pip install cognee</code>
            </div>
          )}

          {kgEnabled && (
            <>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm text-slate-300">Auto-process after ingestion</p>
                  <p className="text-xs text-slate-500">Process knowledge graph automatically when documents are ingested</p>
                </div>
                <label className="cursor-pointer">
                  <input
                    type="checkbox"
                    checked={kgAutoProcess}
                    onChange={(e) => setKgAutoProcess(e.target.checked)}
                    className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-600 focus:ring-blue-500"
                  />
                </label>
              </div>

              <div className="bg-slate-900/40 border border-white/[0.04] rounded-lg p-4 text-xs text-slate-500 space-y-1">
                <p>The knowledge graph processes ingested data to extract:</p>
                <ul className="list-disc list-inside ml-2 space-y-0.5">
                  <li>Named entities (people, places, concepts)</li>
                  <li>Relationships between entities</li>
                  <li>Structured summaries and insights</li>
                </ul>
              </div>

              <button
                onClick={handleProcessGraph}
                disabled={processing}
                className="px-4 py-2.5 bg-gradient-to-r from-purple-600 to-purple-500 hover:from-purple-500 hover:to-purple-400 disabled:opacity-50 text-white rounded-lg font-medium text-sm transition-all shadow-lg shadow-purple-500/20 flex items-center gap-2"
              >
                {processing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Brain className="w-4 h-4" />
                )}
                {processing ? 'Processing...' : 'Process Knowledge Graph'}
              </button>

              {processResult && (
                <p
                  className={`text-xs ${
                    processResult.includes('Failed') ? 'text-red-400' : 'text-green-400'
                  }`}
                >
                  {processResult}
                </p>
              )}
            </>
          )}
        </div>
      </SectionCard>

      <SectionCard title="Memory Settings">
        <div className="space-y-2 text-xs">
          <div className="flex justify-between py-1">
            <span className="text-slate-500">Short-term memory</span>
            <span className="text-slate-300">50 messages max</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-500">Episodic memory</span>
            <span className="text-slate-300">1000 episodes max</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-500">RAG pipeline</span>
            <StatusBadge ok={true} label="Active" />
          </div>
        </div>
      </SectionCard>

      <SaveButton onClick={() => saveMutation.mutate()} loading={saveMutation.isPending} />
    </div>
  );
}

// ── Dashboard Tab ────────────────────────────────────────────────────────────

function DashboardTab({ config }: { config: FullConfig }) {
  const queryClient = useQueryClient();
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [saved, setSaved] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const data: Record<string, any> = {};
      if (newPassword) data.admin_password = newPassword;
      await configApi.updateDashboard(data);
    },
    onSuccess: () => {
      setSaved(true);
      setNewPassword('');
      setConfirmPassword('');
      queryClient.invalidateQueries({ queryKey: ['config'] });
      setTimeout(() => setSaved(false), 3000);
    },
  });

  const handleRestart = async () => {
    setRestarting(true);
    try {
      await restartApi.restart();
      setTimeout(() => window.location.reload(), 3000);
    } catch {
      setRestarting(false);
    }
  };

  const passwordsMatch = newPassword === confirmPassword;

  return (
    <div>
      <TabHeader title="Dashboard" subtitle="Web dashboard settings." />
      {saved && (
        <div className="mb-5 p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-green-400 text-sm flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" /> Saved.
        </div>
      )}

      <SectionCard title="Info">
        <div className="space-y-2 text-xs">
          <div className="flex justify-between py-1">
            <span className="text-slate-500">Port</span>
            <span className="text-slate-300">{config.dashboard.port}</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-500">URL</span>
            <span className="text-slate-300">http://localhost:{config.dashboard.port}</span>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Restart Aria">
        <div className="space-y-3">
          <p className="text-xs text-slate-500">
            Restart the Aria server to apply configuration changes. The page will reload automatically.
          </p>
          <button
            onClick={handleRestart}
            disabled={restarting}
            className="px-4 py-2.5 bg-gradient-to-r from-orange-600 to-red-500 hover:from-orange-500 hover:to-red-400 disabled:opacity-50 text-white rounded-lg font-medium text-sm transition-all shadow-lg shadow-orange-500/20 flex items-center gap-2"
          >
            {restarting ? <Loader2 className="w-4 h-4 animate-spin" /> : <RotateCcw className="w-4 h-4" />}
            {restarting ? 'Restarting Aria...' : 'Restart Aria'}
          </button>
        </div>
      </SectionCard>

      <SectionCard title="Change Admin Password">
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">New Password</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Enter new password"
              className="w-full px-3 py-2.5 glass-input"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">Confirm Password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Confirm new password"
              className="w-full px-3 py-2.5 glass-input"
            />
            {newPassword && confirmPassword && !passwordsMatch && (
              <p className="text-red-400 text-xs mt-1.5">Passwords don't match</p>
            )}
          </div>
        </div>
      </SectionCard>

      <SaveButton
        onClick={() => saveMutation.mutate()}
        loading={saveMutation.isPending}
        disabled={!newPassword || !passwordsMatch}
      />

      <SectionCard title="Reset All Settings">
        <div className="space-y-3">
          <p className="text-xs text-slate-500">
            This will delete all configuration files (settings.yaml, .env) and restart Aria. The setup wizard will run again on next start. Your data (conversations, logs) will be preserved.
          </p>
          {!showResetConfirm ? (
            <button
              onClick={() => setShowResetConfirm(true)}
              className="px-4 py-2.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg font-medium text-sm transition-all border border-red-500/20 flex items-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Reset All Settings
            </button>
          ) : (
            <div className="bg-red-500/[0.06] border border-red-500/20 rounded-lg p-4 space-y-3">
              <p className="text-sm text-red-400 font-medium flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                Are you sure? This cannot be undone.
              </p>
              <p className="text-xs text-slate-500">
                All API keys, tokens, and configuration will be deleted. You will need to re-run the setup wizard.
              </p>
              <div className="flex gap-2">
                <button
                  onClick={async () => {
                    setResetting(true);
                    try {
                      await systemApi.reset();
                      setTimeout(() => window.location.reload(), 3000);
                    } catch {
                      setResetting(false);
                      setShowResetConfirm(false);
                    }
                  }}
                  disabled={resetting}
                  className="px-4 py-2 bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium flex items-center gap-2 transition-all"
                >
                  {resetting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                  {resetting ? 'Resetting...' : 'Yes, Reset Everything'}
                </button>
                <button
                  onClick={() => setShowResetConfirm(false)}
                  className="px-4 py-2 btn-ghost text-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </SectionCard>
    </div>
  );
}
