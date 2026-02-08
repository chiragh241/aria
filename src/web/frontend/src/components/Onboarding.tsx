import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Bot, Key, Puzzle, CheckCircle2, Loader2, ArrowRight, ArrowLeft } from 'lucide-react';
import { configApi } from '../services/api';

const ONBOARDING_DONE_KEY = 'aria-onboarding-done';

export default function Onboarding() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [step, setStep] = useState(1);
  const [skills, setSkills] = useState<Record<string, boolean>>({});
  const [integrations, setIntegrations] = useState({
    notion: { enabled: false, api_key: '' },
    todoist: { enabled: false, api_key: '' },
    linear: { enabled: false, api_key: '' },
    spotify: { enabled: false, client_id: '', client_secret: '' },
  });

  const { data: config, isLoading } = useQuery({
    queryKey: ['config'],
    queryFn: async () => (await configApi.get()).data as any,
    retry: 2,
  });

  const skillsMutation = useMutation({
    mutationFn: () => configApi.updateSkills(skills),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['config'] }),
  });

  const integrationsMutation = useMutation({
    mutationFn: () =>
      configApi.updateIntegrations({
        notion_enabled: integrations.notion.enabled,
        ...(integrations.notion.api_key && { notion_api_key: integrations.notion.api_key }),
        todoist_enabled: integrations.todoist.enabled,
        ...(integrations.todoist.api_key && { todoist_api_key: integrations.todoist.api_key }),
        linear_enabled: integrations.linear.enabled,
        ...(integrations.linear.api_key && { linear_api_key: integrations.linear.api_key }),
        spotify_enabled: integrations.spotify.enabled,
        ...(integrations.spotify.client_id && { spotify_client_id: integrations.spotify.client_id }),
        ...(integrations.spotify.client_secret && { spotify_client_secret: integrations.spotify.client_secret }),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['config'] }),
  });

  const SKILL_GROUPS: Record<string, { label: string; skills: string[] }> = {
    Core: { label: 'Core', skills: ['filesystem', 'shell', 'browser', 'memory'] },
    Media: { label: 'Media', skills: ['tts', 'stt', 'image', 'video', 'documents'] },
    Communication: { label: 'Communication', skills: ['calendar', 'email', 'sms'] },
    JARVIS: { label: 'JARVIS', skills: ['weather', 'news', 'finance', 'contacts', 'tracking', 'home', 'webhook', 'agent', 'research'] },
    Integrations: { label: 'Integrations', skills: ['notion', 'todoist', 'linear', 'spotify'] },
  };

  if (isLoading || !config) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
        <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
      </div>
    );
  }

  useEffect(() => {
    if (!config) return;
    const keys = Object.values(SKILL_GROUPS).flatMap((g) => g.skills);
    setSkills((prev) =>
      Object.keys(prev).length ? prev : Object.fromEntries(keys.map((k) => [k, config.skills?.[k] ?? false]))
    );
  }, [config]);

  const handleSkillsSave = async () => {
    await skillsMutation.mutateAsync();
    setStep(2);
  };

  const handleIntegrationsSave = async () => {
    await integrationsMutation.mutateAsync();
    localStorage.setItem(ONBOARDING_DONE_KEY, 'true');
    navigate('/settings');
  };

  const handleSkip = () => {
    localStorage.setItem(ONBOARDING_DONE_KEY, 'true');
    navigate('/');
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ backgroundColor: 'var(--bg-primary)' }}>
      <div className="w-full max-w-2xl">
        <div className="text-center mb-10">
          <div className="w-14 h-14 mx-auto rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mb-4">
            <Bot className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-theme-primary">Welcome to Aria</h1>
          <p className="text-theme-secondary mt-1">Set up your assistant in a few steps</p>
          <div className="flex justify-center gap-2 mt-4">
            {[1, 2].map((s) => (
              <div
                key={s}
                className={`w-2 h-2 rounded-full ${step === s ? 'bg-blue-500' : ''}`}
                style={step !== s ? { backgroundColor: 'var(--text-secondary)' } : undefined}
              />
            ))}
          </div>
        </div>

        <div className="glass-card p-6">
          {step === 1 && (
            <>
              <div className="flex items-center gap-3 mb-4">
                <Puzzle className="w-5 h-5 text-blue-400" />
                <h2 className="text-lg font-semibold text-theme-primary">Select skills</h2>
              </div>
              <p className="text-sm text-theme-secondary mb-4">Choose which capabilities Aria should have.</p>
              <div className="space-y-4 max-h-[400px] overflow-y-auto">
                {Object.entries(SKILL_GROUPS).map(([groupKey, { label, skills: skillList }]) => (
                  <div key={groupKey}>
                    <p className="text-xs font-medium text-theme-secondary uppercase tracking-wider mb-2">{label}</p>
                    <div className="flex flex-wrap gap-2">
                      {skillList.map((name) => (
                        <label
                          key={name}
                          className="flex items-center gap-2 px-3 py-2 rounded-lg card-theme cursor-pointer hover:border-blue-500/30 border-theme"
                        >
                          <input
                            type="checkbox"
                            checked={skills[name] ?? false}
                            onChange={(e) => setSkills((p) => ({ ...p, [name]: e.target.checked }))}
                            className="rounded border border-theme bg-theme-muted text-blue-600"
                          />
                          <span className="text-sm text-theme-primary capitalize">{name.replace(/_/g, ' ')}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex justify-between mt-6">
                <button onClick={handleSkip} className="text-theme-secondary hover:text-theme-primary text-sm">
                  Skip for now
                </button>
                <button
                  onClick={handleSkillsSave}
                  disabled={skillsMutation.isPending}
                  className="px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium text-sm flex items-center gap-2"
                >
                  {skillsMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                  Continue
                </button>
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <div className="flex items-center gap-3 mb-4">
                <Key className="w-5 h-5 text-blue-400" />
                <h2 className="text-lg font-semibold text-theme-primary">Integrations</h2>
              </div>
              <p className="text-sm text-theme-secondary mb-4">Add API keys for Notion, Todoist, Linear, or Spotify.</p>
              <div className="space-y-4">
                {(['notion', 'todoist', 'linear'] as const).map((name) => (
                  <div key={name} className="flex items-center gap-4">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={integrations[name].enabled}
                        onChange={(e) =>
                          setIntegrations((p) => ({
                            ...p,
                            [name]: { ...p[name], enabled: e.target.checked },
                          }))
                        }
                        className="rounded border border-theme bg-theme-muted text-blue-600"
                      />
                      <span className="text-sm text-theme-primary capitalize">{name}</span>
                    </label>
                    <input
                      type="password"
                      placeholder={`${name} API key`}
                      value={integrations[name].api_key}
                      onChange={(e) =>
                        setIntegrations((p) => ({
                          ...p,
                          [name]: { ...p[name], api_key: e.target.value },
                        }))
                      }
                      className="flex-1 px-3 py-2 rounded-lg glass-input text-sm"
                    />
                  </div>
                ))}
                <div className="flex items-center gap-4">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={integrations.spotify.enabled}
                      onChange={(e) =>
                        setIntegrations((p) => ({
                          ...p,
                          spotify: { ...p.spotify, enabled: e.target.checked },
                        }))
                      }
                      className="rounded border border-theme bg-theme-muted text-blue-600"
                    />
                    <span className="text-sm text-theme-primary">Spotify</span>
                  </label>
                  <input
                    type="password"
                    placeholder="Client ID"
                    value={integrations.spotify.client_id}
                    onChange={(e) =>
                      setIntegrations((p) => ({
                        ...p,
                        spotify: { ...p.spotify, client_id: e.target.value },
                      }))
                    }
                    className="flex-1 px-3 py-2 rounded-lg glass-input text-sm"
                  />
                  <input
                    type="password"
                    placeholder="Client Secret"
                    value={integrations.spotify.client_secret}
                    onChange={(e) =>
                      setIntegrations((p) => ({
                        ...p,
                        spotify: { ...p.spotify, client_secret: e.target.value },
                      }))
                    }
                    className="flex-1 px-3 py-2 rounded-lg glass-input text-sm"
                  />
                </div>
              </div>
              <div className="flex justify-between mt-6">
                <button
                  onClick={() => setStep(1)}
                  className="text-theme-secondary hover:text-theme-primary flex items-center gap-1"
                >
                  <ArrowLeft className="w-4 h-4" /> Back
                </button>
                <div className="flex gap-2">
                  <button onClick={handleSkip} className="px-4 py-2.5 text-theme-secondary hover:text-theme-primary text-sm">
                    Skip
                  </button>
                  <button
                    onClick={handleIntegrationsSave}
                    disabled={integrationsMutation.isPending}
                    className="px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium text-sm flex items-center gap-2"
                  >
                    {integrationsMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
                    Finish
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        <p className="text-center text-theme-secondary text-xs mt-4">
          You can always change these in Settings.
        </p>
      </div>
    </div>
  );
}
