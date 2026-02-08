import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: add token from persisted auth (avoids 401 race on rehydration)
api.interceptors.request.use((config) => {
  try {
    const stored = localStorage.getItem('aria-auth');
    if (stored) {
      const parsed = JSON.parse(stored);
      const token = parsed?.state?.token ?? parsed?.token;
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
  } catch {
    // ignore
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear auth and redirect to login
      localStorage.removeItem('aria-auth');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;

// API functions
export const userApi = {
  getProfile: () =>
    api.get('/user/profile'),
};

export const chatApi = {
  sendMessage: (content: string) =>
    api.post('/chat/message', { content, channel: 'web' }),

  getHistory: (limit = 50) =>
    api.get('/chat/history', { params: { limit } }),

  clearHistory: () =>
    api.delete('/chat/history'),
};

export const approvalsApi = {
  getPending: () =>
    api.get('/approvals/pending'),

  respond: (approvalId: string, approved: boolean) =>
    api.post('/approvals/respond', { approval_id: approvalId, approved }),
};

export const skillsApi = {
  list: () =>
    api.get('/skills'),

  get: (name: string) =>
    api.get(`/skills/${name}`),

  toggle: (skillName: string, enabled: boolean) =>
    api.post('/skills/toggle', { skill_name: skillName, enabled }),

  getCredentials: (name: string) =>
    api.get(`/skills/${name}/credentials`),

  saveCredentials: (name: string, credentials: Record<string, string>) =>
    api.post(`/skills/${name}/credentials`, { credentials }),

  testConnection: (name: string) =>
    api.post(`/skills/${name}/test`),
};

export const settingsApi = {
  get: () =>
    api.get('/settings'),

  update: (settings: Record<string, any>) =>
    api.post('/settings', settings),

  getProfiles: () =>
    api.get('/settings/profiles'),
};

export const auditApi = {
  getLog: (limit = 100, offset = 0, event?: string) =>
    api.get('/audit', { params: { limit, offset, event } }),

  getStats: () =>
    api.get('/audit/stats'),
};

export const systemApi = {
  getStatus: () =>
    api.get('/system/status'),

  healthCheck: () =>
    api.get('/system/health'),

  reset: () =>
    api.post('/system/reset'),
};

export const featuresApi = {
  list: () =>
    api.get('/features'),
  toggle: (featureId: string, enabled: boolean) =>
    api.put(`/features/${featureId}`, { enabled }),
};

export const pushApi = {
  subscribe: (subscription: object) =>
    api.post('/push/subscribe', subscription),
};

export const debugApi = {
  getTrace: () =>
    api.get('/debug/trace'),
};

export const skillTemplatesApi = {
  list: () =>
    api.get('/skill-templates'),
};

export const widgetsApi = {
  list: () =>
    api.get('/widgets'),
  update: (widgets: object[]) =>
    api.put('/widgets', { widgets }),
};

export const usageApi = {
  getStats: () =>
    api.get('/usage'),
};

export const exportApi = {
  export: (type: 'all' | 'conversations' | 'audit' = 'all') =>
    api.get('/export', { params: { type }, responseType: 'blob' }),
};

export const hudApi = {
  getVitals: () =>
    api.get('/hud/vitals'),

  getTimeline: () =>
    api.get('/hud/timeline'),

  getAgents: () =>
    api.get('/hud/agents'),

  getAllAgentsFull: () =>
    api.get('/hud/agents/full'),
};

export const agentsApi = {
  list: () =>
    api.get('/agents'),

  run: (task: string, agentType?: string) =>
    api.post('/agents/run', { task, agent_type: agentType }),
};

export const knowledgeApi = {
  processGraph: () =>
    api.post('/knowledge/process'),
};

export const whatsappApi = {
  getStatus: () =>
    api.get('/whatsapp/status'),

  getQR: () =>
    api.get('/whatsapp/qr'),

  start: () =>
    api.post('/whatsapp/start'),
};

export const channelsApi = {
  testSlack: () =>
    api.post('/channels/test-slack'),

  testWhatsapp: () =>
    api.post('/channels/test-whatsapp'),
};

export const transcribeApi = {
  transcribe: (audio: string, format: string = 'webm') =>
    api.post('/chat/transcribe', { audio, format }),
};

export const logApi = {
  event: (event: string, details?: Record<string, unknown>) =>
    api.post('/log/event', { event, details: details || {} }),
};

export const restartApi = {
  restart: () =>
    api.post('/system/restart'),
};

export const dockerApi = {
  start: () =>
    api.post('/system/docker/start'),

  stop: () =>
    api.post('/system/docker/stop'),

  status: () =>
    api.get('/system/docker/status'),
};

// ── Configuration management API ─────────────────────────────────────────────

export const configApi = {
  /** Get full configuration state */
  get: () =>
    api.get('/config'),

  /** Update LLM config */
  updateLlm: (data: Record<string, any>) =>
    api.put('/config/llm', { data }),

  /** Update channels config */
  updateChannels: (data: Record<string, any>) =>
    api.put('/config/channels', { data }),

  /** Update environment config */
  updateEnvironment: (data: Record<string, any>) =>
    api.put('/config/environment', { data }),

  /** Update security config */
  updateSecurity: (data: Record<string, any>) =>
    api.put('/config/security', { data }),

  /** Update browser config */
  updateBrowser: (data: Record<string, any>) =>
    api.put('/config/browser', { data }),

  /** Update skills config */
  updateSkills: (data: Record<string, any>) =>
    api.put('/config/skills', { data }),

  /** Update integrations (Notion, Todoist, Linear, Spotify) */
  updateIntegrations: (data: Record<string, any>) =>
    api.put('/config/integrations', { data }),

  /** Update dashboard config */
  updateDashboard: (data: Record<string, any>) =>
    api.put('/config/dashboard', { data }),

  /** Update memory/knowledge graph config */
  updateMemory: (data: Record<string, any>) =>
    api.put('/config/memory', { data }),

  /** Validate an API key */
  validateKey: (keyType: string, keyValue: string, extra?: Record<string, string>) =>
    api.post('/config/validate-key', { key_type: keyType, key_value: keyValue, extra }),

  /** Run system detection */
  detection: () =>
    api.get('/config/detection'),
};
