import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../hooks/useAuth';
import { useQuery } from '@tanstack/react-query';
import {
  MessageSquare,
  Shield,
  Puzzle,
  Settings,
  FileText,
  LogOut,
  Bot,
  Activity,
  LayoutDashboard,
  Sun,
  Moon,
  Sparkles,
} from 'lucide-react';
import { systemApi } from '../services/api';
import { useTheme } from '../contexts/ThemeContext';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';

const navItems = [
  { path: '/chat', icon: MessageSquare, label: 'Chat' },
  { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/onboarding', icon: Sparkles, label: 'Setup' },
  { path: '/approvals', icon: Shield, label: 'Approvals' },
  { path: '/skills', icon: Puzzle, label: 'Skills' },
  { path: '/settings', icon: Settings, label: 'Settings' },
  { path: '/logs', icon: FileText, label: 'Logs' },
];

export default function Layout() {
  const { logout, user } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();
  const { theme, setTheme, resolved } = useTheme();

  useKeyboardShortcuts([
    { key: '1', meta: true, handler: () => navigate('/chat') },
    { key: '2', meta: true, handler: () => navigate('/dashboard') },
    { key: '3', meta: true, handler: () => navigate('/dashboard') },
    { key: 'k', meta: true, handler: () => navigate('/chat') },
  ]);

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      try {
        const res = await systemApi.healthCheck();
        return res.data;
      } catch {
        return null;
      }
    },
    refetchInterval: 30000,
    retry: 1,
  });

  const isHealthy = health?.status === 'healthy';

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="flex h-screen bg-[#0a1120]">
      {/* Sidebar */}
      <aside className="w-[260px] glass-sidebar flex flex-col flex-shrink-0">
        {/* Logo */}
        <div className="p-5 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 via-blue-600 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-[15px] font-bold text-white tracking-tight">Aria</h1>
              <div className="flex items-center gap-1.5">
                <div className={isHealthy ? 'status-online' : 'w-2 h-2 bg-slate-600 rounded-full'} />
                <p className="text-[11px] text-slate-500">
                  {isHealthy ? 'Online' : 'Connecting...'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-0.5 mt-1">
          <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-widest px-3 mb-2">
            Navigation
          </p>
          {navItems.map((item) => {
            const isActive = location.pathname === item.path ||
              (item.path === '/chat' && (location.pathname === '/' || location.pathname === '/chat'));
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-gradient-to-r from-blue-600/20 to-blue-500/10 text-blue-400 border border-blue-500/20 shadow-sm shadow-blue-500/5'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]'
                }`}
              >
                <item.icon className={`w-[18px] h-[18px] ${isActive ? 'text-blue-400' : ''}`} />
                <span>{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-400" />
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* System status */}
        <div className="mx-3 mb-3">
          <div className="glass-card p-3">
            <div className="flex items-center gap-2 text-xs">
              <Activity className="w-3.5 h-3.5 text-slate-500" />
              <span className="text-slate-500">System</span>
              <span className={`ml-auto text-[10px] font-medium px-1.5 py-0.5 rounded-full ${
                isHealthy
                  ? 'bg-green-500/10 text-green-400'
                  : 'bg-yellow-500/10 text-yellow-400'
              }`}>
                {isHealthy ? 'Healthy' : 'Checking'}
              </span>
            </div>
          </div>
        </div>

        {/* Theme toggle */}
        <div className="px-3 py-2 flex items-center justify-between">
          <span className="text-xs text-slate-500">Theme</span>
          <button
            onClick={() => setTheme(resolved === 'dark' ? 'light' : 'dark')}
            className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] transition-all"
            title={`Switch to ${resolved === 'dark' ? 'light' : 'dark'} mode`}
          >
            {resolved === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>

        {/* User section */}
        <div className="p-3 border-t border-white/[0.06]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-slate-700 to-slate-600 flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-semibold text-white">
                  {user?.charAt(0).toUpperCase()}
                </span>
              </div>
              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-200 truncate">{user}</p>
                <p className="text-[10px] text-slate-600">Administrator</p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto bg-[#0a1120]">
        <Outlet />
      </main>
    </div>
  );
}
