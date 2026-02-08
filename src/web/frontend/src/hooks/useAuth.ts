import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '../services/api';

interface AuthState {
  token: string | null;
  user: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      isAuthenticated: false,

      login: async (username: string, password: string) => {
        const response = await api.post('/auth/login', { username, password });
        const { access_token } = response.data;

        api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

        set({
          token: access_token,
          user: username,
          isAuthenticated: true,
        });
      },

      logout: () => {
        delete api.defaults.headers.common['Authorization'];
        set({
          token: null,
          user: null,
          isAuthenticated: false,
        });
      },
    }),
    {
      name: 'aria-auth',
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
      onRehydrateStorage: () => (state) => {
        if (state?.token) {
          api.defaults.headers.common['Authorization'] = `Bearer ${state.token}`;
        }
      },
    }
  )
);
