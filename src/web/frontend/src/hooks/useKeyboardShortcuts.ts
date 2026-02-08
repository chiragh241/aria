import { useEffect } from 'react';

type Shortcut = { key: string; meta?: boolean; ctrl?: boolean; shift?: boolean; handler: () => void };

export function useKeyboardShortcuts(shortcuts: Shortcut[], enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const handle = (e: KeyboardEvent) => {
      for (const s of shortcuts) {
        const key = s.key.toLowerCase();
        const meta = s.meta ?? false;
        const ctrl = s.ctrl ?? false;
        const shift = s.shift ?? false;
        const match =
          e.key.toLowerCase() === key &&
          (meta ? e.metaKey : !e.metaKey) &&
          (ctrl ? e.ctrlKey : !e.ctrlKey) &&
          (shift ? e.shiftKey : !e.shiftKey);
        if (match) {
          e.preventDefault();
          s.handler();
          return;
        }
      }
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [enabled, shortcuts]);
}
