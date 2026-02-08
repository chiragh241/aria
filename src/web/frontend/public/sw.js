/* Service worker for PWA and push notifications */
const CACHE = 'aria-v1';

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(['/'])));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener('push', (e) => {
  const data = e.data?.json() || { title: 'Aria', body: 'Notification' };
  e.waitUntil(
    self.registration.showNotification(data.title || 'Aria', {
      body: data.body || '',
      icon: '/vite.svg',
      badge: '/vite.svg',
      tag: data.tag || 'aria',
    })
  );
});

self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  e.waitUntil(self.clients.openWindow('/'));
});
