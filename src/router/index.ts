import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@views/Home.vue')
  },
  {
    path: '/chat',
    name: 'Chat',
    component: () => import('@views/Chat.vue')
  },
  {
    path: '/models',
    name: 'Models',
    component: () => import('@views/Models.vue')
  },
  {
    path: '/downloads',
    name: 'Downloads',
    component: () => import('@views/Downloads.vue')
  },
  {
    path: '/hardware',
    name: 'Hardware',
    component: () => import('@views/Hardware.vue')
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('@views/Settings.vue')
  },
  {
    path: '/plugins',
    name: 'Plugins',
    component: () => import('@views/Plugins.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router