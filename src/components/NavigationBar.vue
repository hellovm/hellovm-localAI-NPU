<template>
  <nav class="bg-white dark:bg-gray-800 shadow-lg">
    <div class="container mx-auto px-4">
      <div class="flex justify-between items-center py-4">
        <!-- Logo and Title -->
        <div class="flex items-center space-x-4">
          <router-link to="/" class="flex items-center space-x-2">
            <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <span class="text-white font-bold text-sm">AI</span>
            </div>
            <span class="text-xl font-bold text-gray-900 dark:text-white">
              {{ $t('app.title') }}
            </span>
          </router-link>
        </div>

        <!-- Navigation Links -->
        <div class="hidden md:flex items-center space-x-6">
          <router-link 
            v-for="item in navItems" 
            :key="item.path"
            :to="item.path"
            class="flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors"
            :class="$route.path === item.path 
              ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300' 
              : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'"
          >
            <component :is="item.icon" class="w-4 h-4" />
            <span>{{ $t(`nav.${item.key}`) }}</span>
          </router-link>
        </div>

        <!-- Right Side Controls -->
        <div class="flex items-center space-x-4">
          <!-- Language Selector -->
          <LanguageSelector />
          
          <!-- Theme Toggle -->
          <button
            @click="toggleTheme"
            class="p-2 rounded-md text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <SunIcon v-if="isDark" class="w-5 h-5" />
            <MoonIcon v-else class="w-5 h-5" />
          </button>

          <!-- Mobile Menu Toggle -->
          <button
            @click="mobileMenuOpen = !mobileMenuOpen"
            class="md:hidden p-2 rounded-md text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <MenuIcon v-if="!mobileMenuOpen" class="w-5 h-5" />
            <XIcon v-else class="w-5 h-5" />
          </button>
        </div>
      </div>

      <!-- Mobile Menu -->
      <div v-if="mobileMenuOpen" class="md:hidden pb-4">
        <div class="flex flex-col space-y-2">
          <router-link 
            v-for="item in navItems" 
            :key="item.path"
            :to="item.path"
            class="flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors"
            :class="$route.path === item.path 
              ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300' 
              : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'"
            @click="mobileMenuOpen = false"
          >
            <component :is="item.icon" class="w-4 h-4" />
            <span>{{ $t(`nav.${item.key}`) }}</span>
          </router-link>
        </div>
      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { 
  HomeIcon, 
  ChatBubbleLeftRightIcon, 
  CpuChipIcon, 
  ArrowDownTrayIcon,
  PuzzlePieceIcon,
  CogIcon,
  SunIcon,
  MoonIcon,
  MenuIcon,
  XIcon
} from '@heroicons/vue/24/outline'
import LanguageSelector from './LanguageSelector.vue'

const { locale } = useI18n()
const mobileMenuOpen = ref(false)

const navItems = [
  { path: '/', key: 'home', icon: HomeIcon },
  { path: '/chat', key: 'chat', icon: ChatBubbleLeftRightIcon },
  { path: '/models', key: 'models', icon: CpuChipIcon },
  { path: '/downloads', key: 'downloads', icon: ArrowDownTrayIcon },
  { path: '/hardware', key: 'hardware', icon: CpuChipIcon },
  { path: '/plugins', key: 'plugins', icon: PuzzlePieceIcon },
  { path: '/settings', key: 'settings', icon: CogIcon }
]

const isDark = computed(() => document.documentElement.classList.contains('dark'))

function toggleTheme() {
  const html = document.documentElement
  if (html.classList.contains('dark')) {
    html.classList.remove('dark')
    localStorage.setItem('theme', 'light')
  } else {
    html.classList.add('dark')
    localStorage.setItem('theme', 'dark')
  }
}

// Initialize theme
const savedTheme = localStorage.getItem('theme')
if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
  document.documentElement.classList.add('dark')
}
</script>