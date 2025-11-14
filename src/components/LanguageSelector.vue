<template>
  <div class="relative">
    <button
      @click="isOpen = !isOpen"
      class="flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
    >
      <GlobeIcon class="w-4 h-4" />
      <span>{{ currentLanguage.label }}</span>
      <ChevronDownIcon class="w-4 h-4" />
    </button>

    <!-- Dropdown Menu -->
    <div 
      v-if="isOpen" 
      class="absolute right-0 mt-2 w-32 bg-white dark:bg-gray-800 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 z-50"
    >
      <div class="py-1">
        <button
          v-for="lang in languages"
          :key="lang.code"
          @click="selectLanguage(lang.code)"
          class="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          :class="{ 'bg-blue-50 dark:bg-blue-900 text-blue-700 dark:text-blue-300': locale === lang.code }"
        >
          {{ lang.label }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { GlobeIcon, ChevronDownIcon } from '@heroicons/vue/24/outline'

const { locale } = useI18n()
const isOpen = ref(false)

const languages = [
  { code: 'en', label: 'English' },
  { code: 'zh', label: '中文' }
]

const currentLanguage = computed(() => 
  languages.find(lang => lang.code === locale.value) || languages[0]
)

function selectLanguage(code: string) {
  locale.value = code
  localStorage.setItem('language', code)
  isOpen.value = false
}

// Close dropdown when clicking outside
const handleClickOutside = (event: MouseEvent) => {
  const target = event.target as Element
  if (!target.closest('.relative')) {
    isOpen.value = false
  }
}

if (typeof window !== 'undefined') {
  document.addEventListener('click', handleClickOutside)
}

// Initialize language from localStorage
const savedLanguage = localStorage.getItem('language')
if (savedLanguage && languages.some(lang => lang.code === savedLanguage)) {
  locale.value = savedLanguage
}
</script>