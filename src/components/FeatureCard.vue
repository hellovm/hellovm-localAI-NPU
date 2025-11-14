<template>
  <router-link 
    :to="link"
    class="block bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 p-6 group"
  >
    <div class="flex items-center justify-center w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg mb-4 group-hover:bg-blue-200 dark:group-hover:bg-blue-800 transition-colors">
      <span class="text-2xl">{{ icon }}</span>
    </div>
    
    <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">
      {{ title }}
    </h3>
    
    <p class="text-gray-600 dark:text-gray-300 text-sm mb-4">
      {{ description }}
    </p>
    
    <div class="flex items-center text-blue-600 dark:text-blue-400 group-hover:text-blue-700 dark:group-hover:text-blue-300 transition-colors">
      <span class="text-sm font-medium">{{ $t('app.settings') }}</span>
      <ArrowRightIcon class="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
    </div>
  </router-link>
</template>

<script setup lang="ts">
import { ArrowRightIcon } from '@heroicons/vue/20/solid'

defineProps<{
  icon: string
  title: string
  description: string
  link: string
}>()
</script>