<template>
  <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
    <div class="flex items-center justify-between mb-2">
      <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">
        {{ title }}
      </h3>
      <div 
        class="w-3 h-3 rounded-full"
        :class="statusClass"
      />
    </div>
    <p class="text-2xl font-bold text-gray-900 dark:text-white">
      {{ value }}
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  title: string
  value: string
  status: 'online' | 'offline' | 'active' | 'idle' | 'error'
}>()

const statusClass = computed(() => {
  switch (props.status) {
    case 'online':
      return 'bg-green-500'
    case 'offline':
      return 'bg-red-500'
    case 'active':
      return 'bg-blue-500'
    case 'idle':
      return 'bg-yellow-500'
    case 'error':
      return 'bg-red-500'
    default:
      return 'bg-gray-500'
  }
})
</script>