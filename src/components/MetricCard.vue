<template>
  <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
    <div class="flex items-center justify-between mb-2">
      <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">
        {{ title }}
      </h3>
      <div 
        class="flex items-center space-x-1"
        :class="trendColor"
      >
        <component 
          :is="trendIcon" 
          class="w-4 h-4" 
        />
        <span class="text-xs font-medium">{{ trendLabel }}</span>
      </div>
    </div>
    
    <div class="flex items-baseline space-x-1">
      <span class="text-2xl font-bold text-gray-900 dark:text-white">
        {{ value }}
      </span>
      <span class="text-sm text-gray-500 dark:text-gray-400">
        {{ unit }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { 
  ArrowUpIcon, 
  ArrowDownIcon, 
  MinusIcon 
} from '@heroicons/vue/20/solid'

const props = defineProps<{
  title: string
  value: string | number
  unit: string
  trend: 'up' | 'down' | 'stable'
}>()

const trendIcon = computed(() => {
  switch (props.trend) {
    case 'up':
      return ArrowUpIcon
    case 'down':
      return ArrowDownIcon
    case 'stable':
      return MinusIcon
    default:
      return MinusIcon
  }
})

const trendColor = computed(() => {
  switch (props.trend) {
    case 'up':
      return 'text-green-500'
    case 'down':
      return 'text-red-500'
    case 'stable':
      return 'text-gray-500'
    default:
      return 'text-gray-500'
  }
})

const trendLabel = computed(() => {
  switch (props.trend) {
    case 'up':
      return '↑'
    case 'down':
      return '↓'
    case 'stable':
      return '→'
    default:
      return '→'
  }
})
</script>