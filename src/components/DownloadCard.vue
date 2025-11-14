<template>
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 mb-4 transition-all duration-200 hover:shadow-lg">
    <div class="flex items-center justify-between mb-3">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
          <DocumentArrowDownIcon class="w-6 h-6 text-blue-600 dark:text-blue-400" />
        </div>
        <div class="flex-1">
          <h3 class="font-semibold text-gray-900 dark:text-white text-sm">{{ task.modelName }}</h3>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            {{ formatFileSize(task.downloadedSize) }} / {{ formatFileSize(task.totalSize) }}
          </p>
        </div>
      </div>
      <div class="flex items-center space-x-2">
        <button
          @click="toggleDownload"
          class="p-2 rounded-lg transition-colors"
          :class="getActionButtonClass()"
        >
          <PlayIcon v-if="task.status === 'paused' || task.status === 'pending'" class="w-4 h-4" />
          <PauseIcon v-else-if="task.status === 'downloading'" class="w-4 h-4" />
          <ArrowPathIcon v-else-if="task.status === 'error'" class="w-4 h-4" />
        </button>
        <button
          @click="cancelDownload"
          class="p-2 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
        >
          <XMarkIcon class="w-4 h-4" />
        </button>
      </div>
    </div>

    <!-- Progress Bar -->
    <div class="mb-3">
      <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
        <span>{{ Math.round(task.progress) }}%</span>
        <span>{{ formatSpeed(task.speed) }} • {{ formatETA(task.eta) }}</span>
      </div>
      <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
        <div
          class="h-full rounded-full transition-all duration-300 ease-out"
          :class="getProgressBarClass()"
          :style="{ width: `${task.progress}%` }"
        ></div>
      </div>
    </div>

    <!-- Multi-threading Info -->
    <div class="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
      <div class="flex items-center space-x-2">
        <span>{{ $t('downloads.threads') }}: {{ task.threads }}</span>
        <span>•</span>
        <span>{{ $t('downloads.connections') }}: {{ task.activeConnections }}/{{ task.maxConnections }}</span>
      </div>
      <div class="flex items-center space-x-1">
        <div
          v-for="i in task.threads"
          :key="i"
          class="w-2 h-2 rounded-full"
          :class="getThreadStatusClass(i)"
        ></div>
      </div>
    </div>

    <!-- Error Message -->
    <div v-if="task.status === 'error' && task.error" class="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
      <p class="text-xs text-red-600 dark:text-red-400">{{ task.error }}</p>
    </div>

    <!-- Resume Info -->
    <div v-if="task.resumable && task.downloadedSize > 0" class="mt-2 text-xs text-green-600 dark:text-green-400">
      <div class="flex items-center space-x-1">
        <CheckCircleIcon class="w-3 h-3" />
        <span>{{ $t('downloads.resumable') }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { 
  DocumentArrowDownIcon, 
  PlayIcon, 
  PauseIcon, 
  ArrowPathIcon, 
  XMarkIcon,
  CheckCircleIcon
} from '@heroicons/vue/24/outline'
import type { DownloadTask } from '@/types'

interface Props {
  task: DownloadTask
}

interface Emits {
  (e: 'toggle', taskId: string): void
  (e: 'cancel', taskId: string): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()
const { t } = useI18n()

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

const formatSpeed = (bytesPerSecond: number): string => {
  if (bytesPerSecond === 0) return '0 B/s'
  return formatFileSize(bytesPerSecond) + '/s'
}

const formatETA = (seconds: number): string => {
  if (seconds === Infinity || seconds <= 0) return '--:--'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  } else {
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }
}

const getActionButtonClass = () => {
  switch (props.task.status) {
    case 'downloading':
      return 'text-yellow-600 hover:text-yellow-700 hover:bg-yellow-50 dark:hover:bg-yellow-900/20'
    case 'paused':
    case 'pending':
      return 'text-green-600 hover:text-green-700 hover:bg-green-50 dark:hover:bg-green-900/20'
    case 'error':
      return 'text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20'
    default:
      return 'text-gray-600 hover:text-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700'
  }
}

const getProgressBarClass = () => {
  switch (props.task.status) {
    case 'downloading':
      return 'bg-blue-600 dark:bg-blue-400'
    case 'paused':
      return 'bg-yellow-500 dark:bg-yellow-600'
    case 'error':
      return 'bg-red-500 dark:bg-red-600'
    case 'completed':
      return 'bg-green-500 dark:bg-green-600'
    default:
      return 'bg-gray-400 dark:bg-gray-500'
  }
}

const getThreadStatusClass = (threadIndex: number) => {
  const activeThreads = Math.ceil((props.task.activeConnections / props.task.maxConnections) * props.task.threads)
  return threadIndex <= activeThreads 
    ? 'bg-blue-500 dark:bg-blue-400' 
    : 'bg-gray-300 dark:bg-gray-600'
}

const toggleDownload = () => {
  emit('toggle', props.task.id)
}

const cancelDownload = () => {
  emit('cancel', props.task.id)
}
</script>