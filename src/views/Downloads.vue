<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="mb-8">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl font-bold text-gray-900 dark:text-white">{{ $t('downloads.title') }}</h1>
            <p class="mt-2 text-gray-600 dark:text-gray-300">{{ $t('downloads.subtitle') }}</p>
          </div>
          <div class="flex items-center space-x-4">
            <!-- Global Controls -->
            <button
              @click="toggleAllDownloads"
              class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center space-x-2"
            >
              <PauseIcon v-if="hasActiveDownloads" class="w-4 h-4" />
              <PlayIcon v-else class="w-4 h-4" />
              <span>{{ hasActiveDownloads ? $t('downloads.pauseAll') : $t('downloads.resumeAll') }}</span>
            </button>
            <button
              @click="clearCompleted"
              class="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors flex items-center space-x-2"
            >
              <TrashIcon class="w-4 h-4" />
              <span>{{ $t('downloads.clearCompleted') }}</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Stats Cards -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div class="flex items-center">
            <div class="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg">
              <DocumentArrowDownIcon class="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div class="ml-4">
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300">{{ $t('downloads.activeTasks') }}</p>
              <p class="text-2xl font-semibold text-gray-900 dark:text-white">{{ activeTasksCount }}</p>
            </div>
          </div>
        </div>

        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div class="flex items-center">
            <div class="p-3 bg-green-100 dark:bg-green-900 rounded-lg">
              <CheckCircleIcon class="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
            <div class="ml-4">
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300">{{ $t('downloads.completed') }}</p>
              <p class="text-2xl font-semibold text-gray-900 dark:text-white">{{ completedTasksCount }}</p>
            </div>
          </div>
        </div>

        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div class="flex items-center">
            <div class="p-3 bg-yellow-100 dark:bg-yellow-900 rounded-lg">
              <ClockIcon class="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div class="ml-4">
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300">{{ $t('downloads.totalSpeed') }}</p>
              <p class="text-2xl font-semibold text-gray-900 dark:text-white">{{ formatTotalSpeed }}</p>
            </div>
          </div>
        </div>

        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div class="flex items-center">
            <div class="p-3 bg-purple-100 dark:bg-purple-900 rounded-lg">
              <CpuChipIcon class="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div class="ml-4">
              <p class="text-sm font-medium text-gray-600 dark:text-gray-300">{{ $t('downloads.threadUsage') }}</p>
              <p class="text-2xl font-semibold text-gray-900 dark:text-white">{{ totalThreadUsage }}%</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Filters and Search -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
        <div class="flex flex-col sm:flex-row gap-4">
          <div class="flex-1">
            <div class="relative">
              <MagnifyingGlassIcon class="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                v-model="searchQuery"
                type="text"
                :placeholder="$t('downloads.searchPlaceholder')"
                class="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
          <select
            v-model="statusFilter"
            class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">{{ $t('downloads.allStatus') }}</option>
            <option value="downloading">{{ $t('downloads.downloading') }}</option>
            <option value="paused">{{ $t('downloads.paused') }}</option>
            <option value="completed">{{ $t('downloads.completed') }}</option>
            <option value="error">{{ $t('downloads.error') }}</option>
          </select>
        </div>
      </div>

      <!-- Download Tasks -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md">
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
              {{ $t('downloads.downloadTasks') }} ({{ filteredTasks.length }})
            </h2>
            <div class="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
              <span>{{ $t('downloads.maxConcurrent') }}: {{ maxConcurrentDownloads }}</span>
              <button
                @click="showSettings = true"
                class="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <CogIcon class="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        <div class="p-6">
          <div v-if="filteredTasks.length === 0" class="text-center py-12">
            <InboxIcon class="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">{{ $t('downloads.noTasks') }}</h3>
            <p class="text-gray-500 dark:text-gray-400">{{ $t('downloads.noTasksDescription') }}</p>
          </div>

          <div v-else class="space-y-4">
            <DownloadCard
              v-for="task in filteredTasks"
              :key="task.id"
              :task="task"
              @toggle="toggleDownload"
              @cancel="cancelDownload"
            />
          </div>
        </div>
      </div>

      <!-- Settings Modal -->
      <div v-if="showSettings" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4">
          <div class="p-6 border-b border-gray-200 dark:border-gray-700">
            <div class="flex items-center justify-between">
              <h3 class="text-lg font-semibold text-gray-900 dark:text-white">{{ $t('downloads.settings') }}</h3>
              <button
                @click="showSettings = false"
                class="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <XMarkIcon class="w-5 h-5 text-gray-500" />
              </button>
            </div>
          </div>
          <div class="p-6 space-y-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {{ $t('downloads.maxConcurrent') }}
              </label>
              <input
                v-model.number="maxConcurrentDownloads"
                type="number"
                min="1"
                max="10"
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {{ $t('downloads.defaultThreads') }}
              </label>
              <input
                v-model.number="defaultThreads"
                type="number"
                min="1"
                max="16"
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
          <div class="p-6 border-t border-gray-200 dark:border-gray-700 flex justify-end space-x-3">
            <button
              @click="showSettings = false"
              class="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              {{ $t('common.cancel') }}
            </button>
            <button
              @click="saveSettings"
              class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              {{ $t('common.save') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { storeToRefs } from 'pinia'
import { useDownloadsStore } from '@/stores/downloads'
import DownloadCard from '@/components/DownloadCard.vue'
import type { DownloadTask, DownloadStatus } from '@/types'
import {
  DocumentArrowDownIcon,
  CheckCircleIcon,
  ClockIcon,
  CpuChipIcon,
  PlayIcon,
  PauseIcon,
  TrashIcon,
  MagnifyingGlassIcon,
  CogIcon,
  XMarkIcon,
  InboxIcon
} from '@heroicons/vue/24/outline'

const { t } = useI18n()
const downloadsStore = useDownloadsStore()
const { downloadTasks } = storeToRefs(downloadsStore)

const searchQuery = ref('')
const statusFilter = ref<'all' | DownloadStatus>('all')
const showSettings = ref(false)
const maxConcurrentDownloads = ref(3)
const defaultThreads = ref(8)

let refreshInterval: number | null = null

const filteredTasks = computed(() => {
  let tasks = downloadTasks.value

  // Filter by search query
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    tasks = tasks.filter(task => 
      task.modelName.toLowerCase().includes(query) ||
      task.filename.toLowerCase().includes(query)
    )
  }

  // Filter by status
  if (statusFilter.value !== 'all') {
    tasks = tasks.filter(task => task.status === statusFilter.value)
  }

  return tasks.sort((a, b) => {
    // Sort by status priority: downloading > paused > pending > error > completed
    const statusPriority = {
      downloading: 0,
      paused: 1,
      pending: 2,
      error: 3,
      completed: 4
    }
    return statusPriority[a.status] - statusPriority[b.status]
  })
})

const activeTasksCount = computed(() => 
  downloadTasks.value.filter(task => task.status === 'downloading').length
)

const completedTasksCount = computed(() => 
  downloadTasks.value.filter(task => task.status === 'completed').length
)

const formatTotalSpeed = computed(() => {
  const totalSpeed = downloadTasks.value
    .filter(task => task.status === 'downloading')
    .reduce((sum, task) => sum + task.speed, 0)
  return formatFileSize(totalSpeed) + '/s'
})

const totalThreadUsage = computed(() => {
  const totalThreads = downloadTasks.value
    .filter(task => task.status === 'downloading')
    .reduce((sum, task) => sum + task.threads, 0)
  const maxThreads = downloadTasks.value.length * defaultThreads.value
  return Math.round((totalThreads / maxThreads) * 100)
})

const hasActiveDownloads = computed(() => activeTasksCount.value > 0)

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

const toggleDownload = (taskId: string) => {
  downloadsStore.toggleDownload(taskId)
}

const cancelDownload = (taskId: string) => {
  downloadsStore.cancelDownload(taskId)
}

const toggleAllDownloads = () => {
  if (hasActiveDownloads.value) {
    downloadsStore.pauseAllDownloads()
  } else {
    downloadsStore.resumeAllDownloads()
  }
}

const clearCompleted = () => {
  downloadsStore.clearCompletedDownloads()
}

const saveSettings = () => {
  // Save settings to localStorage or store
  localStorage.setItem('downloadSettings', JSON.stringify({
    maxConcurrent: maxConcurrentDownloads.value,
    defaultThreads: defaultThreads.value
  }))
  showSettings.value = false
}

const loadSettings = () => {
  const saved = localStorage.getItem('downloadSettings')
  if (saved) {
    try {
      const settings = JSON.parse(saved)
      maxConcurrentDownloads.value = settings.maxConcurrent || 3
      defaultThreads.value = settings.defaultThreads || 8
    } catch (e) {
      console.warn('Failed to load download settings:', e)
    }
  }
}

// Simulate real-time updates
const startRefreshInterval = () => {
  refreshInterval = window.setInterval(() => {
    downloadsStore.updateDownloadProgress()
  }, 1000) // Update every second
}

const stopRefreshInterval = () => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

onMounted(() => {
  loadSettings()
  startRefreshInterval()
})

onUnmounted(() => {
  stopRefreshInterval()
})
</script>