<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <NavigationBar />
    
    <div class="container mx-auto px-4 py-8">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          {{ $t('models.title') }}
        </h1>
        <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <p class="text-gray-600 dark:text-gray-300">
            {{ $t('models.search') }}
          </p>
          
          <!-- Search and Filter Controls -->
          <div class="flex flex-col md:flex-row gap-4">
            <div class="relative">
              <input
                v-model="searchQuery"
                type="text"
                :placeholder="$t('models.search')"
                class="w-full md:w-64 px-4 py-2 pl-10 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <MagnifyingGlassIcon class="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
            </div>
            
            <select
              v-model="selectedFormat"
              class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">{{ $t('models.format') }}</option>
              <option value="gguf">GGUF</option>
              <option value="ggml">GGML</option>
              <option value="onnx">ONNX</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Model Categories -->
      <div class="mb-8">
        <div class="flex space-x-4 border-b border-gray-200 dark:border-gray-700">
          <button
            v-for="category in categories"
            :key="category.key"
            @click="activeCategory = category.key"
            class="px-4 py-2 font-medium text-sm border-b-2 transition-colors"
            :class="activeCategory === category.key 
              ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'"
          >
            {{ category.label }}
            <span class="ml-1 text-xs">({{ category.count }})</span>
          </button>
        </div>
      </div>

      <!-- Models Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <ModelCard
          v-for="model in filteredModels"
          :key="model.id"
          :model="model"
          @download="handleDownload"
          @select="selectModel"
          @delete="handleDelete"
        />
      </div>

      <!-- No Results -->
      <div v-if="filteredModels.length === 0" class="text-center py-12">
        <CpuChipIcon class="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">
          {{ $t('models.noModels') }}
        </h3>
        <p class="text-gray-600 dark:text-gray-300">
          Try adjusting your search or filter criteria
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useModelsStore } from '@stores/models'
import { useDownloadsStore } from '@stores/downloads'
import { useI18n } from 'vue-i18n'
import { MagnifyingGlassIcon, CpuChipIcon } from '@heroicons/vue/24/outline'
import NavigationBar from '@components/NavigationBar.vue'
import ModelCard from '@components/ModelCard.vue'
import type { ModelInfo } from '@types/index'

const { t } = useI18n()
const modelsStore = useModelsStore()
const downloadsStore = useDownloadsStore()

const searchQuery = ref('')
const selectedFormat = ref('')
const activeCategory = ref('all')

const categories = computed(() => [
  {
    key: 'all',
    label: 'All Models',
    count: modelsStore.models.length
  },
  {
    key: 'downloaded',
    label: 'Downloaded',
    count: modelsStore.downloadedModels.length
  },
  {
    key: 'available',
    label: 'Available',
    count: modelsStore.availableModels.length
  }
])

const filteredModels = computed(() => {
  let models = modelsStore.models

  // Filter by category
  if (activeCategory.value === 'downloaded') {
    models = modelsStore.downloadedModels
  } else if (activeCategory.value === 'available') {
    models = modelsStore.availableModels
  }

  // Filter by search query
  if (searchQuery.value.trim()) {
    const query = searchQuery.value.toLowerCase()
    models = models.filter(model =>
      model.name.toLowerCase().includes(query) ||
      model.description.toLowerCase().includes(query) ||
      model.tags.some(tag => tag.toLowerCase().includes(query))
    )
  }

  // Filter by format
  if (selectedFormat.value) {
    models = models.filter(model => model.format === selectedFormat.value)
  }

  return models
})

function handleDownload(model: ModelInfo) {
  const taskId = downloadsStore.addDownloadTask({
    modelId: model.id,
    modelName: model.name,
    totalSize: model.size * 1024 * 1024 * 1024, // Convert GB to bytes
    downloadedSize: 0,
    speed: 0,
    status: 'pending',
    progress: 0,
    resumeSupported: true,
    threads: 4
  })
  
  downloadsStore.startDownload(taskId)
}

function selectModel(model: ModelInfo) {
  modelsStore.selectModel(model.id)
}

function handleDelete(model: ModelInfo) {
  if (confirm(`Are you sure you want to delete ${model.name}?`)) {
    modelsStore.removeModel(model.id)
  }
}

onMounted(async () => {
  await modelsStore.loadModels()
})
</script>