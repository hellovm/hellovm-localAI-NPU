<template>
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow p-6">
    <!-- Header -->
    <div class="flex items-start justify-between mb-4">
      <div class="flex-1">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-1">
          {{ model.name }}
        </h3>
        <p class="text-sm text-gray-600 dark:text-gray-300 line-clamp-2">
          {{ model.description }}
        </p>
      </div>
      
      <div class="flex items-center space-x-2 ml-4">
        <div 
          v-if="model.downloaded"
          class="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 text-xs rounded-full font-medium"
        >
          {{ $t('models.downloaded') }}
        </div>
        <div 
          v-else
          class="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-xs rounded-full font-medium"
        >
          {{ $t('models.available') }}
        </div>
      </div>
    </div>

    <!-- Model Specs -->
    <div class="grid grid-cols-2 gap-4 mb-4 text-sm">
      <div>
        <p class="text-gray-500 dark:text-gray-400 text-xs">{{ $t('models.size') }}</p>
        <p class="font-medium text-gray-900 dark:text-white">{{ model.size }} GB</p>
      </div>
      
      <div>
        <p class="text-gray-500 dark:text-gray-400 text-xs">{{ $t('models.format') }}</p>
        <p class="font-medium text-gray-900 dark:text-white uppercase">{{ model.format }}</p>
      </div>
      
      <div v-if="model.quantization">
        <p class="text-gray-500 dark:text-gray-400 text-xs">{{ $t('models.quantization') }}</p>
        <p class="font-medium text-gray-900 dark:text-white">{{ model.quantization }}</p>
      </div>
      
      <div>
        <p class="text-gray-500 dark:text-gray-400 text-xs">{{ $t('models.contextLength') }}</p>
        <p class="font-medium text-gray-900 dark:text-white">{{ formatNumber(model.contextLength) }}</p>
      </div>
    </div>

    <!-- Tags -->
    <div class="mb-4">
      <p class="text-gray-500 dark:text-gray-400 text-xs mb-2">{{ $t('models.tags') }}</p>
      <div class="flex flex-wrap gap-1">
        <span 
          v-for="tag in model.tags.slice(0, 3)" 
          :key="tag"
          class="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-full"
        >
          {{ tag }}
        </span>
        <span 
          v-if="model.tags.length > 3"
          class="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-full"
        >
          +{{ model.tags.length - 3 }}
        </span>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex space-x-2">
      <button
        v-if="!model.downloaded"
        @click="$emit('download', model)"
        class="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
      >
        <ArrowDownTrayIcon class="w-4 h-4" />
        <span>{{ $t('models.download') }}</span>
      </button>
      
      <button
        v-else
        @click="$emit('select', model)"
        class="flex-1 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
      >
        <CheckIcon class="w-4 h-4" />
        <span>{{ $t('models.select') }}</span>
      </button>
      
      <button
        @click="showDetails = true"
        class="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
      >
        {{ $t('models.modelInfo') }}
      </button>
      
      <button
        v-if="model.downloaded"
        @click="$emit('delete', model)"
        class="px-4 py-2 border border-red-300 dark:border-red-600 text-red-700 dark:text-red-400 rounded-lg font-medium hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
      >
        <TrashIcon class="w-4 h-4" />
      </button>
    </div>
  </div>

  <!-- Model Details Modal -->
  <div 
    v-if="showDetails"
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    @click="showDetails = false"
  >
    <div 
      class="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto"
      @click.stop
    >
      <div class="p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
            {{ $t('models.modelDetails') }}
          </h2>
          <button
            @click="showDetails = false"
            class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <XMarkIcon class="w-6 h-6" />
          </button>
        </div>
        
        <div class="space-y-4">
          <div>
            <h3 class="font-medium text-gray-900 dark:text-white mb-2">{{ model.name }}</h3>
            <p class="text-gray-600 dark:text-gray-300">{{ model.description }}</p>
          </div>
          
          <div class="grid grid-cols-2 gap-4">
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">{{ $t('models.size') }}</p>
              <p class="font-medium">{{ model.size }} GB</p>
            </div>
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">{{ $t('models.format') }}</p>
              <p class="font-medium uppercase">{{ model.format }}</p>
            </div>
            <div v-if="model.quantization">
              <p class="text-sm text-gray-500 dark:text-gray-400">{{ $t('models.quantization') }}</p>
              <p class="font-medium">{{ model.quantization }}</p>
            </div>
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">{{ $t('models.contextLength') }}</p>
              <p class="font-medium">{{ formatNumber(model.contextLength) }}</p>
            </div>
          </div>
          
          <div>
            <p class="text-sm text-gray-500 dark:text-gray-400 mb-2">{{ $t('models.tags') }}</p>
            <div class="flex flex-wrap gap-1">
              <span 
                v-for="tag in model.tags" 
                :key="tag"
                class="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-full"
              >
                {{ tag }}
              </span>
            </div>
          </div>
          
          <div v-if="model.downloadUrl" class="pt-4 border-t border-gray-200 dark:border-gray-700">
            <p class="text-sm text-gray-500 dark:text-gray-400 mb-2">Download URL</p>
            <p class="text-sm font-mono bg-gray-100 dark:bg-gray-700 p-2 rounded break-all">
              {{ model.downloadUrl }}
            </p>
          </div>
        </div>
        
        <div class="flex justify-end space-x-2 mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button
            @click="showDetails = false"
            class="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            {{ $t('app.cancel') }}
          </button>
          <button
            v-if="!model.downloaded"
            @click="handleDownload"
            class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            {{ $t('models.download') }}
          </button>
          <button
            v-else
            @click="handleSelect"
            class="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
          >
            {{ $t('models.select') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { 
  ArrowDownTrayIcon, 
  CheckIcon, 
  TrashIcon,
  XMarkIcon 
} from '@heroicons/vue/24/outline'
import type { ModelInfo } from '@types/index'

const props = defineProps<{
  model: ModelInfo
}>()

const emit = defineEmits<{
  download: [model: ModelInfo]
  select: [model: ModelInfo]
  delete: [model: ModelInfo]
}>()

const showDetails = ref(false)

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toString()
}

function handleDownload() {
  showDetails.value = false
  emit('download', props.model)
}

function handleSelect() {
  showDetails.value = false
  emit('select', props.model)
}
</script>