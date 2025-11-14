<template>
  <div 
    class="border rounded-lg p-4 cursor-pointer transition-all duration-200"
    :class="cardClasses"
    @click="handleClick"
  >
    <div class="flex items-center justify-between mb-3">
      <div class="flex items-center space-x-3">
        <div 
          class="w-10 h-10 rounded-lg flex items-center justify-center"
          :class="iconClasses"
        >
          <component 
            :is="deviceIcon" 
            class="w-5 h-5 text-white" 
          />
        </div>
        <div>
          <h3 class="font-semibold text-gray-900 dark:text-white">
            {{ device.name }}
          </h3>
          <p class="text-sm text-gray-600 dark:text-gray-300">
            {{ getDeviceTypeLabel(device.type) }}
          </p>
        </div>
      </div>
      
      <div class="flex items-center space-x-2">
        <div 
          v-if="isPrimary"
          class="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-xs rounded-full font-medium"
        >
          Primary
        </div>
        <div 
          v-else-if="isSelected"
          class="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 text-xs rounded-full font-medium"
        >
          Selected
        </div>
        
        <div 
          class="w-3 h-3 rounded-full"
          :class="statusClass"
        />
      </div>
    </div>

    <!-- Device Metrics -->
    <div class="grid grid-cols-3 gap-4 text-sm">
      <div v-if="device.memory">
        <p class="text-gray-500 dark:text-gray-400 text-xs">{{ $t('hardware.memory') }}</p>
        <p class="font-medium text-gray-900 dark:text-white">{{ device.memory }} GB</p>
      </div>
      
      <div v-if="device.utilization !== undefined">
        <p class="text-gray-500 dark:text-gray-400 text-xs">{{ $t('hardware.utilization') }}</p>
        <p class="font-medium text-gray-900 dark:text-white">{{ device.utilization }}%</p>
      </div>
      
      <div v-if="device.temperature">
        <p class="text-gray-500 dark:text-gray-400 text-xs">{{ $t('hardware.temperature') }}</p>
        <p class="font-medium text-gray-900 dark:text-white">{{ device.temperature }}°C</p>
      </div>
    </div>

    <!-- Progress Bar for Utilization -->
    <div v-if="device.utilization !== undefined" class="mt-3">
      <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div 
          class="h-2 rounded-full transition-all duration-300"
          :class="utilizationBarClass"
          :style="{ width: `${Math.min(device.utilization, 100)}%` }"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { 
  CpuChipIcon, 
  RectangleStackIcon,
  MicrochipIcon,
  CommandLineIcon 
} from '@heroicons/vue/24/outline'
import type { HardwareDevice, HardwareType } from '@types/index'

const props = defineProps<{
  device: HardwareDevice
  isPrimary?: boolean
  isSelected?: boolean
}>()

const emit = defineEmits<{
  select: [device: HardwareDevice]
}>()

const { t } = useI18n()

const cardClasses = computed(() => {
  const base = 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
  
  if (props.isPrimary) {
    return 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
  } else if (props.isSelected) {
    return 'border-green-500 bg-green-50 dark:bg-green-900/20'
  }
  
  return base
})

const iconClasses = computed(() => {
  switch (props.device.type) {
    case HardwareType.CPU:
      return 'bg-blue-500'
    case HardwareType.INTEL_GPU:
    case HardwareType.NVIDIA_GPU:
    case HardwareType.AMD_GPU:
      return 'bg-purple-500'
    case HardwareType.INTEL_NPU:
      return 'bg-orange-500'
    default:
      return 'bg-gray-500'
  }
})

const deviceIcon = computed(() => {
  switch (props.device.type) {
    case HardwareType.CPU:
      return CpuChipIcon
    case HardwareType.INTEL_GPU:
    case HardwareType.NVIDIA_GPU:
    case HardwareType.AMD_GPU:
      return RectangleStackIcon
    case HardwareType.INTEL_NPU:
      return MicrochipIcon
    default:
      return CommandLineIcon
  }
})

const statusClass = computed(() => {
  if (!props.device.supported) return 'bg-red-500'
  if (props.device.utilization && props.device.utilization > 80) return 'bg-yellow-500'
  return 'bg-green-500'
})

const utilizationBarClass = computed(() => {
  if (props.device.utilization && props.device.utilization > 80) return 'bg-red-500'
  if (props.device.utilization && props.device.utilization > 60) return 'bg-yellow-500'
  return 'bg-green-500'
})

function handleClick() {
  emit('select', props.device)
}

function getDeviceTypeLabel(type: HardwareType): string {
  switch (type) {
    case HardwareType.CPU:
      return t('hardware.cpu')
    case HardwareType.INTEL_GPU:
      return `${t('hardware.intel')} ${t('hardware.gpu')}`
    case HardwareType.INTEL_NPU:
      return `${t('hardware.intel')} ${t('hardware.npu')}`
    case HardwareType.NVIDIA_GPU:
      return `${t('hardware.nvidia')} ${t('hardware.gpu')}`
    case HardwareType.AMD_GPU:
      return `${t('hardware.amd')} ${t('hardware.gpu')}`
    default:
      return 'Unknown'
  }
}
</script>