<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <NavigationBar />
    
    <div class="container mx-auto px-4 py-8">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          {{ $t('hardware.title') }}
        </h1>
        <p class="text-gray-600 dark:text-gray-300">
          {{ $t('hardware.detect') }}
        </p>
      </div>

      <!-- Hardware Detection -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
            {{ $t('hardware.deviceInfo') }}
          </h2>
          <button
            @click="detectHardware"
            :disabled="hardwareStore.isDetecting"
            class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            <ArrowPathIcon 
              :class="{ 'animate-spin': hardwareStore.isDetecting }" 
              class="w-4 h-4" 
            />
            <span>{{ $t('hardware.detect') }}</span>
          </button>
        </div>

        <!-- Acceleration Mode Selection -->
        <div class="mb-6">
          <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-3">
            {{ $t('hardware.accelerationMode') }}
          </h3>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div 
              v-for="mode in accelerationModes" 
              :key="mode.value"
              @click="selectAccelerationMode(mode.value)"
              class="border rounded-lg p-4 cursor-pointer transition-colors"
              :class="hardwareStore.selectedConfig.mode === mode.value 
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'"
            >
              <div class="flex items-center space-x-3">
                <div 
                  class="w-4 h-4 rounded-full border-2"
                  :class="hardwareStore.selectedConfig.mode === mode.value 
                    ? 'border-blue-500 bg-blue-500' 
                    : 'border-gray-300 dark:border-gray-600'"
                />
                <div>
                  <h4 class="font-medium text-gray-900 dark:text-white">
                    {{ mode.label }}
                  </h4>
                  <p class="text-sm text-gray-600 dark:text-gray-300">
                    {{ mode.description }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Device List -->
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-medium text-gray-900 dark:text-white">
              {{ $t('hardware.primaryDevice') }}
            </h3>
          </div>
          
          <DeviceCard 
            :device="hardwareStore.primaryDevice"
            :is-primary="true"
            @select="selectPrimaryDevice"
          />

          <div v-if="hardwareStore.selectedConfig.mode !== 'single'" class="mt-6">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-lg font-medium text-gray-900 dark:text-white">
                {{ $t('hardware.secondaryDevices') }}
              </h3>
            </div>
            
            <div class="space-y-3">
              <DeviceCard 
                v-for="device in hardwareStore.availableDevices.filter(d => d.id !== hardwareStore.primaryDevice.id)"
                :key="device.id"
                :device="device"
                :is-primary="false"
                :is-selected="isSecondaryDevice(device)"
                @select="toggleSecondaryDevice"
              />
            </div>
          </div>
        </div>
      </div>

      <!-- Performance Metrics -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-6">
          {{ $t('performance.title') }}
        </h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard 
            :title="$t('performance.tokensPerSecond')"
            value="45.2"
            unit="tokens/s"
            trend="up"
          />
          <MetricCard 
            :title="$t('performance.memoryUsage')"
            value="8.4"
            unit="GB"
            trend="stable"
          />
          <MetricCard 
            :title="$t('performance.cpuUsage')"
            value="35"
            unit="%"
            trend="down"
          />
          <MetricCard 
            :title="$t('performance.latency')"
            value="124"
            unit="ms"
            trend="up"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useHardwareStore } from '@stores/hardware'
import { useI18n } from 'vue-i18n'
import { ArrowPathIcon } from '@heroicons/vue/24/outline'
import NavigationBar from '@components/NavigationBar.vue'
import DeviceCard from '@components/DeviceCard.vue'
import MetricCard from '@components/MetricCard.vue'
import type { HardwareDevice } from '@types/index'

const { t } = useI18n()
const hardwareStore = useHardwareStore()

const accelerationModes = [
  {
    value: 'single',
    label: t('hardware.singleMode'),
    description: 'Use a single device for acceleration'
  },
  {
    value: 'multi',
    label: t('hardware.multiMode'),
    description: 'Use multiple devices in parallel'
  },
  {
    value: 'hybrid',
    label: t('hardware.hybridMode'),
    description: 'Combine different device types'
  }
]

onMounted(async () => {
  await hardwareStore.detectHardware()
})

async function detectHardware() {
  await hardwareStore.detectHardware()
}

function selectPrimaryDevice(device: HardwareDevice) {
  hardwareStore.selectPrimaryDevice(device)
}

function selectAccelerationMode(mode: 'single' | 'multi' | 'hybrid') {
  hardwareStore.setAccelerationMode(mode)
}

function isSecondaryDevice(device: HardwareDevice): boolean {
  return hardwareStore.secondaryDevices.some(d => d.id === device.id)
}

function toggleSecondaryDevice(device: HardwareDevice) {
  if (isSecondaryDevice(device)) {
    hardwareStore.removeSecondaryDevice(device.id)
  } else {
    hardwareStore.addSecondaryDevice(device)
  }
}
</script>