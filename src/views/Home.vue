<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
    <NavigationBar />
    
    <div class="container mx-auto px-4 py-8">
      <!-- Hero Section -->
      <div class="text-center mb-12">
        <h1 class="text-5xl font-bold text-gray-900 dark:text-white mb-4">
          {{ $t('app.title') }}
        </h1>
        <p class="text-xl text-gray-600 dark:text-gray-300 mb-8">
          {{ $t('app.subtitle') }}
        </p>
        
        <div class="flex justify-center space-x-4">
          <router-link 
            to="/chat" 
            class="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold transition-colors"
          >
            {{ $t('chat.title') }}
          </router-link>
          <router-link 
            to="/models" 
            class="bg-gray-600 hover:bg-gray-700 text-white px-8 py-3 rounded-lg font-semibold transition-colors"
          >
            {{ $t('models.title') }}
          </router-link>
        </div>
      </div>

      <!-- Features Grid -->
      <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
        <FeatureCard 
          icon="⚡"
          :title="$t('hardware.title')"
          :description="$t('hardware.detect')"
          link="/hardware"
        />
        <FeatureCard 
          icon="🤖"
          :title="$t('models.title')"
          :description="$t('models.search')"
          link="/models"
        />
        <FeatureCard 
          icon="💬"
          :title="$t('chat.title')"
          :description="$t('chat.noMessages')"
          link="/chat"
        />
        <FeatureCard 
          icon="⬇️"
          :title="$t('downloads.title')"
          :description="$t('downloads.active')"
          link="/downloads"
        />
        <FeatureCard 
          icon="🔌"
          :title="$t('plugins.title')"
          :description="$t('plugins.installed')"
          link="/plugins"
        />
        <FeatureCard 
          icon="⚙️"
          :title="$t('settings.title')"
          :description="$t('settings.general')"
          link="/settings"
        />
      </div>

      <!-- System Status -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          System Status
        </h2>
        
        <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatusCard 
            :title="$t('hardware.title')"
            :value="hardwareStore.devices.length + ' devices'"
            :status="hardwareStore.devices.length > 0 ? 'online' : 'offline'"
          />
          <StatusCard 
            :title="$t('models.title')"
            :value="modelsStore.downloadedModels.length + ' downloaded'"
            status="online"
          />
          <StatusCard 
            :title="$t('downloads.title')"
            :value="downloadsStore.activeDownloads.length + ' active'"
            :status="downloadsStore.activeDownloads.length > 0 ? 'active' : 'idle'"
          />
          <StatusCard 
            :title="Performance"
            value="Ready"
            status="online"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useHardwareStore } from '@stores/hardware'
import { useModelsStore } from '@stores/models'
import { useDownloadsStore } from '@stores/downloads'
import NavigationBar from '@components/NavigationBar.vue'
import FeatureCard from '@components/FeatureCard.vue'
import StatusCard from '@components/StatusCard.vue'

const hardwareStore = useHardwareStore()
const modelsStore = useModelsStore()
const downloadsStore = useDownloadsStore()

onMounted(async () => {
  // Initialize stores
  await hardwareStore.detectHardware()
  await modelsStore.loadModels()
})
</script>