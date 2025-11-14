import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { DownloadTask, DownloadStatus } from '@types/index'

export const useDownloadsStore = defineStore('downloads', () => {
  const downloadTasks = ref<DownloadTask[]>([])
  const isProcessing = ref(false)

  const activeDownloads = computed(() => 
    downloadTasks.value.filter(task => 
      task.status === DownloadStatus.DOWNLOADING || 
      task.status === DownloadStatus.PENDING
    )
  )
  
  const completedDownloads = computed(() => 
    downloadTasks.value.filter(task => task.status === DownloadStatus.COMPLETED)
  )
  
  const failedDownloads = computed(() => 
    downloadTasks.value.filter(task => task.status === DownloadStatus.FAILED)
  )

  const totalDownloadSpeed = computed(() => 
    activeDownloads.value.reduce((sum, task) => sum + task.speed, 0)
  )

  function addDownloadTask(task: Omit<DownloadTask, 'id'>) {
    const id = `download-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newTask: DownloadTask = {
      ...task,
      id,
      status: DownloadStatus.PENDING
    }
    downloadTasks.value.push(newTask)
    return id
  }

  function updateDownloadTask(taskId: string, updates: Partial<DownloadTask>) {
    const task = downloadTasks.value.find(t => t.id === taskId)
    if (task) {
      Object.assign(task, updates)
      
      // Update progress based on downloaded size
      if (updates.downloadedSize !== undefined && task.totalSize > 0) {
        task.progress = Math.round((updates.downloadedSize / task.totalSize) * 100)
      }
      
      // Calculate ETA if speed is available
      if (updates.speed && task.totalSize > 0 && task.downloadedSize < task.totalSize) {
        const remainingBytes = task.totalSize - task.downloadedSize
        task.eta = Math.round(remainingBytes / updates.speed)
      }
    }
  }

  function pauseDownload(taskId: string) {
    updateDownloadTask(taskId, { status: DownloadStatus.PAUSED })
  }

  function resumeDownload(taskId: string) {
    const task = downloadTasks.value.find(t => t.id === taskId)
    if (task && task.resumeSupported) {
      updateDownloadTask(taskId, { status: DownloadStatus.DOWNLOADING })
    }
  }

  function cancelDownload(taskId: string) {
    const index = downloadTasks.value.findIndex(t => t.id === taskId)
    if (index >= 0) {
      downloadTasks.value.splice(index, 1)
    }
  }

  function retryDownload(taskId: string) {
    const task = downloadTasks.value.find(t => t.id === taskId)
    if (task) {
      updateDownloadTask(taskId, { 
        status: DownloadStatus.DOWNLOADING,
        error: undefined,
        downloadedSize: 0,
        progress: 0
      })
    }
  }

  function markAsCompleted(taskId: string) {
    updateDownloadTask(taskId, { 
      status: DownloadStatus.COMPLETED,
      progress: 100,
      speed: 0,
      eta: undefined
    })
  }

  function markAsFailed(taskId: string, error: string) {
    updateDownloadTask(taskId, { 
      status: DownloadStatus.FAILED,
      error,
      speed: 0,
      eta: undefined
    })
  }

  function markAsVerifying(taskId: string) {
    updateDownloadTask(taskId, { status: DownloadStatus.VERIFYING })
  }

  function clearCompletedDownloads() {
    downloadTasks.value = downloadTasks.value.filter(
      task => task.status !== DownloadStatus.COMPLETED
    )
  }

  function clearFailedDownloads() {
    downloadTasks.value = downloadTasks.value.filter(
      task => task.status !== DownloadStatus.FAILED
    )
  }

  // Simulate download progress (for demo purposes)
  function simulateDownloadProgress(taskId: string) {
    const task = downloadTasks.value.find(t => t.id === taskId)
    if (!task || task.status !== DownloadStatus.DOWNLOADING) return

    const speed = Math.random() * 50 * 1024 * 1024 // 0-50 MB/s
    const increment = speed * 0.1 // Update every 100ms
    
    if (task.downloadedSize + increment < task.totalSize) {
      updateDownloadTask(taskId, {
        downloadedSize: task.downloadedSize + increment,
        speed: speed
      })
      
      // Continue simulation
      setTimeout(() => simulateDownloadProgress(taskId), 100)
    } else {
      // Download completed
      markAsCompleted(taskId)
    }
  }

  function startDownload(taskId: string) {
    updateDownloadTask(taskId, { status: DownloadStatus.DOWNLOADING })
    simulateDownloadProgress(taskId)
  }

  return {
    downloadTasks,
    isProcessing,
    activeDownloads,
    completedDownloads,
    failedDownloads,
    totalDownloadSpeed,
    addDownloadTask,
    updateDownloadTask,
    pauseDownload,
    resumeDownload,
    cancelDownload,
    retryDownload,
    markAsCompleted,
    markAsFailed,
    markAsVerifying,
    clearCompletedDownloads,
    clearFailedDownloads,
    startDownload
  }
})