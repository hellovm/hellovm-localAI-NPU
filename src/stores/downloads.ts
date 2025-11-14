import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { DownloadTask, DownloadStatus } from '@types/index'

export const useDownloadsStore = defineStore('downloads', () => {
  const downloadTasks = ref<DownloadTask[]>([
    // Mock download tasks for demonstration
    {
      id: 'download-1',
      modelId: 'qwen-7b-chat',
      modelName: 'Qwen-7B-Chat',
      filename: 'qwen-7b-chat-q4_k_m.gguf',
      url: 'https://modelscope.cn/models/qwen/qwen-7b-chat/resolve/main/qwen-7b-chat-q4_k_m.gguf',
      totalSize: 4.2 * 1024 * 1024 * 1024, // 4.2 GB
      downloadedSize: 2.1 * 1024 * 1024 * 1024, // 2.1 GB (50%)
      progress: 50,
      status: DownloadStatus.DOWNLOADING,
      speed: 15.5 * 1024 * 1024, // 15.5 MB/s
      eta: 136, // ~2.3 minutes remaining
      threads: 8,
      maxConnections: 16,
      activeConnections: 12,
      resumable: true,
      resumeSupported: true,
      checksum: 'sha256:abc123...',
      error: undefined
    },
    {
      id: 'download-2',
      modelId: 'llama2-13b-chat',
      modelName: 'Llama2-13B-Chat',
      filename: 'llama-2-13b-chat-q5_k_m.gguf',
      url: 'https://modelscope.cn/models/meta-llama/llama-2-13b-chat/resolve/main/llama-2-13b-chat-q5_k_m.gguf',
      totalSize: 8.5 * 1024 * 1024 * 1024, // 8.5 GB
      downloadedSize: 6.8 * 1024 * 1024 * 1024, // 6.8 GB (80%)
      progress: 80,
      status: DownloadStatus.DOWNLOADING,
      speed: 22.3 * 1024 * 1024, // 22.3 MB/s
      eta: 82, // ~1.4 minutes remaining
      threads: 12,
      maxConnections: 20,
      activeConnections: 16,
      resumable: true,
      resumeSupported: true,
      checksum: 'sha256:def456...',
      error: undefined
    },
    {
      id: 'download-3',
      modelId: 'deepseek-coder-6.7b',
      modelName: 'DeepSeek-Coder-6.7B',
      filename: 'deepseek-coder-6.7b-q4_k_m.gguf',
      url: 'https://modelscope.cn/models/deepseek-ai/deepseek-coder-6.7b/resolve/main/deepseek-coder-6.7b-q4_k_m.gguf',
      totalSize: 3.8 * 1024 * 1024 * 1024, // 3.8 GB
      downloadedSize: 3.8 * 1024 * 1024 * 1024, // 3.8 GB (100%)
      progress: 100,
      status: DownloadStatus.COMPLETED,
      speed: 0,
      eta: undefined,
      threads: 6,
      maxConnections: 12,
      activeConnections: 0,
      resumable: true,
      resumeSupported: true,
      checksum: 'sha256:ghi789...',
      error: undefined
    },
    {
      id: 'download-4',
      modelId: 'yi-34b-chat',
      modelName: 'Yi-34B-Chat',
      filename: 'yi-34b-chat-q3_k_m.gguf',
      url: 'https://modelscope.cn/models/01-ai/yi-34b-chat/resolve/main/yi-34b-chat-q3_k_m.gguf',
      totalSize: 15.2 * 1024 * 1024 * 1024, // 15.2 GB
      downloadedSize: 4.6 * 1024 * 1024 * 1024, // 4.6 GB (30%)
      progress: 30,
      status: DownloadStatus.PAUSED,
      speed: 0,
      eta: undefined,
      threads: 10,
      maxConnections: 18,
      activeConnections: 0,
      resumable: true,
      resumeSupported: true,
      checksum: 'sha256:jkl012...',
      error: undefined
    },
    {
      id: 'download-5',
      modelId: 'baichuan2-13b-chat',
      modelName: 'Baichuan2-13B-Chat',
      filename: 'baichuan2-13b-chat-q4_k_m.gguf',
      url: 'https://modelscope.cn/models/baichuan-inc/baichuan2-13b-chat/resolve/main/baichuan2-13b-chat-q4_k_m.gguf',
      totalSize: 8.2 * 1024 * 1024 * 1024, // 8.2 GB
      downloadedSize: 1.2 * 1024 * 1024 * 1024, // 1.2 GB (15%)
      progress: 15,
      status: DownloadStatus.ERROR,
      speed: 0,
      eta: undefined,
      threads: 8,
      maxConnections: 16,
      activeConnections: 0,
      resumable: true,
      resumeSupported: true,
      checksum: 'sha256:mno345...',
      error: 'Network connection timeout'
    }
  ])
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

  function toggleDownload(taskId: string) {
    const task = downloadTasks.value.find(t => t.id === taskId)
    if (!task) return

    switch (task.status) {
      case DownloadStatus.DOWNLOADING:
        pauseDownload(taskId)
        break
      case DownloadStatus.PAUSED:
      case DownloadStatus.PENDING:
        resumeDownload(taskId)
        break
      case DownloadStatus.FAILED:
        retryDownload(taskId)
        break
    }
  }

  function pauseAllDownloads() {
    downloadTasks.value
      .filter(task => task.status === DownloadStatus.DOWNLOADING)
      .forEach(task => pauseDownload(task.id))
  }

  function resumeAllDownloads() {
    downloadTasks.value
      .filter(task => task.status === DownloadStatus.PAUSED || task.status === DownloadStatus.PENDING)
      .forEach(task => resumeDownload(task.id))
  }

  // Real-time progress update for UI
  function updateDownloadProgress() {
    downloadTasks.value.forEach(task => {
      if (task.status === DownloadStatus.DOWNLOADING) {
        // Simulate slight speed variations for realism
        const speedVariation = (Math.random() - 0.5) * 0.2 // ±10% variation
        const newSpeed = task.speed * (1 + speedVariation)
        
        // Update active connections based on thread usage
        const activeConnections = Math.ceil((Math.random() * 0.4 + 0.6) * task.maxConnections)
        
        updateDownloadTask(task.id, {
          speed: Math.max(0, newSpeed),
          activeConnections: Math.min(activeConnections, task.maxConnections)
        })
      }
    })
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
    toggleDownload,
    pauseAllDownloads,
    resumeAllDownloads,
    markAsCompleted,
    markAsFailed,
    markAsVerifying,
    clearCompletedDownloads,
    clearFailedDownloads,
    startDownload,
    updateDownloadProgress
  }
})