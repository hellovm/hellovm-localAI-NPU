// Hardware acceleration types
export interface HardwareDevice {
  id: string
  name: string
  type: HardwareType
  memory?: number // in GB
  utilization?: number // percentage
  temperature?: number // celsius
  supported: boolean
  selected: boolean
}

export enum HardwareType {
  CPU = 'cpu',
  INTEL_GPU = 'intel_gpu',
  INTEL_NPU = 'intel_npu',
  NVIDIA_GPU = 'nvidia_gpu',
  AMD_GPU