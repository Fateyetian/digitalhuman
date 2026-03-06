import { create } from 'zustand'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'

export const useStore = create((set, get) => ({
  // 数据
  trajectories: [],
  currentTrajectory: null,
  statistics: null,

  // UI 状态
  loading: false,
  error: null,
  autoRefresh: false,
  autoRefreshInterval: null,
  lastReloadResult: null,

  // 筛选器
  filters: {
    status: null,
    taskType: null,
    minSteps: null,
    maxSteps: null,
  },

  // 分页
  pagination: {
    skip: 0,
    limit: 50,
  },

  // 视图模式: 'list' | 'groups' | 'metrics'
  currentView: 'list',

  // 轨迹分组数据
  trajectoryGroups: null,
  groupsLoading: false,
  selectedGroup: null,

  // 训练指标数据
  trainingMetrics: null,
  metricsLoading: false,

  // 获取轨迹列表
  fetchTrajectories: async () => {
    set({ loading: true, error: null })
    try {
      const { filters, pagination } = get()
      const params = new URLSearchParams({
        skip: pagination.skip,
        limit: pagination.limit,
        ...(filters.status && { status: filters.status }),
        ...(filters.taskType && { task_type: filters.taskType }),
        ...(filters.minSteps && { min_steps: filters.minSteps }),
        ...(filters.maxSteps && { max_steps: filters.maxSteps }),
      })

      const response = await fetch(`${API_BASE}/trajectories?${params}`)
      if (!response.ok) throw new Error('Failed to fetch trajectories')

      const data = await response.json()
      set({ trajectories: data, loading: false })
    } catch (error) {
      set({ error: error.message, loading: false })
    }
  },

  // 获取轨迹详情
  fetchTrajectoryDetail: async (id) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`${API_BASE}/trajectories/${id}`)
      if (!response.ok) throw new Error('Failed to fetch trajectory detail')

      const data = await response.json()
      set({ currentTrajectory: data, loading: false })
    } catch (error) {
      set({ error: error.message, loading: false })
    }
  },

  // 获取统计信息
  fetchStatistics: async () => {
    try {
      const response = await fetch(`${API_BASE}/statistics`)
      if (!response.ok) throw new Error('Failed to fetch statistics')

      const data = await response.json()
      set({ statistics: data })
    } catch (error) {
      console.error('Failed to fetch statistics:', error)
    }
  },

  // 重新加载所有数据源
  reloadData: async () => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`${API_BASE}/reload`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to reload data')

      const result = await response.json()
      set({ lastReloadResult: result })

      // 刷新列表和统计
      await get().fetchTrajectories()
      await get().fetchStatistics()
      set({ loading: false })
      return result
    } catch (error) {
      set({ error: error.message, loading: false })
      return null
    }
  },

  // 切换自动刷新
  toggleAutoRefresh: () => {
    const { autoRefresh, autoRefreshInterval } = get()
    if (autoRefresh) {
      // 关闭
      if (autoRefreshInterval) clearInterval(autoRefreshInterval)
      set({ autoRefresh: false, autoRefreshInterval: null })
    } else {
      // 开启：每 15 秒刷新
      const interval = setInterval(async () => {
        await get().reloadData()
      }, 15000)
      set({ autoRefresh: true, autoRefreshInterval: interval })
      // 立即刷新一次
      get().reloadData()
    }
  },

  // 设置筛选器
  setFilter: (key, value) => {
    set(state => ({
      filters: { ...state.filters, [key]: value },
      pagination: { ...state.pagination, skip: 0 } // 重置分页
    }))
  },

  // 清除筛选器
  clearFilters: () => {
    set({
      filters: {
        status: null,
        taskType: null,
        minSteps: null,
        maxSteps: null,
      },
      pagination: { skip: 0, limit: 50 }
    })
  },

  // 设置当前轨迹
  setCurrentTrajectory: (trajectory) => {
    set({ currentTrajectory: trajectory })
  },

  // 切换视图
  setCurrentView: (view) => {
    set({ currentView: view })
    if (view === 'groups') {
      get().fetchTrajectoryGroups()
    } else if (view === 'metrics') {
      get().fetchTrainingMetrics()
    }
  },

  // 获取轨迹分组
  fetchTrajectoryGroups: async (source = 'webshop_rl') => {
    set({ groupsLoading: true })
    try {
      const params = source ? `?source=${source}` : ''
      const response = await fetch(`${API_BASE}/trajectory-groups${params}`)
      if (!response.ok) throw new Error('Failed to fetch trajectory groups')
      const data = await response.json()
      set({ trajectoryGroups: data, groupsLoading: false })
    } catch (error) {
      console.error('Failed to fetch trajectory groups:', error)
      set({ groupsLoading: false })
    }
  },

  // 获取训练指标
  fetchTrainingMetrics: async () => {
    set({ metricsLoading: true })
    try {
      const response = await fetch(`${API_BASE}/training-metrics`)
      if (!response.ok) throw new Error('Failed to fetch training metrics')
      const data = await response.json()
      set({ trainingMetrics: data, metricsLoading: false })
    } catch (error) {
      console.error('Failed to fetch training metrics:', error)
      set({ metricsLoading: false })
    }
  },

  // 选中分组
  setSelectedGroup: (group) => {
    set({ selectedGroup: group })
  },

  // 分页
  nextPage: () => {
    set(state => ({
      pagination: {
        ...state.pagination,
        skip: state.pagination.skip + state.pagination.limit
      }
    }))
  },

  prevPage: () => {
    set(state => ({
      pagination: {
        ...state.pagination,
        skip: Math.max(0, state.pagination.skip - state.pagination.limit)
      }
    }))
  },
}))
