import { useEffect } from 'react'
import { useStore } from '../store'

export default function TrajectoryList() {
  const {
    trajectories,
    currentTrajectory,
    fetchTrajectoryDetail,
    loading,
    pagination,
    nextPage,
    prevPage,
    filters
  } = useStore()

  useEffect(() => {
    // 当筛选器变化时自动刷新
    useStore.getState().fetchTrajectories()
  }, [filters])

  const getStatusBadge = (status) => {
    const badges = {
      success: { color: 'bg-green-100 text-green-800', icon: '✓', text: '成功' },
      failed: { color: 'bg-red-100 text-red-800', icon: '✗', text: '失败' },
      unknown: { color: 'bg-gray-100 text-gray-800', icon: '?', text: '未知' }
    }
    const badge = badges[status] || badges.unknown
    return (
      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${badge.color}`}>
        <span className="mr-1">{badge.icon}</span>
        {badge.text}
      </span>
    )
  }

  const getTaskTypeBadge = (taskType) => {
    const configs = {
      webshop: { color: 'bg-amber-100 text-amber-800', label: 'WebShop' },
      put: { color: 'bg-blue-100 text-blue-800', label: 'Put' },
      clean: { color: 'bg-purple-100 text-purple-800', label: 'Clean' },
      cool: { color: 'bg-cyan-100 text-cyan-800', label: 'Cool' },
      heat: { color: 'bg-orange-100 text-orange-800', label: 'Heat' },
      find: { color: 'bg-pink-100 text-pink-800', label: 'Find' },
      examine: { color: 'bg-green-100 text-green-800', label: 'Examine' },
      use: { color: 'bg-yellow-100 text-yellow-800', label: 'Use' },
      other: { color: 'bg-gray-100 text-gray-800', label: 'Other' },
      unknown: { color: 'bg-gray-100 text-gray-800', label: 'Unknown' }
    }
    const config = configs[taskType] || configs.unknown
    return (
      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${config.color}`}>
        {config.label}
      </span>
    )
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 列表 */}
      <div className="flex-1 overflow-y-auto">
        {loading && trajectories.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-gray-400">加载中...</div>
          </div>
        ) : trajectories.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-gray-400 text-sm">没有找到轨迹</div>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {trajectories.map((traj) => (
              <div
                key={traj.id}
                onClick={() => fetchTrajectoryDetail(traj.id)}
                className={`p-3 cursor-pointer hover:bg-gray-50 transition-colors ${
                  currentTrajectory?.id === traj.id ? 'bg-blue-50 border-l-4 border-primary' : ''
                }`}
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  {getStatusBadge(traj.status)}
                  {getTaskTypeBadge(traj.task_type)}
                </div>

                <div className="text-sm text-gray-900 line-clamp-2 mb-2">
                  {traj.task || '无任务描述'}
                </div>

                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>{traj.id}</span>
                  <span>{traj.steps} 步</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 分页 */}
      <div className="border-t border-gray-200 p-3 flex items-center justify-between bg-gray-50">
        <button
          onClick={prevPage}
          disabled={pagination.skip === 0}
          className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          上一页
        </button>
        <span className="text-xs text-gray-600">
          {pagination.skip + 1} - {pagination.skip + trajectories.length}
        </span>
        <button
          onClick={() => {
            nextPage()
            setTimeout(() => useStore.getState().fetchTrajectories(), 100)
          }}
          disabled={trajectories.length < pagination.limit}
          className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          下一页
        </button>
      </div>
    </div>
  )
}
