import { useStore } from '../store'

export default function FilterPanel() {
  const { filters, setFilter, clearFilters, fetchTrajectories } = useStore()

  const handleApply = () => {
    fetchTrajectories()
  }

  return (
    <div className="p-4 border-b border-gray-200">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-700">筛选器</h3>
        <button
          onClick={() => {
            clearFilters()
            fetchTrajectories()
          }}
          className="text-xs text-primary hover:text-blue-600"
        >
          清除
        </button>
      </div>

      <div className="space-y-3">
        {/* 状态筛选 */}
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">状态</label>
          <select
            value={filters.status || ''}
            onChange={(e) => setFilter('status', e.target.value || null)}
            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="">全部</option>
            <option value="success">成功</option>
            <option value="failed">失败</option>
            <option value="unknown">未知</option>
          </select>
        </div>

        {/* 任务类型筛选 */}
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">任务类型</label>
          <select
            value={filters.taskType || ''}
            onChange={(e) => setFilter('taskType', e.target.value || null)}
            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="">全部</option>
            <option value="webshop">🛒 WebShop (网购)</option>
            <option value="retrosynthesis">🧪 Retrosynthesis (逆合成)</option>
            <option value="put">Put (放置)</option>
            <option value="clean">Clean (清洁)</option>
            <option value="cool">Cool (冷却)</option>
            <option value="heat">Heat (加热)</option>
            <option value="find">Find (寻找)</option>
            <option value="examine">Examine (检查)</option>
            <option value="use">Use (使用)</option>
            <option value="other">Other (其他)</option>
          </select>
        </div>

        {/* 步数范围 */}
        <div className="flex gap-2">
          <div className="flex-1">
            <label className="block text-xs font-medium text-gray-600 mb-1">最小步数</label>
            <input
              type="number"
              min="0"
              value={filters.minSteps || ''}
              onChange={(e) => setFilter('minSteps', e.target.value ? parseInt(e.target.value) : null)}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary"
              placeholder="0"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs font-medium text-gray-600 mb-1">最大步数</label>
            <input
              type="number"
              min="0"
              value={filters.maxSteps || ''}
              onChange={(e) => setFilter('maxSteps', e.target.value ? parseInt(e.target.value) : null)}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary"
              placeholder="∞"
            />
          </div>
        </div>

        <button
          onClick={handleApply}
          className="w-full bg-primary text-white py-2 rounded hover:bg-blue-600 transition-colors text-sm font-medium"
        >
          应用筛选
        </button>
      </div>
    </div>
  )
}
