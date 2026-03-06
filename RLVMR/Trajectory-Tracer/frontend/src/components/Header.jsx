import { useStore } from '../store'

export default function Header() {
  const { statistics, reloadData, toggleAutoRefresh, autoRefresh, lastReloadResult, loading } = useStore()

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Trajectory Viewer</h1>
          <p className="text-sm text-gray-500 mt-1">ReBel / RetroEvo Trajectories Visualization</p>
        </div>

        <div className="flex items-center gap-4">
          {/* 刷新控制区 */}
          <div className="flex items-center gap-2">
            <button
              onClick={reloadData}
              disabled={loading}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                loading
                  ? 'border-gray-200 text-gray-400 cursor-not-allowed'
                  : 'border-blue-300 text-blue-700 hover:bg-blue-50 active:bg-blue-100'
              }`}
              title="重新加载数据源"
            >
              <svg className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Reload
            </button>

            <button
              onClick={toggleAutoRefresh}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                autoRefresh
                  ? 'border-green-400 bg-green-50 text-green-700'
                  : 'border-gray-300 text-gray-600 hover:bg-gray-50'
              }`}
              title={autoRefresh ? '关闭自动刷新 (15s)' : '开启自动刷新 (15s)'}
            >
              <span className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
              Auto
            </button>

            {lastReloadResult && lastReloadResult.delta > 0 && (
              <span className="text-xs text-green-600 font-medium">
                +{lastReloadResult.delta} new
              </span>
            )}
          </div>

          {/* 统计卡片 */}
          {statistics && (
            <div className="flex gap-4">
              <div className="text-center bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg px-4 py-2 shadow-sm border border-blue-200">
                <div className="text-2xl font-bold text-primary">{statistics.total}</div>
                <div className="text-xs text-gray-600 mt-1">Total</div>
              </div>
              <div className="text-center bg-gradient-to-br from-green-50 to-green-100 rounded-lg px-4 py-2 shadow-sm border border-green-200">
                <div className="text-2xl font-bold text-success">
                  {statistics.by_status?.success || 0}
                </div>
                <div className="text-xs text-gray-600 mt-1">Success</div>
              </div>
              {statistics.by_source?.webshop_rebel > 0 && (
                <div className="text-center bg-gradient-to-br from-amber-50 to-amber-100 rounded-lg px-4 py-2 shadow-sm border border-amber-200">
                  <div className="text-2xl font-bold text-amber-600">
                    {statistics.by_source.webshop_rebel}
                  </div>
                  <div className="text-xs text-gray-600 mt-1">WebShop</div>
                </div>
              )}
              <div className="text-center bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg px-4 py-2 shadow-sm border border-purple-200">
                <div className="text-2xl font-bold text-purple-600">{statistics.avg_steps}</div>
                <div className="text-xs text-gray-600 mt-1">Avg Steps</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
