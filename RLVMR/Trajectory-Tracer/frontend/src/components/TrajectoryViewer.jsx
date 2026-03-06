import { useStore } from '../store'
import MessageBubble from './MessageBubble'

export default function TrajectoryViewer() {
  const { currentTrajectory } = useStore()

  if (!currentTrajectory) return null

  const getStatusIcon = (status) => {
    return status === 'success' ? '✅' : status === 'failed' ? '❌' : '❓'
  }

  const isRetrosynthesis = currentTrajectory.task_type === 'retrosynthesis'
  const isWebShop = currentTrajectory.task_type === 'webshop'
  const metadata = currentTrajectory.metadata || {}

  return (
    <div className="flex-1 flex flex-col bg-white overflow-hidden">
      {/* 任务信息卡片 */}
      <div className={`border-b border-gray-200 p-6 ${
        isWebShop
          ? 'bg-gradient-to-r from-amber-50 to-orange-50'
          : 'bg-gradient-to-r from-blue-50 to-indigo-50'
      }`}>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-2xl">{getStatusIcon(currentTrajectory.status)}</span>
              <h2 className="text-xl font-bold text-gray-900">
                {currentTrajectory.task || '无任务描述'}
              </h2>
            </div>

            {/* WebShop 特有信息 */}
            {isWebShop && (
              <div className="flex items-center gap-2 mt-2">
                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800 border border-amber-200">
                  🛒 WebShop ReBel
                </span>
                <span className="text-xs text-gray-500">ID: {metadata.item_id}</span>
                {metadata.annotation_success_rate !== undefined && (
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                    metadata.annotation_success_rate >= 0.9
                      ? 'bg-green-100 text-green-800'
                      : metadata.annotation_success_rate >= 0.5
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-red-100 text-red-800'
                  }`}>
                    Annotation: {(metadata.annotation_success_rate * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            )}

            {/* 逆合成轨迹特有信息 */}
            {isRetrosynthesis && metadata.target_molecule && (
              <div className="bg-purple-50 border-l-4 border-purple-400 p-3 rounded mt-3">
                <div className="text-xs font-semibold text-purple-800 mb-1">🧪 目标分子 (SMILES)</div>
                <div className="text-xs font-mono text-purple-900 break-all">
                  {metadata.target_molecule}
                </div>
              </div>
            )}

            {currentTrajectory.environment && !isRetrosynthesis && !isWebShop && (
              <div className="text-sm text-gray-600 bg-white/50 rounded p-3 mt-3">
                <div className="font-medium mb-1">🌍 环境描述:</div>
                <div className="whitespace-pre-wrap">{currentTrajectory.environment}</div>
              </div>
            )}
          </div>

          <div className="flex gap-4 ml-4 flex-wrap justify-end">
            <div className="text-center bg-white rounded-lg px-4 py-2 shadow-sm">
              <div className="text-2xl font-bold text-primary">{currentTrajectory.steps}</div>
              <div className="text-xs text-gray-500 mt-1">Steps</div>
            </div>

            {isWebShop && (
              <>
                <div className="text-center bg-white rounded-lg px-4 py-2 shadow-sm">
                  <div className="text-2xl font-bold text-amber-600">
                    {currentTrajectory.messages.filter(m => m.metadata?.belief_json).length}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Beliefs</div>
                </div>
                <div className="text-center bg-white rounded-lg px-4 py-2 shadow-sm">
                  <div className="text-2xl font-bold text-purple-600">
                    {currentTrajectory.messages.filter(m => m.thought).length}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Reasoning</div>
                </div>
              </>
            )}

            {!isWebShop && (
              <div className="text-center bg-white rounded-lg px-4 py-2 shadow-sm">
                <div className="text-2xl font-bold text-purple-600">
                  {currentTrajectory.messages.filter(m => m.thought).length}
                </div>
                <div className="text-xs text-gray-500 mt-1">Thoughts</div>
              </div>
            )}

            {/* 逆合成轨迹特有统计 */}
            {isRetrosynthesis && (
              <>
                <div className="text-center bg-white rounded-lg px-4 py-2 shadow-sm">
                  <div className="text-2xl font-bold text-green-600">
                    {metadata.valid_steps || 0}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Valid</div>
                </div>
                <div className="text-center bg-white rounded-lg px-4 py-2 shadow-sm">
                  <div className="text-2xl font-bold text-orange-600">
                    {(metadata.total_reward || 0).toFixed(1)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Reward</div>
                </div>
                {currentTrajectory.status === 'success' && (
                  <div className="text-center bg-white rounded-lg px-4 py-2 shadow-sm">
                    <div className="text-2xl font-bold text-blue-600">
                      {metadata.pathway_length}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Path</div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* 对话区域 */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-5xl mx-auto space-y-4">
          {currentTrajectory.messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
              index={index}
              isRetrosynthesis={isRetrosynthesis}
              isWebShop={isWebShop}
            />
          ))}
        </div>
      </div>

      {/* 成功轨迹的反应路径展示 */}
      {isRetrosynthesis && currentTrajectory.status === 'success' && (
        <div className="border-t border-gray-200 bg-gradient-to-r from-green-50 to-emerald-50 p-6">
          <div className="max-w-5xl mx-auto">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-2xl">🧪</span>
              <h3 className="text-lg font-bold text-gray-900">合成反应路径</h3>
              <span className="ml-auto text-sm text-gray-600">
                {metadata.pathway_validity ? '✅ 有效' : '⚠️ 无效'}
              </span>
            </div>

            <div className="bg-white rounded-lg p-4 border border-green-200">
              <div className="space-y-2">
                {metadata.final_pathway.map((reaction, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                      <span className="text-sm font-bold text-green-700">{idx + 1}</span>
                    </div>
                    <div className="flex-1 font-mono text-sm text-gray-700 break-all">
                      {typeof reaction === 'string' ? reaction : JSON.stringify(reaction)}
                    </div>
                  </div>
                ))}
              </div>

              {metadata.ground_truth_length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="text-xs text-gray-600">
                    <span className="font-semibold">路径长度:</span> {metadata.pathway_length} / {metadata.ground_truth_length}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
