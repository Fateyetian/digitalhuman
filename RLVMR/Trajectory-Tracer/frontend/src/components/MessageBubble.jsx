export default function MessageBubble({ message, index, isRetrosynthesis = false, isWebShop = false }) {
  const isHuman = message.role === 'human'
  const isAgent = message.role === 'agent'
  const metadata = message.metadata || {}

  // Agent 消息（右侧）
  if (isAgent) {
    const isValidAction = metadata.is_valid_action
    const reward = metadata.reward
    const cumulativeReward = metadata.cumulative_reward
    const toolName = metadata.tool_name
    const beliefJson = metadata.belief_json
    const beliefSummary = metadata.belief_summary

    return (
      <div className="flex justify-end">
        <div className="max-w-2xl">
          <div className="flex items-center justify-end gap-2 mb-2">
            <span className="text-xs text-gray-400">Step #{metadata.step !== undefined ? metadata.step : Math.floor(index / 2) + 1}</span>
            {isRetrosynthesis && toolName && (
              <span className={`text-xs px-2 py-0.5 rounded ${
                isValidAction ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                {toolName}
              </span>
            )}
            <span className="text-sm font-medium text-primary">
              {isWebShop ? '🛒 Agent' : '🤖 Agent'}
            </span>
          </div>

          <div className={`rounded-2xl rounded-tr-sm px-5 py-4 shadow-md ${
            isWebShop
              ? 'bg-gradient-to-br from-amber-500 to-orange-600 text-white'
              : isRetrosynthesis
                ? isValidAction
                  ? 'bg-gradient-to-br from-green-500 to-green-600 text-white'
                  : 'bg-gradient-to-br from-blue-500 to-blue-600 text-white'
                : 'bg-gradient-to-br from-blue-500 to-blue-600 text-white'
          }`}>

            {/* WebShop ReBel: Belief State */}
            {isWebShop && beliefSummary && (
              <div className="mb-3 pb-3 border-b border-white/20">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold opacity-80">🧠 BELIEF STATE</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-white/15 rounded px-2 py-1">
                    <span className="opacity-70">Status: </span>
                    <span className="font-medium">{beliefSummary.search_status}</span>
                  </div>
                  <div className="bg-white/15 rounded px-2 py-1">
                    <span className="opacity-70">Match: </span>
                    <span className={`font-medium ${
                      beliefSummary.product_match === 'exact' ? 'text-green-200' :
                      beliefSummary.product_match === 'partial' ? 'text-yellow-200' :
                      'text-white/80'
                    }`}>{beliefSummary.product_match}</span>
                  </div>
                  <div className="bg-white/15 rounded px-2 py-1">
                    <span className="opacity-70">Price: </span>
                    <span className="font-medium">{beliefSummary.price_constraint}</span>
                  </div>
                  <div className="bg-white/15 rounded px-2 py-1">
                    <span className="opacity-70">Explored: </span>
                    <span className="font-medium">
                      Q:{beliefSummary.queries_count} P:{beliefSummary.products_viewed_count} O:{beliefSummary.options_selected_count}
                    </span>
                  </div>
                </div>
                {beliefSummary.subgoal && (
                  <div className="mt-2 text-xs bg-white/10 rounded px-2 py-1">
                    <span className="opacity-70">Subgoal: </span>
                    <span className="italic">{beliefSummary.subgoal.length > 100 ? beliefSummary.subgoal.slice(0, 100) + '...' : beliefSummary.subgoal}</span>
                  </div>
                )}
              </div>
            )}

            {/* WebShop ReBel: Full Belief JSON (collapsible) */}
            {isWebShop && beliefJson && (
              <details className="mb-3 pb-3 border-b border-white/20">
                <summary className="text-xs font-semibold opacity-80 cursor-pointer hover:opacity-100">
                  📋 Full Belief JSON
                </summary>
                <pre className="mt-2 text-xs bg-white/10 rounded px-3 py-2 overflow-x-auto whitespace-pre-wrap font-mono">
                  {JSON.stringify(beliefJson, null, 2)}
                </pre>
              </details>
            )}

            {/* Thought/Reasoning */}
            {message.thought && (
              <div className="mb-3 pb-3 border-b border-white/20">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold opacity-80">
                    {isWebShop ? '💭 REASONING' : '💭 THOUGHT'}
                  </span>
                </div>
                <div className="text-sm italic leading-relaxed">
                  {message.thought}
                </div>
              </div>
            )}

            {/* Action */}
            {message.action && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold opacity-80">⚡ ACTION</span>
                </div>
                <div className={`font-mono text-sm font-medium bg-white/10 px-3 py-2 rounded ${
                  isWebShop && message.action.toLowerCase().includes('buy now')
                    ? 'ring-2 ring-green-300'
                    : ''
                }`}>
                  {isWebShop ? formatWebShopAction(message.action) : message.action}
                </div>
              </div>
            )}

            {!message.thought && !message.action && !(isWebShop && beliefSummary) && (
              <div className="text-sm whitespace-pre-wrap">
                {message.content}
              </div>
            )}

            {/* 逆合成轨迹特有信息：奖励 */}
            {isRetrosynthesis && reward !== undefined && (
              <div className="mt-3 pt-3 border-t border-white/20 flex gap-4 text-xs">
                <span className={`${reward >= 0 ? 'text-green-200' : 'text-red-200'}`}>
                  奖励: {reward > 0 ? '+' : ''}{reward.toFixed(2)}
                </span>
                {cumulativeReward !== undefined && (
                  <span className="text-white/70">
                    累计: {cumulativeReward.toFixed(2)}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  // Human 消息（左侧）
  if (isHuman) {
    // 检查是否是初始指令
    const isInitial = message.content.includes('Interact with a household')

    // 检查是否包含任务描述
    const hasTask = message.content.includes('Your task is to:')
    const hasAvailableActions = message.content.includes('AVAILABLE ACTIONS:')

    // WebShop 特有的解析
    const isWebShopObs = isWebShop && (
      message.content.includes('Observation:') || message.content.includes('[SEP]')
    )
    const hasBeliefState = message.content.includes('Current Belief State:')

    // 逆合成轨迹的工具响应
    const isToolResponse = metadata.type === 'tool_response'
    const respondingToolName = metadata.tool_name

    return (
      <div className="flex justify-start">
        <div className="max-w-2xl">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-gray-600">
              {isWebShop ? '🌐 WebShop' : isRetrosynthesis ? '🔬 Tool Response' : '🌍 Environment'}
            </span>
            {isRetrosynthesis && respondingToolName && (
              <span className="text-xs text-gray-500">({respondingToolName})</span>
            )}
          </div>

          <div className={`rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm ${
            isWebShopObs
              ? 'bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-200'
              : isInitial
                ? 'bg-gradient-to-br from-gray-100 to-gray-200 border-2 border-gray-300'
                : isRetrosynthesis && isToolResponse
                  ? 'bg-gradient-to-br from-amber-50 to-orange-50 border border-orange-200'
                  : 'bg-gray-100'
          }`}>
            {isWebShopObs ? (
              // WebShop 观察格式化
              <WebShopObservation content={message.content} hasBeliefState={hasBeliefState} />
            ) : hasTask ? (
              // 解析并格式化包含任务的消息
              <div className="space-y-3">
                {message.content.split('Your task is to:')[0] && (
                  <div className="text-sm text-gray-700 leading-relaxed">
                    {message.content.split('Your task is to:')[0].trim()}
                  </div>
                )}

                {hasTask && (
                  <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 rounded">
                    <div className="text-xs font-semibold text-yellow-800 mb-1">🎯 TASK</div>
                    <div className="text-sm font-medium text-yellow-900">
                      {message.content.split('Your task is to:')[1].split('\n')[0].trim()}
                    </div>
                  </div>
                )}

                {hasAvailableActions && (
                  <div className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded">
                    <div className="text-xs font-semibold text-blue-800 mb-1">🎮 AVAILABLE ACTIONS</div>
                    <div className="text-xs text-blue-900 font-mono">
                      {message.content.split('AVAILABLE ACTIONS:')[1].trim()}
                    </div>
                  </div>
                )}
              </div>
            ) : isRetrosynthesis && isToolResponse ? (
              // 逆合成轨迹的工具响应 - 特殊格式化
              <div className="text-sm text-gray-700 leading-relaxed">
                {message.content.includes('Reactions for molecule') ? (
                  <div className="space-y-2">
                    <div className="font-medium text-orange-800 mb-2">🧪 可用反应:</div>
                    <div className="font-mono text-xs bg-white/50 p-2 rounded whitespace-pre-wrap break-all">
                      {message.content}
                    </div>
                  </div>
                ) : message.content.includes('Selected reaction') ? (
                  <div className="space-y-2">
                    <div className="font-medium text-green-800 mb-2">✅ 反应已选择:</div>
                    <div className="font-mono text-xs bg-white/50 p-2 rounded whitespace-pre-wrap">
                      {message.content}
                    </div>
                  </div>
                ) : message.content.includes('Unknown tool') || message.content.includes('Invalid') ? (
                  <div className="space-y-2">
                    <div className="font-medium text-red-800 mb-2">❌ 错误:</div>
                    <div className="font-mono text-xs text-red-700 bg-red-50 p-2 rounded">
                      {message.content}
                    </div>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap break-all">{message.content}</div>
                )}
              </div>
            ) : (
              // 普通观察消息
              <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                {message.content}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  return null
}


// ============================================================
// WebShop 专用子组件
// ============================================================

function WebShopObservation({ content, hasBeliefState }) {
  // 解析结构化观察
  const sections = {}

  // 提取 Task
  const taskMatch = content.match(/Task:\s*(.+?)(?:\n|$)/)
  if (taskMatch) sections.task = taskMatch[1].trim()

  // 提取 Observation（支持多行）
  const obsMatch = content.match(/Observation:\s*\n?([\s\S]*?)(?=\n(?:Current Belief State:|Available Actions:|$))/)
  if (obsMatch) sections.observation = obsMatch[1].trim()

  // 提取 Current Belief State
  if (hasBeliefState) {
    const beliefMatch = content.match(/Current Belief State:\s*\n?([\s\S]*?)(?=\n(?:Available Actions:|$))/)
    if (beliefMatch) sections.beliefState = beliefMatch[1].trim()
  }

  // 提取 Available Actions
  const actionsMatch = content.match(/Available Actions:\s*(.+?)$/)
  if (actionsMatch) sections.availableActions = actionsMatch[1].trim()

  // 如果解析失败，显示原始内容
  if (!sections.observation && !sections.task) {
    return (
      <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
        {content}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {sections.task && (
        <div className="bg-amber-100/50 border-l-4 border-amber-400 p-2 rounded">
          <div className="text-xs font-semibold text-amber-800 mb-1">🛒 Shopping Task</div>
          <div className="text-sm font-medium text-amber-900">{sections.task}</div>
        </div>
      )}

      {sections.observation && (
        <div className="text-sm text-gray-700">
          <div className="text-xs font-semibold text-gray-500 mb-1">👁 Observation</div>
          <div className="bg-white/60 rounded p-2 text-xs leading-relaxed whitespace-pre-wrap max-h-48 overflow-y-auto">
            {formatWebShopObsText(sections.observation)}
          </div>
        </div>
      )}

      {sections.beliefState && (
        <details className="text-sm">
          <summary className="text-xs font-semibold text-purple-700 cursor-pointer hover:text-purple-900">
            🧠 Previous Belief State
          </summary>
          <pre className="mt-1 bg-purple-50 rounded p-2 text-xs font-mono whitespace-pre-wrap overflow-x-auto max-h-32 overflow-y-auto">
            {sections.beliefState}
          </pre>
        </details>
      )}

      {sections.availableActions && sections.availableActions !== 'N/A' && (
        <details className="text-sm">
          <summary className="text-xs font-semibold text-blue-700 cursor-pointer hover:text-blue-900">
            🎮 Available Actions
          </summary>
          <div className="mt-1 text-xs text-blue-900 font-mono bg-blue-50 rounded p-2">
            {sections.availableActions}
          </div>
        </details>
      )}
    </div>
  )
}


function formatWebShopObsText(text) {
  // 格式化 WebShop [SEP] 分隔的文本
  if (text.includes('[SEP]')) {
    return text.split('[SEP]').map((part, i) => (
      <span key={i}>
        {i > 0 && <span className="text-amber-400 mx-1">|</span>}
        {part.trim()}
      </span>
    ))
  }
  return text
}


function formatWebShopAction(action) {
  // 高亮 search[...] 和 click[...] 动作
  const searchMatch = action.match(/^(search)\[(.+)\]$/i)
  const clickMatch = action.match(/^(click)\[(.+)\]$/i)

  if (searchMatch) {
    return (
      <span>
        <span className="text-yellow-200">search</span>
        <span className="opacity-60">[</span>
        <span className="text-white font-semibold">{searchMatch[2]}</span>
        <span className="opacity-60">]</span>
      </span>
    )
  }

  if (clickMatch) {
    const value = clickMatch[2]
    const isBuy = value.toLowerCase() === 'buy now'
    return (
      <span>
        <span className="text-yellow-200">click</span>
        <span className="opacity-60">[</span>
        <span className={`font-semibold ${isBuy ? 'text-green-200' : 'text-white'}`}>{value}</span>
        <span className="opacity-60">]</span>
      </span>
    )
  }

  return action
}
