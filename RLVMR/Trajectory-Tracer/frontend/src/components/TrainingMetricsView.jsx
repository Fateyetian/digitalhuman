import { useStore } from '../store'

function SVGLineChart({ data, xKey, yKey, color = '#3b82f6', height = 120, label }) {
  if (!data || data.length === 0) return null

  const values = data.map(d => d[yKey])
  const xValues = data.map(d => d[xKey])

  const minY = Math.min(...values)
  const maxY = Math.max(...values)
  const rangeY = maxY - minY || 1

  const width = 400
  const padLeft = 40
  const padRight = 10
  const padTop = 10
  const padBottom = 20
  const chartW = width - padLeft - padRight
  const chartH = height - padTop - padBottom

  const toX = (i) => padLeft + (i / (data.length - 1)) * chartW
  const toY = (v) => padTop + chartH - ((v - minY) / rangeY) * chartH

  const points = data.map((d, i) => `${toX(i)},${toY(d[yKey])}`).join(' ')

  // Y-axis ticks
  const yTicks = [minY, (minY + maxY) / 2, maxY]

  // X-axis ticks (show ~5 ticks)
  const xTickIndices = data.length <= 5
    ? data.map((_, i) => i)
    : [0, Math.floor(data.length / 4), Math.floor(data.length / 2), Math.floor(3 * data.length / 4), data.length - 1]

  return (
    <div>
      {label && <div className="text-xs font-medium text-gray-600 mb-1">{label}</div>}
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        {/* Grid lines */}
        {yTicks.map((tick, i) => (
          <g key={i}>
            <line
              x1={padLeft} y1={toY(tick)}
              x2={width - padRight} y2={toY(tick)}
              stroke="#e5e7eb" strokeWidth="1"
            />
            <text x={padLeft - 4} y={toY(tick) + 4} textAnchor="end" fontSize="9" fill="#9ca3af">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}

        {/* X-axis ticks */}
        {xTickIndices.map((idx) => (
          <text key={idx} x={toX(idx)} y={height - 4} textAnchor="middle" fontSize="9" fill="#9ca3af">
            {xValues[idx]}
          </text>
        ))}

        {/* Line */}
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinejoin="round"
        />

        {/* Dots for small datasets */}
        {data.length <= 30 && data.map((d, i) => (
          <circle
            key={i}
            cx={toX(i)} cy={toY(d[yKey])}
            r="2.5"
            fill={color}
          >
            <title>{`Step ${d[xKey]}: ${d[yKey]?.toFixed(4)}`}</title>
          </circle>
        ))}
      </svg>
    </div>
  )
}

function MetricCard({ title, value, unit = '', color = 'blue', trend }) {
  const colorMap = {
    blue: 'text-blue-600 bg-blue-50',
    green: 'text-green-600 bg-green-50',
    red: 'text-red-600 bg-red-50',
    orange: 'text-orange-600 bg-orange-50',
    purple: 'text-purple-600 bg-purple-50',
  }
  return (
    <div className={`rounded-lg p-3 ${colorMap[color]}`}>
      <div className="text-xs font-medium opacity-70 mb-1">{title}</div>
      <div className="text-xl font-bold">
        {typeof value === 'number' ? value.toFixed(4) : value}
        {unit && <span className="text-sm ml-1 font-normal opacity-70">{unit}</span>}
      </div>
      {trend && (
        <div className={`text-xs mt-1 ${trend > 0 ? 'text-green-600' : trend < 0 ? 'text-red-500' : 'text-gray-500'}`}>
          {trend > 0 ? '↑' : trend < 0 ? '↓' : '→'} {Math.abs(trend).toFixed(4)}
        </div>
      )}
    </div>
  )
}

export default function TrainingMetricsView() {
  const { trainingMetrics, metricsLoading, fetchTrainingMetrics } = useStore()

  if (metricsLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-gray-400">加载训练指标中...</div>
      </div>
    )
  }

  if (!trainingMetrics || !trainingMetrics.steps || trainingMetrics.steps.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-gray-400">
          <div className="text-4xl mb-3">📊</div>
          <p className="mb-3">{trainingMetrics?.message || '暂无训练指标数据'}</p>
          <button
            onClick={fetchTrainingMetrics}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            重新加载
          </button>
        </div>
      </div>
    )
  }

  const { steps, total_steps } = trainingMetrics
  const latest = steps[steps.length - 1] || {}
  const first = steps[0] || {}

  // Compute trends (last 10 vs first 10)
  const n = Math.min(10, Math.floor(steps.length / 2))
  const earlyReward = steps.slice(0, n).reduce((s, d) => s + d.reward_mean, 0) / n
  const lateReward = steps.slice(-n).reduce((s, d) => s + d.reward_mean, 0) / n
  const rewardTrend = lateReward - earlyReward

  const earlyValid = steps.slice(0, n).reduce((s, d) => s + d.valid_action_ratio, 0) / n
  const lateValid = steps.slice(-n).reduce((s, d) => s + d.valid_action_ratio, 0) / n
  const validTrend = lateValid - earlyValid

  return (
    <div className="flex-1 flex flex-col overflow-y-auto bg-gray-50">
      <div className="p-6 max-w-4xl mx-auto w-full space-y-6">
        {/* 标题 */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-gray-900">训练过程分析</h2>
            <p className="text-sm text-gray-500">共 {total_steps} 个训练步骤</p>
          </div>
          <button
            onClick={fetchTrainingMetrics}
            className="px-3 py-1.5 text-sm bg-white border border-gray-300 rounded hover:bg-gray-50"
          >
            刷新
          </button>
        </div>

        {/* 关键指标卡片 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            title="最终 Reward"
            value={latest.reward_mean}
            color={latest.reward_mean > 0 ? 'green' : 'red'}
            trend={rewardTrend}
          />
          <MetricCard
            title="成功率"
            value={(latest.success_rate * 100).toFixed(1)}
            unit="%"
            color={latest.success_rate > 0.1 ? 'green' : 'red'}
          />
          <MetricCard
            title="有效动作比"
            value={latest.valid_action_ratio}
            color={latest.valid_action_ratio > 0.5 ? 'green' : lateValid < 0.1 ? 'red' : 'orange'}
            trend={validTrend}
          />
          <MetricCard
            title="响应长度"
            value={latest.response_length_mean?.toFixed(0)}
            unit="tokens"
            color={latest.response_length_mean > 1200 ? 'red' : 'blue'}
          />
        </div>

        {/* 问题诊断 */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="font-semibold text-gray-800 mb-3">🔍 问题诊断</h3>
          <div className="space-y-2">
            {rewardTrend < -0.01 && (
              <div className="flex items-start gap-2 text-sm text-red-700 bg-red-50 p-2 rounded">
                <span>⬇️</span>
                <span>Reward 呈下降趋势 ({earlyReward.toFixed(4)} → {lateReward.toFixed(4)})，训练可能发生了退化</span>
              </div>
            )}
            {lateValid < 0.1 && earlyValid > 0.3 && (
              <div className="flex items-start gap-2 text-sm text-red-700 bg-red-50 p-2 rounded">
                <span>💥</span>
                <span>valid_action_ratio 从 {earlyValid.toFixed(2)} 崩溃至 {lateValid.toFixed(2)}，模型输出可能进入重复循环</span>
              </div>
            )}
            {latest.response_length_mean > 1200 && (
              <div className="flex items-start gap-2 text-sm text-orange-700 bg-orange-50 p-2 rounded">
                <span>📏</span>
                <span>响应长度 {latest.response_length_mean?.toFixed(0)} tokens 接近上限，可能存在无限重复</span>
              </div>
            )}
            {latest.success_rate === 0 && (
              <div className="flex items-start gap-2 text-sm text-red-700 bg-red-50 p-2 rounded">
                <span>❌</span>
                <span>成功率为 0%，模型从未完成购买（click[Buy Now]），WebShop 永远无正 reward</span>
              </div>
            )}
            {lateValid > 0.7 && lateReward > earlyReward && (
              <div className="flex items-start gap-2 text-sm text-green-700 bg-green-50 p-2 rounded">
                <span>✅</span>
                <span>训练稳定，有效动作比良好，reward 有改善趋势</span>
              </div>
            )}
          </div>
        </div>

        {/* 图表 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <SVGLineChart
              data={steps}
              xKey="step"
              yKey="reward_mean"
              color="#ef4444"
              height={130}
              label="Reward Mean (每步)"
            />
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <SVGLineChart
              data={steps}
              xKey="step"
              yKey="valid_action_ratio"
              color="#3b82f6"
              height={130}
              label="Valid Action Ratio"
            />
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <SVGLineChart
              data={steps}
              xKey="step"
              yKey="response_length_mean"
              color="#f59e0b"
              height={130}
              label="Response Length (tokens)"
            />
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <SVGLineChart
              data={steps}
              xKey="step"
              yKey="success_rate"
              color="#10b981"
              height={130}
              label="Success Rate"
            />
          </div>

          {steps[0]?.entropy_loss !== undefined && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <SVGLineChart
                data={steps}
                xKey="step"
                yKey="entropy_loss"
                color="#8b5cf6"
                height={130}
                label="Entropy Loss"
              />
            </div>
          )}
        </div>

        {/* 原始数据表格 */}
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
            <h3 className="font-semibold text-gray-700 text-sm">原始数据 (最后20步)</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Step</th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600">Reward</th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600">Valid%</th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600">Resp Len</th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600">Success%</th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600">Entropy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {steps.slice(-20).map((s, i) => (
                  <tr key={i} className={`hover:bg-gray-50 ${
                    s.valid_action_ratio < 0.1 ? 'bg-red-50' : ''
                  }`}>
                    <td className="px-3 py-1.5 font-mono text-gray-700">{s.step}</td>
                    <td className={`px-3 py-1.5 text-right font-mono ${
                      s.reward_mean > 0 ? 'text-green-600' : 'text-red-500'
                    }`}>
                      {s.reward_mean?.toFixed(4)}
                    </td>
                    <td className={`px-3 py-1.5 text-right font-mono ${
                      s.valid_action_ratio < 0.1 ? 'text-red-600 font-bold' : 'text-gray-700'
                    }`}>
                      {(s.valid_action_ratio * 100).toFixed(1)}%
                    </td>
                    <td className={`px-3 py-1.5 text-right font-mono ${
                      s.response_length_mean > 1200 ? 'text-orange-600 font-bold' : 'text-gray-700'
                    }`}>
                      {s.response_length_mean?.toFixed(0)}
                    </td>
                    <td className={`px-3 py-1.5 text-right font-mono ${
                      s.success_rate > 0 ? 'text-green-600' : 'text-gray-400'
                    }`}>
                      {(s.success_rate * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-gray-500">
                      {s.entropy_loss?.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
