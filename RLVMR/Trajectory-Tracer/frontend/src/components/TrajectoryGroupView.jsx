import { useState } from 'react'
import { useStore } from '../store'

// ─── 奖励色块 ──────────────────────────────────────────────────────────────
function RewardBadge({ reward }) {
  const r = parseFloat(reward) || 0
  let cls = 'bg-gray-100 text-gray-600'
  if (r >= 4.0)       cls = 'bg-green-500 text-white'
  else if (r >= 0.5)  cls = 'bg-green-200 text-green-800'
  else if (r > 0)     cls = 'bg-yellow-100 text-yellow-700'
  else if (r < -0.1)  cls = 'bg-red-200 text-red-800'
  else                cls = 'bg-red-100 text-red-600'
  return (
    <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-bold ${cls}`}>
      {r >= 0 ? '+' : ''}{r.toFixed(3)}
    </span>
  )
}

// ─── 动作色块 ──────────────────────────────────────────────────────────────
function ActionChip({ action, reward }) {
  const a = action || ''
  let label = a, cls = 'bg-gray-100 text-gray-600'
  if (a.startsWith('search[')) {
    label = '🔍 ' + a.slice(7, -1).slice(0, 24)
    cls = 'bg-blue-100 text-blue-700'
  } else if (/buy now/i.test(a)) {
    label = '💰 Buy Now'
    cls = 'bg-green-300 text-green-900 font-bold ring-1 ring-green-500'
  } else if (a.startsWith('click[')) {
    const inner = a.slice(6, -1)
    label = '👆 ' + inner.slice(0, 18)
    cls = 'bg-amber-100 text-amber-700'
  }
  const r = parseFloat(reward) || 0
  return (
    <span className="inline-flex items-center gap-0.5 flex-shrink-0">
      <span className={`text-xs px-1.5 py-0.5 rounded ${cls}`} title={a}>{label}</span>
      {r !== 0 && (
        <span className={`text-xs ${r > 0 ? 'text-green-600' : 'text-red-400'}`}>
          {r > 0 ? '+' : ''}{r.toFixed(2)}
        </span>
      )}
    </span>
  )
}

// ─── 单条轨迹卡片（含内联动作序列） ────────────────────────────────────────
function TrajCard({ traj, rank, maxAbs, onDetail }) {
  const [expanded, setExpanded] = useState(false)
  const isSuccess = traj.status === 'success'
  const actions = traj.actions || []
  const pct = Math.min(Math.abs(traj.total_reward) / (maxAbs || 1), 1) * 100

  return (
    <div className={`rounded-lg border overflow-hidden ${
      isSuccess ? 'border-green-300 bg-green-50/50' : 'border-gray-200 bg-white'
    }`}>
      {/* 头部行 */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-inherit">
        <span className="text-xs font-mono text-gray-400 w-5">#{rank}</span>

        <span className={`text-xs px-1.5 py-0.5 rounded font-medium flex-shrink-0 ${
          isSuccess ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-600'
        }`}>
          {isSuccess ? '✓ 成功' : '✗ 失败'}
        </span>

        <span className="text-xs text-gray-400">{traj.steps} 步</span>

        {/* reward 进度条 */}
        <div className="flex-1 h-2 bg-gray-100 rounded overflow-hidden">
          <div
            className={`h-full rounded ${traj.total_reward >= 0 ? 'bg-green-400' : 'bg-red-400'}`}
            style={{ width: `${pct}%` }}
          />
        </div>

        <RewardBadge reward={traj.total_reward} />

        {traj.exp_tag && (
          <span className="text-xs text-gray-400 truncate max-w-[120px]" title={traj.exp_tag}>
            {traj.exp_tag.slice(0, 20)}…
          </span>
        )}

        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            onClick={() => setExpanded(v => !v)}
            className="text-xs px-2 py-0.5 rounded bg-gray-100 hover:bg-gray-200 text-gray-600"
          >
            {expanded ? '收起 ▲' : '展开 ▼'}
          </button>
          <button
            onClick={() => onDetail(traj.id)}
            className="text-xs px-2 py-0.5 rounded bg-blue-100 hover:bg-blue-200 text-blue-700"
          >
            全文 →
          </button>
        </div>
      </div>

      {/* 动作序列（默认展示，始终可见） */}
      <div className="px-3 py-2 flex items-center gap-1 flex-wrap">
        {actions.length > 0 ? (
          actions.map((a, i) => (
            <ActionChip key={i} action={a} reward={traj.step_rewards?.[i]} />
          ))
        ) : (
          <span className="text-xs text-gray-400 italic">无动作数据（点击"全文"查看）</span>
        )}
      </div>

      {/* 展开：显示obs摘要（需要从group消息里取，暂显示索引信息） */}
      {expanded && (
        <div className="px-3 pb-3 pt-1 bg-gray-50 border-t border-gray-200 text-xs text-gray-500 space-y-1">
          <div>Trajectory ID: <span className="font-mono">{traj.id}</span></div>
          <div>traj_index: {traj.traj_index} | source: {traj.source}</div>
        </div>
      )}
    </div>
  )
}

// ─── 右侧对比面板 ──────────────────────────────────────────────────────────
function TrajectoryCompare({ group, onDetail }) {
  const [sortBy, setSortBy] = useState('reward')   // reward | index
  const [filterStatus, setFilterStatus] = useState('all') // all | success | failed

  if (!group) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400">
        <div className="text-center">
          <div className="text-4xl mb-3">👈</div>
          <p>选择左侧一个任务组查看同任务对比</p>
        </div>
      </div>
    )
  }

  const { trajectories: rawTrajs = [], task, reward_mean, reward_std, count, success_count } = group
  const maxAbs = Math.max(...rawTrajs.map(t => Math.abs(t.total_reward)), 0.01)

  // 过滤 + 排序
  let trajs = rawTrajs.filter(t => filterStatus === 'all' || t.status === filterStatus)
  if (sortBy === 'reward') trajs = [...trajs].sort((a, b) => b.total_reward - a.total_reward)
  else trajs = [...trajs].sort((a, b) => a.traj_index - b.traj_index)

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 组标题 */}
      <div className="bg-amber-50 border-b border-amber-200 px-4 py-3">
        <p className="font-semibold text-gray-900 text-sm line-clamp-2 mb-2">{task}</p>
        <div className="flex items-center gap-4 text-xs text-gray-600 flex-wrap">
          <span>{count} 条轨迹</span>
          <span className={success_count > 0 ? 'text-green-600 font-medium' : 'text-red-500'}>
            {success_count > 0 ? `✓ ${success_count} 成功` : '✗ 全部失败'}
          </span>
          <span>avg <RewardBadge reward={reward_mean} /></span>
          <span className="text-gray-400">std {reward_std?.toFixed(3)}</span>

          {/* 排序 */}
          <div className="ml-auto flex items-center gap-2">
            <span className="text-gray-400">排序:</span>
            {['reward', 'index'].map(s => (
              <button key={s}
                onClick={() => setSortBy(s)}
                className={`px-2 py-0.5 rounded text-xs ${sortBy === s ? 'bg-blue-500 text-white' : 'bg-white border text-gray-600 hover:bg-gray-50'}`}
              >
                {s === 'reward' ? 'Reward↓' : 'Epoch↑'}
              </button>
            ))}
            {/* 过滤 */}
            <span className="text-gray-400 ml-2">显示:</span>
            {['all', 'success', 'failed'].map(f => (
              <button key={f}
                onClick={() => setFilterStatus(f)}
                className={`px-2 py-0.5 rounded text-xs ${filterStatus === f ? 'bg-blue-500 text-white' : 'bg-white border text-gray-600 hover:bg-gray-50'}`}
              >
                {f === 'all' ? '全部' : f === 'success' ? '成功' : '失败'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 轨迹列表 */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {trajs.length === 0
          ? <p className="text-gray-400 text-sm text-center mt-8">无匹配轨迹</p>
          : trajs.map((traj, i) => (
              <TrajCard
                key={traj.id}
                traj={traj}
                rank={i + 1}
                maxAbs={maxAbs}
                onDetail={onDetail}
              />
            ))
        }
      </div>
    </div>
  )
}

// ─── 左侧任务组列表 ────────────────────────────────────────────────────────
function GroupCard({ group, isSelected, onSelect }) {
  const { reward_mean, success_count, count, task, group_id } = group
  return (
    <div
      onClick={() => onSelect(group)}
      className={`p-2.5 rounded-lg border cursor-pointer transition-all hover:shadow-sm ${
        isSelected
          ? 'border-blue-400 bg-blue-50 shadow-sm'
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className={`text-xs px-1.5 py-0.5 rounded font-medium flex-shrink-0 ${
          success_count > 0 ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-600'
        }`}>
          {success_count}/{count}
        </span>
        <RewardBadge reward={reward_mean} />
      </div>
      <p className="text-xs text-gray-700 line-clamp-2">{task || group_id}</p>
    </div>
  )
}

// ─── 主视图 ────────────────────────────────────────────────────────────────
export default function TrajectoryGroupView() {
  const {
    trajectoryGroups,
    groupsLoading,
    selectedGroup,
    setSelectedGroup,
    fetchTrajectoryGroups,
    fetchTrajectoryDetail,
    setCurrentView,
  } = useStore()

  const [sourceFilter, setSourceFilter] = useState('webshop_rl')
  const [searchText, setSearchText] = useState('')

  const handleRefresh = () => fetchTrajectoryGroups(sourceFilter || undefined)

  const handleDetail = (trajId) => {
    fetchTrajectoryDetail(trajId)
    setCurrentView('list')
  }

  if (groupsLoading) {
    return <div className="flex-1 flex items-center justify-center text-gray-400">加载分组数据中...</div>
  }

  if (!trajectoryGroups) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-gray-400">
          <p className="mb-3">暂无分组数据</p>
          <button onClick={handleRefresh} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
            加载分组
          </button>
        </div>
      </div>
    )
  }

  const { groups: allGroups = [], total_groups } = trajectoryGroups

  // 文本搜索过滤
  const groups = searchText
    ? allGroups.filter(g => g.task?.toLowerCase().includes(searchText.toLowerCase()))
    : allGroups

  const successGroups = groups.filter(g => g.success_count > 0).length

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* 左侧任务组列表 */}
      <div className="w-72 flex flex-col bg-white border-r border-gray-200 flex-shrink-0">
        {/* 顶部工具栏 */}
        <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">
              {groups.length} / {total_groups} 组
              <span className="ml-1 text-green-600 text-xs">({successGroups} 有成功)</span>
            </span>
            <button onClick={handleRefresh} className="text-xs text-blue-500 hover:text-blue-700">刷新</button>
          </div>

          {/* 数据源过滤 */}
          <div className="flex gap-1">
            {['webshop_rl', 'webshop_rebel', ''].map(s => (
              <button key={s}
                onClick={() => { setSourceFilter(s); fetchTrajectoryGroups(s || undefined) }}
                className={`flex-1 text-xs py-0.5 rounded border ${
                  sourceFilter === s ? 'bg-blue-500 text-white border-blue-500' : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'
                }`}
              >
                {s === 'webshop_rl' ? 'RL训练' : s === 'webshop_rebel' ? 'SFT' : '全部'}
              </button>
            ))}
          </div>

          {/* 任务搜索 */}
          <input
            type="text"
            placeholder="搜索任务..."
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            className="w-full text-xs px-2 py-1 border border-gray-200 rounded focus:outline-none focus:border-blue-300"
          />
        </div>

        {/* 分组列表 */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
          {groups.map(group => (
            <GroupCard
              key={group.group_id}
              group={group}
              isSelected={selectedGroup?.group_id === group.group_id}
              onSelect={setSelectedGroup}
            />
          ))}
        </div>
      </div>

      {/* 右侧对比面板 */}
      <TrajectoryCompare group={selectedGroup} onDetail={handleDetail} />
    </div>
  )
}
