import { useState, useEffect, useMemo } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'

// ── Phase/Env badge configs ───────────────────────────────────────────────
const PHASE_STYLE = {
  rl:         'bg-blue-100 text-blue-800',
  sft:        'bg-green-100 text-green-800',
  annotation: 'bg-purple-100 text-purple-800',
  basic:      'bg-gray-100 text-gray-600',
  rebel:      'bg-amber-100 text-amber-800',
}
const ENV_STYLE = {
  alfworld:  'bg-teal-100 text-teal-800',
  webshop:   'bg-orange-100 text-orange-800',
  sciworld:  'bg-indigo-100 text-indigo-800',
  general:   'bg-gray-100 text-gray-600',
}
const ISSUE_STYLE = {
  missing_action_format:        { cls: 'bg-red-100 text-red-700', label: '⚠ 缺少 <action> 格式指令' },
  missing_action_history_placeholder: { cls: 'bg-red-100 text-red-700', label: '⚠ 缺少 {action_history}' },
  unexpected_action_history:    { cls: 'bg-yellow-100 text-yellow-700', label: '~ 意外包含 {action_history}' },
}

// ── Highlight {placeholders} in template text ─────────────────────────────
function HighlightedPrompt({ content }) {
  if (!content) return null
  // Split on {placeholder} patterns (not {{ or }})
  const parts = content.split(/(\{[a-zA-Z_][a-zA-Z0-9_]*\})/g)
  return (
    <pre className="whitespace-pre-wrap text-xs font-mono leading-relaxed text-gray-700 break-words">
      {parts.map((part, i) => {
        if (/^\{[a-zA-Z_][a-zA-Z0-9_]*\}$/.test(part)) {
          return (
            <mark key={i} className="bg-yellow-200 text-yellow-900 rounded px-0.5 not-italic">
              {part}
            </mark>
          )
        }
        return part
      })}
    </pre>
  )
}

// ── Placeholder diff between two prompts ─────────────────────────────────
function PlaceholderDiff({ p1, p2 }) {
  if (!p1 || !p2) return null
  const s1 = new Set(p1.placeholders)
  const s2 = new Set(p2.placeholders)

  const onlyLeft  = [...s1].filter(p => !s2.has(p)).sort()
  const shared    = [...s1].filter(p =>  s2.has(p)).sort()
  const onlyRight = [...s2].filter(p => !s1.has(p)).sort()

  const Col = ({ items, label, bg, text, border, dot }) => (
    <div className={`flex-1 rounded-lg border ${border} ${bg} p-2`}>
      <div className="flex items-center gap-1.5 mb-2">
        <span className={`w-2 h-2 rounded-full ${dot}`} />
        <span className={`text-[11px] font-semibold ${text}`}>{label}</span>
        <span className={`ml-auto text-[10px] font-bold ${text} opacity-70`}>{items.length}</span>
      </div>
      {items.length === 0
        ? <p className="text-[10px] text-gray-400 italic text-center py-1">—</p>
        : items.map(ph => (
            <div key={ph} className="text-[10px] font-mono bg-white/70 rounded px-1.5 py-0.5 mb-1 truncate" title={`{${ph}}`}>
              {`{${ph}}`}
            </div>
          ))
      }
    </div>
  )

  return (
    <div className="text-xs">
      <p className="text-[11px] font-semibold text-gray-500 mb-2 uppercase tracking-wide">占位符对比</p>
      <div className="flex gap-2">
        <Col items={onlyLeft}  label="仅对比A"  bg="bg-blue-50"  text="text-blue-700"  border="border-blue-200"  dot="bg-blue-400" />
        <Col items={shared}    label="共有"      bg="bg-gray-50"  text="text-gray-600"  border="border-gray-200"  dot="bg-gray-400" />
        <Col items={onlyRight} label="仅对比B"  bg="bg-green-50" text="text-green-700" border="border-green-200" dot="bg-green-400" />
      </div>
    </div>
  )
}

// ── Single prompt card in the left sidebar ────────────────────────────────
function PromptListItem({ prompt, isSelected, slot, onSelect }) {
  const slotLabel = slot === 'left' ? '左' : slot === 'right' ? '右' : null
  return (
    <div
      onClick={() => onSelect(prompt)}
      className={`p-2 rounded border cursor-pointer text-xs transition-all ${
        slot
          ? slot === 'left'
            ? 'border-blue-400 bg-blue-50'
            : 'border-green-400 bg-green-50'
          : isSelected
            ? 'border-blue-300 bg-blue-50/50'
            : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
      }`}
    >
      <div className="flex items-center gap-1 flex-wrap mb-1">
        {slotLabel && (
          <span className={`px-1 rounded font-bold text-white text-[10px] ${slot === 'left' ? 'bg-blue-500' : 'bg-green-500'}`}>
            {slotLabel}
          </span>
        )}
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${ENV_STYLE[prompt.env] || ENV_STYLE.general}`}>
          {prompt.env}
        </span>
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${PHASE_STYLE[prompt.phase] || PHASE_STYLE.basic}`}>
          {prompt.phase}
        </span>
        {!prompt.has_history && (
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-gray-100 text-gray-500">no_his</span>
        )}
        {prompt.variant && (
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-pink-100 text-pink-700">{prompt.variant}</span>
        )}
      </div>
      <p className="font-mono text-gray-700 truncate" title={prompt.name}>{prompt.name}</p>
      <p className="text-gray-400 mt-0.5">{prompt.file} · {prompt.char_count} chars</p>
      {prompt.issues?.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {prompt.issues.map(issue => (
            <span key={issue} className={`px-1 rounded text-[10px] ${ISSUE_STYLE[issue]?.cls}`}>
              {ISSUE_STYLE[issue]?.label || issue}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main Prompt Compare View ───────────────────────────────────────────────
export default function PromptCompareView() {
  const [catalog, setCatalog] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [envFilter, setEnvFilter] = useState('all')
  const [phaseFilter, setPhaseFilter] = useState('all')
  const [searchText, setSearchText] = useState('')
  const [issueOnly, setIssueOnly] = useState(false)

  // Two slots for side-by-side comparison
  const [leftPrompt, setLeftPrompt] = useState(null)
  const [rightPrompt, setRightPrompt] = useState(null)
  const [activeSlot, setActiveSlot] = useState('left')  // which slot to assign next click to

  useEffect(() => {
    fetch(`${API_BASE}/prompts`)
      .then(r => {
        if (!r.ok) throw new Error('Failed to fetch prompts')
        return r.json()
      })
      .then(data => {
        setCatalog(data)
        // Auto-select first two prompts for comparison
        if (data.prompts?.length >= 2) {
          setLeftPrompt(data.prompts[0])
          setRightPrompt(data.prompts[1])
        }
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  const envOptions = useMemo(() => {
    if (!catalog) return []
    return ['all', ...Object.keys(catalog.by_env)]
  }, [catalog])

  const phaseOptions = useMemo(() => {
    if (!catalog) return []
    return ['all', ...Object.keys(catalog.by_phase)]
  }, [catalog])

  const filteredPrompts = useMemo(() => {
    if (!catalog) return []
    return catalog.prompts.filter(p => {
      if (envFilter !== 'all' && p.env !== envFilter) return false
      if (phaseFilter !== 'all' && p.phase !== phaseFilter) return false
      if (issueOnly && (!p.issues || p.issues.length === 0)) return false
      if (searchText && !p.name.toLowerCase().includes(searchText.toLowerCase()) &&
          !p.content.toLowerCase().includes(searchText.toLowerCase())) return false
      return true
    })
  }, [catalog, envFilter, phaseFilter, searchText, issueOnly])

  const handleSelectPrompt = (prompt) => {
    if (activeSlot === 'left') {
      setLeftPrompt(prompt)
      setActiveSlot('right')
    } else {
      setRightPrompt(prompt)
      setActiveSlot('left')
    }
  }

  const getSlot = (prompt) => {
    if (!prompt) return null
    if (leftPrompt?.id === prompt.id) return 'left'
    if (rightPrompt?.id === prompt.id) return 'right'
    return null
  }

  if (loading) return <div className="flex-1 flex items-center justify-center text-gray-400">加载 Prompt 目录中...</div>
  if (error) return <div className="flex-1 flex items-center justify-center text-red-500">错误: {error}</div>

  const issueCount = catalog?.known_issues?.length || 0

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* ── 左侧 Prompt 列表 ── */}
      <div className="w-72 flex flex-col bg-white border-r border-gray-200 flex-shrink-0">
        {/* 工具栏 */}
        <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">
              {filteredPrompts.length} / {catalog?.total} 个模板
            </span>
            {issueCount > 0 && (
              <button
                onClick={() => setIssueOnly(v => !v)}
                className={`text-xs px-2 py-0.5 rounded ${issueOnly ? 'bg-red-500 text-white' : 'bg-red-100 text-red-600 hover:bg-red-200'}`}
              >
                ⚠ {issueCount} 问题
              </button>
            )}
          </div>

          {/* 环境过滤 */}
          <div className="flex gap-1 flex-wrap">
            {envOptions.map(e => (
              <button key={e}
                onClick={() => setEnvFilter(e)}
                className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
                  envFilter === e ? 'bg-blue-500 text-white border-blue-500' : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'
                }`}
              >
                {e === 'all' ? '全部' : e}
              </button>
            ))}
          </div>

          {/* 阶段过滤 */}
          <div className="flex gap-1 flex-wrap">
            {phaseOptions.map(p => (
              <button key={p}
                onClick={() => setPhaseFilter(p)}
                className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
                  phaseFilter === p ? 'bg-green-500 text-white border-green-500' : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'
                }`}
              >
                {p === 'all' ? '全部' : p}
              </button>
            ))}
          </div>

          {/* 搜索 */}
          <input
            type="text"
            placeholder="搜索名称或内容..."
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            className="w-full text-xs px-2 py-1 border border-gray-200 rounded focus:outline-none focus:border-blue-300"
          />

          {/* Slot 选择说明 */}
          <div className="text-[10px] text-gray-500 flex items-center gap-1">
            <span>点击分配到:</span>
            <button
              onClick={() => setActiveSlot('left')}
              className={`px-2 py-0.5 rounded font-medium ${activeSlot === 'left' ? 'bg-blue-500 text-white' : 'bg-blue-100 text-blue-700'}`}
            >
              左 (对比A)
            </button>
            <button
              onClick={() => setActiveSlot('right')}
              className={`px-2 py-0.5 rounded font-medium ${activeSlot === 'right' ? 'bg-green-500 text-white' : 'bg-green-100 text-green-700'}`}
            >
              右 (对比B)
            </button>
          </div>
        </div>

        {/* 列表 */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
          {filteredPrompts.map(prompt => (
            <PromptListItem
              key={prompt.id}
              prompt={prompt}
              isSelected={leftPrompt?.id === prompt.id || rightPrompt?.id === prompt.id}
              slot={getSlot(prompt)}
              onSelect={handleSelectPrompt}
            />
          ))}
          {filteredPrompts.length === 0 && (
            <p className="text-gray-400 text-xs text-center mt-4">无匹配结果</p>
          )}
        </div>
      </div>

      {/* ── 右侧对比面板 ── */}
      <div className="flex-1 flex flex-col overflow-hidden bg-gray-50">
        {/* 对比头部 */}
        <div className="grid grid-cols-2 gap-px bg-gray-200 border-b border-gray-200 flex-shrink-0">
          {[['left', leftPrompt, 'bg-blue-50 border-blue-200', 'text-blue-700'],
            ['right', rightPrompt, 'bg-green-50 border-green-200', 'text-green-700']].map(
            ([slot, prompt, bgcls, tcls]) => (
              <div key={slot} className={`px-4 py-3 ${bgcls}`}>
                {prompt ? (
                  <div>
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <span className={`text-xs font-bold ${tcls}`}>{slot === 'left' ? '对比 A' : '对比 B'}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${ENV_STYLE[prompt.env]}`}>{prompt.env}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${PHASE_STYLE[prompt.phase]}`}>{prompt.phase}</span>
                      <span className="text-xs text-gray-400">{prompt.file}</span>
                    </div>
                    <p className="text-sm font-mono font-semibold text-gray-800 truncate">{prompt.name}</p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {prompt.char_count} chars · {prompt.line_count} lines · {prompt.placeholders.length} 占位符
                    </p>
                    {prompt.issues?.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {prompt.issues.map(issue => (
                          <span key={issue} className={`text-[10px] px-1.5 py-0.5 rounded ${ISSUE_STYLE[issue]?.cls}`}>
                            {ISSUE_STYLE[issue]?.label || issue}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <p className={`text-sm ${tcls} opacity-60`}>
                    {slot === 'left' ? '← 点击左侧列表选择对比 A' : '← 点击左侧列表选择对比 B'}
                  </p>
                )}
              </div>
            )
          )}
        </div>

        {/* 占位符对比栏 */}
        {leftPrompt && rightPrompt && (
          <div className="px-4 py-3 bg-white border-b border-gray-200 flex-shrink-0">
            <PlaceholderDiff p1={leftPrompt} p2={rightPrompt} />
          </div>
        )}

        {/* Prompt 内容并排显示 */}
        <div className="flex-1 grid grid-cols-2 gap-px bg-gray-200 overflow-hidden">
          {[leftPrompt, rightPrompt].map((prompt, idx) => (
            <div key={idx} className="bg-white overflow-auto p-4">
              {prompt ? (
                <>
                  {/* Placeholders list */}
                  <div className="mb-3 flex flex-wrap gap-1">
                    {prompt.placeholders.map(ph => (
                      <code key={ph} className="text-[10px] px-1.5 py-0.5 bg-yellow-100 text-yellow-800 rounded border border-yellow-200">
                        {`{${ph}}`}
                      </code>
                    ))}
                  </div>
                  {/* Content with highlighted placeholders */}
                  <HighlightedPrompt content={prompt.content} />
                </>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-300 text-sm">
                  未选择
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
