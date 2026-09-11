import { memo } from 'react'
import { CheckCircle2, Circle, Loader2, XCircle, X } from 'lucide-react'

// ── Stage status icon ─────────────────────────────────────────────────────────

function StageIcon({ status }) {
  if (status === 'complete') return <CheckCircle2 size={22} className="text-emerald-400 shrink-0" />
  if (status === 'running')
    return <Loader2 size={22} className="text-brand-400 shrink-0 animate-spin" />
  if (status === 'error') return <XCircle size={22} className="text-red-400 shrink-0" />
  return <Circle size={22} className="text-gray-700 shrink-0" />
}

// ── Per-clause progress bar ───────────────────────────────────────────────────

const ClauseProgressBar = memo(function ClauseProgressBar({ index, total }) {
  const pct = total > 0 ? Math.round((index / total) * 100) : 0
  return (
    <div className="mt-2">
      <div className="flex justify-between text-[11px] text-gray-500 mb-1">
        <span>
          clause {index} / {total}
        </span>
        <span>{pct}%</span>
      </div>
      <div className="h-1 w-full rounded-full bg-gray-800 overflow-hidden">
        <div
          className="h-full rounded-full bg-brand-500 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
})

// ── Single stage row ──────────────────────────────────────────────────────────

const StageRow = memo(function StageRow({ stageId, meta, status, clauseProgress }) {
  const isActive = status === 'running'
  const isDone = status === 'complete'
  const isIdle = !status || status === 'idle'

  return (
    <div
      className={`rounded-xl border px-5 py-4 transition-all duration-300
      ${
        isActive
          ? 'border-brand-700 bg-brand-950/40 shadow-lg shadow-brand-900/20'
          : isDone
            ? 'border-emerald-900/50 bg-emerald-950/10'
            : 'border-gray-800 bg-gray-900/30'
      }`}
    >
      <div className="flex items-start gap-4">
        <StageIcon status={status || 'idle'} />
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2">
            <span
              className={`font-mono text-xs font-bold
              ${isActive ? 'text-brand-400' : isDone ? 'text-emerald-400' : 'text-gray-600'}`}
            >
              {stageId}
            </span>
            <span
              className={`font-semibold text-sm truncate
              ${isActive ? 'text-white' : isDone ? 'text-gray-300' : 'text-gray-600'}`}
            >
              {meta.label}
            </span>
          </div>
          <p
            className={`text-xs mt-0.5 font-mono truncate
            ${isActive ? 'text-gray-400' : isDone ? 'text-gray-600' : 'text-gray-700'}`}
          >
            {meta.tech}
          </p>
          {isActive && clauseProgress && clauseProgress.total > 1 && (
            <ClauseProgressBar index={clauseProgress.index} total={clauseProgress.total} />
          )}
        </div>
        {isDone && (
          <span className="text-[10px] text-emerald-600 font-mono shrink-0 mt-0.5">done</span>
        )}
        {isIdle && (
          <span className="text-[10px] text-gray-700 font-mono shrink-0 mt-0.5">pending</span>
        )}
      </div>
    </div>
  )
})

// ── Overall progress header ───────────────────────────────────────────────────

function ProgressHeader({ stageOrder, stages, filename, onCancel }) {
  const completed = stageOrder.filter((s) => stages[s] === 'complete').length
  const pct = Math.round((completed / stageOrder.length) * 100)

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-1">
        <div>
          <h2 className="text-white font-bold text-xl">Analysing document</h2>
          <p className="text-gray-500 text-sm font-mono mt-0.5 truncate max-w-xs">{filename}</p>
        </div>
        <button
          onClick={onCancel}
          className="flex items-center gap-1.5 text-gray-500 hover:text-gray-300
            text-sm transition-colors px-3 py-1.5 rounded-lg hover:bg-gray-800"
          title="Cancel"
        >
          <X size={14} />
          Cancel
        </button>
      </div>
      <div className="flex items-center gap-3 mt-4">
        <div className="flex-1 h-2 rounded-full bg-gray-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-brand-500 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-gray-400 text-sm font-mono w-16 text-right">
          {completed} / {stageOrder.length}
        </span>
      </div>
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────

export default function PipelineProgress({
  filename,
  stages,
  progress,
  stageMeta,
  stageOrder,
  onCancel,
}) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-2xl">
        {/* Logo strip */}
        <div className="flex items-center gap-2 mb-10">
          <span className="text-2xl">⚖️</span>
          <span className="text-xl font-bold text-white">
            FinLingo<span className="text-brand-400">++</span>
          </span>
        </div>

        <ProgressHeader
          stageOrder={stageOrder}
          stages={stages}
          filename={filename}
          onCancel={onCancel}
        />

        <div className="space-y-2">
          {stageOrder.map((stageId) => (
            <StageRow
              key={stageId}
              stageId={stageId}
              meta={stageMeta[stageId]}
              status={stages[stageId]}
              clauseProgress={progress[stageId]}
            />
          ))}
        </div>

        {/* Pulsing indicator */}
        <div className="mt-8 flex items-center gap-3 text-gray-600 text-sm">
          <div className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="block w-1.5 h-1.5 rounded-full bg-brand-600 animate-pulse"
                style={{ animationDelay: `${i * 0.2}s` }}
              />
            ))}
          </div>
          <span>Pipeline running — this may take 1–3 minutes depending on GPU availability</span>
        </div>
      </div>
    </div>
  )
}
