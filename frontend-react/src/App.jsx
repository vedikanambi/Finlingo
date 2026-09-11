import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import UploadCard from './components/UploadCard.jsx'
import PipelineProgress from './components/PipelineProgress.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'
import AskPanel from './components/AskPanel.jsx'

// ── Stage metadata ────────────────────────────────────────────────────────────

export const DEFAULT_STAGE_META = {
  S1: { label: 'Document Parser', tech: 'Configured parser and OCR fallback', icon: '📄' },
  S2: { label: 'Simplification Engine', tech: 'Configured simplification model', icon: '✍️' },
  S3: {
    label: 'Regulatory Retrieval',
    tech: 'Configured regulatory corpus and retrieval method',
    icon: '🔍',
  },
  S4: { label: 'Cross-Encoder Reranking', tech: 'Configured cross-encoder', icon: '🎯' },
  S5: { label: 'Risk Classifier', tech: 'Configured risk model and taxonomy', icon: '⚠️' },
  S6: {
    label: 'Faithfulness Verifier',
    tech: 'Configured NLI verifier and sentence attribution',
    icon: '🔬',
  },
  S7: { label: 'Report Generator', tech: 'FastAPI · JSON · React', icon: '📊' },
}

const STAGE_ORDER = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

// ── Phase definitions ─────────────────────────────────────────────────────────
// idle → streaming → complete
// idle → streaming → error

export default function App() {
  const [phase, setPhase] = useState('idle') // 'idle' | 'streaming' | 'complete' | 'error'
  const [filename, setFilename] = useState('')
  const [stages, setStages] = useState({}) // { S1: 'idle'|'running'|'complete' }
  const [progress, setProgress] = useState({}) // { S2: { index: 3, total: 10 } }
  const [liveLabels, setLiveLabels] = useState({}) // { S3: 'BM25 retrieval' } -- from the running backend
  const [disabledStages, setDisabledStages] = useState({}) // { S4: true } -- stage ran but was a no-op by config
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)
  const [runtimeMeta, setRuntimeMeta] = useState(null)

  useEffect(() => {
    fetch('/api/runtime-metadata')
      .then((response) => (response.ok ? response.json() : null))
      .then(setRuntimeMeta)
      .catch(() => setRuntimeMeta(null))
  }, [])

  const stageMeta = useMemo(() => {
    const base = !runtimeMeta
      ? DEFAULT_STAGE_META
      : {
          ...DEFAULT_STAGE_META,
          S2: {
            ...DEFAULT_STAGE_META.S2,
            tech: `${runtimeMeta.models?.simplifier || 'Configured model'}${runtimeMeta.models?.simplifier_adapter_configured ? ' + adapter' : ' (adapter not reported)'}`,
          },
          S3: {
            ...DEFAULT_STAGE_META.S3,
            // Default is BM25 (not FAISS); actual method is reported live per-run via liveLabels.
            tech: `${runtimeMeta.models?.embedding || 'Configured embedding'} · ${runtimeMeta.models?.embedding_dimensions || '?'} dimensions`,
          },
          S4: { ...DEFAULT_STAGE_META.S4, tech: runtimeMeta.models?.reranker || 'Configured cross-encoder' },
          S5: {
            ...DEFAULT_STAGE_META.S5,
            tech: `${runtimeMeta.models?.risk || 'Configured model'} · ${runtimeMeta.risk_labels?.length || '?'} classes`,
          },
          S6: {
            ...DEFAULT_STAGE_META.S6,
            tech: `${runtimeMeta.models?.verifier || 'Configured verifier'}${runtimeMeta.models?.verifier_adapter_configured ? ' + adapter' : ' (adapter not reported)'}`,
          },
        }
    // Live per-run labels (e.g. "BM25 retrieval") from the backend's SSE
    // stage_start events always win over static config-derived guesses.
    return Object.fromEntries(
      STAGE_ORDER.map((stageId) => {
        let meta = liveLabels[stageId] ? { ...base[stageId], tech: liveLabels[stageId] } : base[stageId]
        // S4 is disabled by default: the deep-dive found reranking demotes the correct chunk more than it helps.
        if (disabledStages[stageId]) {
          meta = { ...meta, tech: 'Disabled by default (reranker demotes correct chunks more often than it helps -- see deep-dive)' }
        }
        return [stageId, meta]
      }),
    )
  }, [runtimeMeta, liveLabels, disabledStages])

  // ── SSE event handler ───────────────────────────────────────────────────────
  const handleEvent = useCallback((event) => {
    switch (event.event) {
      case 'stage_start':
        setStages((prev) => ({ ...prev, [event.stage]: 'running' }))
        if (event.total) {
          setProgress((prev) => ({ ...prev, [event.stage]: { index: 0, total: event.total } }))
        }
        if (event.label) {
          setLiveLabels((prev) => ({ ...prev, [event.stage]: event.label }))
        }
        break

      case 'stage_complete':
        setStages((prev) => ({ ...prev, [event.stage]: 'complete' }))
        if (event.disabled) {
          setDisabledStages((prev) => ({ ...prev, [event.stage]: true }))
        }
        break

      case 'clause_progress':
      case 'stage_progress':
        setProgress((prev) => ({
          ...prev,
          [event.stage]: { index: event.index ?? event.current ?? 0, total: event.total },
        }))
        break

      case 'complete':
        setResult(event.data)
        setPhase('complete')
        break

      case 'error':
        setError(event.message || 'Unknown pipeline error')
        setPhase('error')
        break

      default:
        break
    }
  }, [])

  // ── Upload handler — opens SSE stream via fetch + ReadableStream ────────────
  const handleUpload = useCallback(
    async (file) => {
      setPhase('streaming')
      setFilename(file.name)
      setStages(Object.fromEntries(STAGE_ORDER.map((s) => [s, 'idle'])))
      setProgress({})
      setLiveLabels({})
      setDisabledStages({})
      setResult(null)
      setError(null)

      const formData = new FormData()
      formData.append('file', file)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const response = await fetch('/api/analyse/stream', {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        })

        if (!response.ok) {
          const text = await response.text()
          throw new Error(`Server error ${response.status}: ${text}`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          // SSE lines end with \n\n; split and process complete events only.
          const parts = buffer.split('\n\n')
          buffer = parts.pop() // keep the incomplete trailing fragment
          for (const part of parts) {
            const line = part.trim()
            if (line.startsWith('data: ')) {
              try {
                handleEvent(JSON.parse(line.slice(6)))
              } catch {
                // Malformed JSON — skip.
              }
            }
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message)
          setPhase('error')
        }
      }
    },
    [handleEvent],
  )

  const handleReset = useCallback(() => {
    abortRef.current?.abort()
    setPhase('idle')
    setFilename('')
    setStages({})
    setProgress({})
    setLiveLabels({})
    setDisabledStages({})
    setResult(null)
    setError(null)
  }, [])

  // ── Render ──────────────────────────────────────────────────────────────────
  const clauses = result?.clauses || []
  return (
    <div className="min-h-screen bg-gray-950 font-sans">
      <div className="max-w-[100rem] mx-auto px-5 py-8 grid grid-cols-[1fr_22rem] gap-6 items-start">
        <div className="min-w-0">
          {phase === 'idle' || phase === 'error' ? (
            <UploadCard
              onUpload={handleUpload}
              error={error}
              onClearError={() => setError(null)}
              runtimeMeta={runtimeMeta}
            />
          ) : phase === 'streaming' ? (
            <PipelineProgress
              filename={filename}
              stages={stages}
              progress={progress}
              stageMeta={stageMeta}
              stageOrder={STAGE_ORDER}
              onCancel={handleReset}
            />
          ) : (
            <ResultsPanel result={result} onReset={handleReset} />
          )}
        </div>

        <AskPanel clauses={clauses} />
      </div>
    </div>
  )
}
