import { useState, useRef, useCallback } from 'react'
import { Upload, FileText, AlertCircle, CheckCircle2, Scale } from 'lucide-react'

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function UploadCard({ onUpload, error, onClearError, runtimeMeta }) {
  const acceptedExtensions = runtimeMeta?.accepted_extensions || ['.pdf', '.docx', '.txt']
  const maxMb = runtimeMeta?.max_upload_mb || 30
  const models = runtimeMeta?.models || {}
  const tags = [
    `${models.simplifier || 'Configured simplifier'}${models.simplifier_adapter_configured ? ' + adapter' : ''}`,
    `${models.verifier || 'Configured verifier'}${models.verifier_adapter_configured ? ' + adapter' : ''}`,
    'BM25 / FAISS regulatory retrieval',
    'FastAPI + React',
  ]
  const [dragging, setDragging] = useState(false)
  const [selectedFile, setSelected] = useState(null)
  const [fileError, setFileError] = useState(null)
  const inputRef = useRef(null)

  const validateAndSet = useCallback(
    (file) => {
      if (!file) return
      const ext = '.' + file.name.split('.').pop().toLowerCase()
      if (!acceptedExtensions.includes(ext)) {
        setFileError(`Unsupported file type "${ext}". Please upload PDF, DOCX, or TXT.`)
        setSelected(null)
        return
      }
      if (file.size > maxMb * 1024 * 1024) {
        setFileError(`File is ${formatBytes(file.size)} — exceeds the ${maxMb} MB limit.`)
        setSelected(null)
        return
      }
      setFileError(null)
      onClearError?.()
      setSelected(file)
    },
    [acceptedExtensions, maxMb, onClearError],
  )

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragging(false)
      validateAndSet(e.dataTransfer.files[0])
    },
    [validateAndSet],
  )

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragging(true)
  }
  const handleDragLeave = () => setDragging(false)

  const displayError = fileError || error

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="mb-12 text-center">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Scale className="text-brand-400" size={36} strokeWidth={1.5} />
          <h1 className="text-5xl font-bold tracking-tight text-white">
            FinLingo<span className="text-brand-400">++</span>
          </h1>
        </div>
        <p className="text-gray-400 text-lg max-w-lg text-balance">
          Seven-stage explainable RAG pipeline with sentence-level faithfulness verification for
          consumer financial documents.
        </p>
        <div className="mt-5 flex flex-wrap justify-center gap-2 text-xs">
          {tags.map((tag) => (
            <span
              key={tag}
              className="px-2.5 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-400 font-mono"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* ── Drop Zone ───────────────────────────────────────────────────── */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload financial document"
        className={`w-full max-w-xl rounded-2xl border-2 border-dashed p-14 text-center
          cursor-pointer transition-all duration-200 outline-none focus-visible:ring-2
          focus-visible:ring-brand-500
          ${
            dragging
              ? 'border-brand-400 bg-brand-900/20 scale-[1.01]'
              : selectedFile
                ? 'border-emerald-500/60 bg-emerald-950/10'
                : 'border-gray-700 bg-gray-900/50 hover:border-gray-500 hover:bg-gray-900'
          }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={acceptedExtensions.join(',')}
          className="hidden"
          onChange={(e) => validateAndSet(e.target.files[0])}
        />

        {selectedFile ? (
          <>
            <CheckCircle2 className="mx-auto mb-3 text-emerald-400" size={44} strokeWidth={1.5} />
            <p className="text-emerald-300 font-semibold text-lg mb-1">File ready</p>
            <div className="flex items-center justify-center gap-2 text-gray-400 text-sm">
              <FileText size={14} />
              <span className="font-mono">{selectedFile.name}</span>
              <span className="text-gray-600">·</span>
              <span>{formatBytes(selectedFile.size)}</span>
            </div>
            <p className="mt-3 text-gray-600 text-xs">Click or drop to replace</p>
          </>
        ) : (
          <>
            <Upload className="mx-auto mb-4 text-gray-500" size={44} strokeWidth={1.5} />
            <p className="text-white font-semibold text-lg mb-1">
              {dragging ? 'Drop it here' : 'Drop your financial document'}
            </p>
            <p className="text-gray-500 text-sm">or click to browse</p>
            <p className="mt-3 text-gray-600 text-xs">
              {acceptedExtensions.map((ext) => ext.slice(1).toUpperCase()).join(' · ')} — max{' '}
              {maxMb} MB
            </p>
          </>
        )}
      </div>

      {/* ── Error banner ─────────────────────────────────────────────────── */}
      {displayError && (
        <div
          className="mt-4 w-full max-w-xl flex items-start gap-3 rounded-xl
          bg-red-950/50 border border-red-800/60 px-4 py-3 text-red-300 text-sm"
        >
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>{displayError}</span>
        </div>
      )}

      {/* ── Run Button ───────────────────────────────────────────────────── */}
      {selectedFile && !fileError && (
        <button
          onClick={() => onUpload(selectedFile)}
          className="mt-6 px-10 py-3.5 bg-brand-600 hover:bg-brand-500
            active:bg-brand-700 text-white font-semibold rounded-xl
            transition-colors shadow-lg shadow-brand-900/40
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-400"
        >
          Run 7-stage analysis →
        </button>
      )}

      {/* ── Pipeline summary ─────────────────────────────────────────────── */}
      <div className="mt-16 w-full max-w-2xl grid grid-cols-7 gap-1">
        {[
          'S1 Parse',
          'S2 Simplify',
          'S3 Retrieve',
          'S4 Rerank',
          'S5 Risk',
          'S6 Verify',
          'S7 Report',
        ].map((label, i) => (
          <div key={i} className="text-center">
            <div className="h-1 w-full rounded bg-gray-800 mb-1" />
            <span className="text-gray-600 text-[10px] leading-tight block">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
