import { useState, useCallback } from 'react'
import { Send, MessageCircleQuestion } from 'lucide-react'

export default function AskPanel({ clauses }) {
  const [question, setQuestion] = useState('')
  const [thread, setThread] = useState([]) // [{question, answer, error}]
  const [asking, setAsking] = useState(false)
  const hasDocument = (clauses || []).length > 0

  const ask = useCallback(async () => {
    const trimmed = question.trim()
    if (!trimmed || asking || !hasDocument) return
    setAsking(true)
    setQuestion('')
    setThread((prev) => [...prev, { question: trimmed, answer: null, error: null }])
    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, clauses }),
      })
      if (!response.ok) throw new Error(`Server error ${response.status}`)
      const data = await response.json()
      setThread((prev) =>
        prev.map((item, i) => (i === prev.length - 1 ? { ...item, answer: data.answer } : item))
      )
    } catch (err) {
      setThread((prev) =>
        prev.map((item, i) =>
          i === prev.length - 1 ? { ...item, error: err.message || 'Request failed' } : item
        )
      )
    } finally {
      setAsking(false)
    }
  }, [question, asking, clauses, hasDocument])

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 flex flex-col h-[calc(100vh-4rem)] sticky top-8">
      <h2 className="font-semibold mb-3 flex items-center gap-2 shrink-0">
        <MessageCircleQuestion size={18} className="text-brand-400" />
        Ask about this document
      </h2>

      {!hasDocument && (
        <div className="flex-1 min-h-0 flex items-center justify-center text-center px-4">
          <p className="text-gray-600 text-sm">
            Upload and analyse a document to start asking questions about it.
          </p>
        </div>
      )}

      {hasDocument && thread.length === 0 && (
        <div className="flex-1 min-h-0 flex items-center justify-center text-center px-4">
          <p className="text-gray-600 text-sm">
            Document ready -- ask anything about its content below.
          </p>
        </div>
      )}

      {hasDocument && thread.length > 0 && (
        <div className="space-y-3 mb-4 overflow-y-auto flex-1 min-h-0">
          {thread.map((item, index) => (
            <div key={index} className="text-sm">
              <p className="text-gray-200 font-medium">{item.question}</p>
              {item.answer && (
                <p className="text-gray-400 mt-1 whitespace-pre-wrap">{item.answer}</p>
              )}
              {item.error && <p className="text-red-400 mt-1">{item.error}</p>}
              {!item.answer && !item.error && (
                <p className="text-gray-600 mt-1 italic">Thinking…</p>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2 shrink-0">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && ask()}
          placeholder={hasDocument ? "e.g. What's the late payment fee?" : 'Upload a document first…'}
          className="flex-1 min-w-0 rounded-lg bg-gray-950 border border-gray-700 px-3 py-2 text-sm
            text-gray-100 placeholder-gray-600 outline-none focus:border-brand-500
            disabled:opacity-50"
          disabled={asking || !hasDocument}
        />
        <button
          onClick={ask}
          disabled={asking || !question.trim() || !hasDocument}
          className="btn shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Send size={16} />
        </button>
      </div>
      <p className="muted shrink-0">
        Answers are grounded only in this document's text -- informational only, not legal advice.
      </p>
    </div>
  )
}
