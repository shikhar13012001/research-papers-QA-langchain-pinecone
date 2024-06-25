'use client'
import {
  useState
} from 'react'
import { Button } from '@/components/ui/button'
import { PieChartIcon } from '@radix-ui/react-icons'
// These styles apply to every route in the application
import './globals.css'
export default function Home() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)
  async function createIndexAndEmbeddings() {
    try {
      const result = await fetch('/api/setup', {
        method: "POST"
      })
      const json = await result.json()
      console.log('result: ', json)
    } catch (err) {
      console.log('err:', err)
    }
  }
  async function sendQuery() {
    if (!query) return
    setResult('')
    setLoading(true)
    try {
      const result = await fetch('/api/read', {
        method: "POST",
        body: JSON.stringify(query)
      })
      const json = await result.json()
      setResult(json.data)
      setLoading(false)
    } catch (err) {
      console.log('err:', err)
      setLoading(false)
    }
  }
  return (
    <main className="flex flex-col justify-center items-center p-24 space-y-6 w-full">
    <p className="text-4xl font-bold">
      Langchain, Pinecone, and LLM with Next.js
    </p>
    <input
      className="mt-3 rounded border w-[400px] text-black px-4 py-2"
      onChange={e => setQuery(e.target.value)}
      placeholder="Type your query here"
    />
    <Button
    variant={"default"}
      className="w-[400px] mt-3 bg-blue-500 text-white py-2 rounded hover:bg-blue-600"
      onClick={sendQuery}
    >
      Ask AI
    </Button>
    {loading && <PieChartIcon className="my-5 w-8 h-8 animate-spin" />}
    {result && (
      <p className="my-8 border p-4 rounded w-[400px] bg-gray-900">
        {result}
      </p>
    )}
    <Button
      className="w-[400px] mt-2 border border-blue-500 text-blue-500 py-2 rounded hover:bg-blue-50"
      variant="outline"
      onClick={createIndexAndEmbeddings}
    >
      Create index and embeddings
    </Button>
  </main>
  
  )
}
