import React, { useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { BarChart3, FileCheck2, ShieldAlert, ArrowLeft } from 'lucide-react'

export default function Dashboard() {
  const location = useLocation()
  const navigate = useNavigate()

  const dashboardData = useMemo(() => {
    const stateResult = location.state?.result
    if (stateResult) {
      return stateResult
    }

    const stored = sessionStorage.getItem('predictionResult')
    return stored ? JSON.parse(stored) : null
  }, [location.state])

  const fileName = location.state?.fileName || 'Uploaded CSV'

  if (!dashboardData) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-12">
        <div className="rounded-2xl border border-blue-100 bg-white p-10 text-center shadow-sm">
          <ShieldAlert className="mx-auto h-10 w-10 text-blue-900" />
          <h1 className="mt-4 text-3xl font-bold text-blue-900">No results yet</h1>
          <p className="mt-2 text-gray-600">
            Upload a CSV file in the tools section to generate predictions.
          </p>
          <button
            onClick={() => navigate('/tools')}
            className="mt-6 inline-flex items-center gap-2 rounded-lg bg-blue-800 px-5 py-3 text-white font-semibold hover:bg-blue-900"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Tools
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-slate-50">
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold text-blue-900">Prediction Dashboard</h1>
            <p className="mt-2 text-gray-700">
              Results generated for: <span className="font-semibold">{fileName}</span>
            </p>
          </div>
          <button
            onClick={() => navigate('/tools')}
            className="inline-flex items-center gap-2 rounded-lg border border-blue-200 bg-white px-4 py-2 text-blue-900 font-semibold hover:bg-blue-50"
          >
            <ArrowLeft className="h-4 w-4" />
            Upload another
          </button>
        </div>

        <div className="mt-10 grid gap-6 lg:grid-cols-3">
          <div className="rounded-2xl border border-blue-100 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-3 text-blue-900">
              <BarChart3 className="h-6 w-6" />
              <h2 className="text-lg font-semibold">Summary</h2>
            </div>
            <p className="mt-3 text-sm text-gray-600">
              {dashboardData?.summary || 'Summary details will appear here once the model responds.'}
            </p>
          </div>

          <div className="rounded-2xl border border-blue-100 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-3 text-blue-900">
              <FileCheck2 className="h-6 w-6" />
              <h2 className="text-lg font-semibold">Prediction Output</h2>
            </div>
            <div className="mt-3 text-sm text-gray-700 whitespace-pre-wrap wrap-break-word">
              {dashboardData?.prediction || dashboardData?.result || 'Prediction output will be listed here.'}
            </div>
          </div>

          <div className="rounded-2xl border border-blue-100 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-3 text-blue-900">
              <ShieldAlert className="h-6 w-6" />
              <h2 className="text-lg font-semibold">Risk Level</h2>
            </div>
            <p className="mt-3 text-sm text-gray-700">
              {dashboardData?.risk_level || dashboardData?.risk || 'Risk level will appear here.'}
            </p>
          </div>
        </div>

        <div className="mt-8 rounded-2xl border border-blue-100 bg-white p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-blue-900">Raw Response</h3>
          <pre className="mt-3 max-h-96 overflow-auto rounded-lg bg-slate-900 p-4 text-xs text-slate-100">
            {JSON.stringify(dashboardData, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  )
}
