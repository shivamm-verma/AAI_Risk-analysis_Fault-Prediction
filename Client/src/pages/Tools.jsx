import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import {
  UploadCloud,
  FileText,
  ShieldCheck,
  AlertTriangle,
  ArrowRight,
  CheckCircle,
} from 'lucide-react'

function Tools() {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState(null)
  const [modelName, setModelName] = useState('pickle_example')
  const [availableModels, setAvailableModels] = useState([])
  const [modelsError, setModelsError] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isUploadSuccess, setIsUploadSuccess] = useState(false)

  const apiBaseUrl = useMemo(
    () => import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000',
    []
  )

  const resetError = () => setErrorMessage('')

  useEffect(() => {
    let isMounted = true

    const fetchModels = async () => {
      try {
        setModelsError('')
        const res = await axios.get(`${apiBaseUrl}/models`)
        const models = res?.data?.models || []
        if (!isMounted) return
        setAvailableModels(models)
        if (models.length > 0 && !models.includes(modelName)) {
          setModelName(models[0])
        }
        if (models.length === 0) {
          setModelsError(
            'No models found on the backend. Add a .pkl file in the Model folder and restart the server.'
          )
        }
      } catch (e) {
        if (!isMounted) return
        setModelsError(
          e?.response?.data?.detail ||
            'Unable to fetch models from backend. Ensure the API is running.'
        )
      }
    }

    fetchModels()
    return () => {
      isMounted = false
    }
  }, [apiBaseUrl, modelName])

  const validateCsv = (file) => {
    if (!file) return 'Please select a CSV file.'
    const ok =
      file.name.toLowerCase().endsWith('.csv') ||
      file.type === 'text/csv' ||
      file.type === 'application/vnd.ms-excel'
    return ok ? '' : 'Only CSV files are allowed.'
  }

  const handleFileChange = (e) => {
    resetError()
    const file = e.target.files?.[0]
    const err = validateCsv(file)
    if (err) {
      setSelectedFile(null)
      setErrorMessage(err)
      e.target.value = ''
      return
    }
    setSelectedFile(file)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    resetError()

    const err = validateCsv(selectedFile)
    if (err) return setErrorMessage(err)
    if (!modelName) return setErrorMessage('Please enter a model name.')
    if (availableModels.length === 0) {
      return setErrorMessage(
        'No models are available on the backend. Add a .pkl file in the Model folder.'
      )
    }

    try {
      setIsSubmitting(true)
      setIsUploadSuccess(false)

      const fd = new FormData()
      fd.append('file', selectedFile)

      const res = await axios.post(`${apiBaseUrl}/predict/${modelName}`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      sessionStorage.setItem('predictionResult', JSON.stringify(res.data))
      setIsUploadSuccess(true)
    } catch (e) {
      setErrorMessage(
        e?.response?.data?.message ||
          e?.response?.data?.detail ||
          e?.message ||
          'Unable to process the CSV right now. Please try again.'
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleGoToDashboard = () => {
    const stored = sessionStorage.getItem('predictionResult')
    if (stored) {
      navigate('/dashboard', {
        state: { result: JSON.parse(stored), fileName: selectedFile.name },
      })
    }
  }

  return (
    <div className="bg-slate-50 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <div className="grid gap-8 lg:gap-10 lg:grid-cols-[1.1fr_0.9fr]">
          {/* LEFT */}
          <div className="space-y-6">
            <div className="flex items-start sm:items-center gap-3 text-blue-900">
              <ShieldCheck className="w-8 h-8 sm:w-9 sm:h-9" />
              <div>
                <h1 className="text-2xl sm:text-4xl font-bold">
                  Risk Analysis Toolkit
                </h1>
                <p className="text-sm sm:text-lg text-gray-700 mt-2">
                  Upload a single CSV file to generate fault predictions and risk insights.
                </p>
              </div>
            </div>

            <div className="rounded-2xl border border-blue-100 bg-white p-4 sm:p-6 shadow-sm">
              <h2 className="text-lg sm:text-xl font-semibold text-blue-900 mb-2">
                How it works
              </h2>
              <ul className="space-y-3 text-sm sm:text-base text-gray-700">
                <li className="flex gap-3">
                  <span className="mt-2 h-2 w-2 rounded-full bg-blue-600" />
                  Prepare a clean CSV with system telemetry and fault signals.
                </li>
                <li className="flex gap-3">
                  <span className="mt-2 h-2 w-2 rounded-full bg-blue-600" />
                  Submit the CSV to the ML model for prediction.
                </li>
                <li className="flex gap-3">
                  <span className="mt-2 h-2 w-2 rounded-full bg-blue-600" />
                  View results instantly on the dashboard with actionable insights.
                </li>
              </ul>
            </div>

            <div className="rounded-2xl bg-linear-to-br from-blue-600 to-blue-900 p-4 sm:p-6 text-white shadow-lg">
              <div className="flex gap-4">
                <UploadCloud className="w-8 h-8 sm:w-10 sm:h-10" />
                <div>
                  <h3 className="text-lg sm:text-xl font-semibold">
                    Upload Requirements
                  </h3>
                  <p className="text-xs sm:text-sm text-blue-100 mt-1">
                    Only one CSV file can be uploaded at a time. Maximum 50MB recommended.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT */}
          <div className="rounded-2xl border border-blue-100 bg-white p-4 sm:p-6 shadow-sm">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="flex items-center gap-3 text-blue-900">
                <FileText className="w-6 h-6 sm:w-7 sm:h-7" />
                <h2 className="text-xl sm:text-2xl font-semibold">Upload CSV</h2>
              </div>

              <label className="block">
                <span className="text-sm font-medium text-gray-700">CSV file</span>
                <div className="mt-3 rounded-xl border-2 border-dashed border-blue-200 bg-blue-50/60 p-4 sm:p-6 text-center">
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-gray-700 file:mr-4 file:rounded-lg file:border-0 file:bg-blue-700 file:px-4 file:py-2 file:text-white hover:file:bg-blue-800"
                  />
                  <p className="mt-2 text-xs text-gray-500">
                    Drag and drop is supported. CSV only.
                  </p>
                </div>
              </label>

              <label className="block">
                <span className="text-sm font-medium text-gray-700">Model name</span>
                {availableModels.length > 0 ? (
                  <select
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="mt-2 w-full rounded-lg border border-blue-100 px-3 py-2 text-sm"
                  >
                    {availableModels.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="mt-2 w-full rounded-lg border border-blue-100 px-3 py-2 text-sm"
                    placeholder="e.g. pickle_example"
                  />
                )}
              </label>

              {modelsError && (
                <div className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-700">
                  <AlertTriangle className="h-4 w-4 mt-0.5" />
                  <span>{modelsError}</span>
                </div>
              )}

              {selectedFile && (
                <div className="rounded-xl border border-blue-100 bg-blue-50 p-3 sm:p-4 text-sm text-blue-900">
                  <div className="flex justify-between gap-3">
                    <span className="font-semibold">Selected:</span>
                    <span className="truncate text-gray-700">
                      {selectedFile.name}
                    </span>
                  </div>
                </div>
              )}

              {errorMessage && (
                <div className="flex gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                  <AlertTriangle className="h-4 w-4 mt-0.5" />
                  <span>{errorMessage}</span>
                </div>
              )}

              {isUploadSuccess && (
                <div className="flex gap-2 rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-700">
                  <CheckCircle className="h-4 w-4 mt-0.5" />
                  <span>CSV uploaded successfully! Results are ready.</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isSubmitting || availableModels.length === 0}
                className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-6 py-3 text-white font-semibold shadow-md hover:bg-blue-900 disabled:opacity-70"
              >
                {isSubmitting ? 'Uploading...' : 'Send to ML Model'}
                <ArrowRight className="h-4 w-4" />
              </button>

              {isUploadSuccess && (
                <button
                  type="button"
                  onClick={handleGoToDashboard}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-green-600 px-6 py-3 text-white font-semibold shadow-md hover:bg-green-700"
                >
                  View Results
                  <ArrowRight className="h-4 w-4" />
                </button>
              )}
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Tools
