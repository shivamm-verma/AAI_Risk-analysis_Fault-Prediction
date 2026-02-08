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
  Zap,
  AlertCircle,
} from 'lucide-react'

// Analysis type configurations with required fields
const ANALYSIS_TYPES = {
  rul: {
    label: 'Remaining Useful Life (RUL)',
    description: 'Predicts remaining operational cycles for engines using LSTM deep learning',
    modelName: 'rul_lstm_model',
    requiredFields: [
      'UnitNumber',
      'Cycle',
      'OpSet1',
      'OpSet2',
      'Sensor2',
      'Sensor3',
      'Sensor4',
      'Sensor7',
      'Sensor8',
      'Sensor9',
      'Sensor11',
      'Sensor12',
      'Sensor13',
      'Sensor15',
      'Sensor17',
      'Sensor20',
      'Sensor21',
    ],
  },
  structural: {
    label: 'Structural Integrity / Durability',
    description: 'Assesses structural component durability based on design and environmental parameters',
    modelName: 'structural_integrity',
    requiredFields: [
      'Material Type',
      'E (GPa)',
      'ν',
      'ρ (kg/m³)',
      'Tensile Strength (MPa)',
      "Young's Modulus",
      'Altitude (m)',
      'Temperature (°C)',
      'Pressure (Pa)',
      'Operational Life (years)',
      'Wing Span (m)',
      'Fuselage Length (m)',
      'Structural Thickness (mm)',
      'Structural Shape',
      'Load Distribution',
      'Vibration Damping',
      'Computational Time',
      'Weight Efficiency',
      'Quantum Algorithm Type',
      'Number of Iterations',
      'Optimization Time (sec)',
    ],
  },
  landing_gear: {
    label: 'Landing Gear Fault Analysis',
    description: 'Detects landing gear faults and predicts remaining useful life',
    modelName: 'landing_gear',
    requiredFields: [
      'Max_Deflection',
      'Max_Velocity',
      'Settling_Time',
      'Mass',
      'K_Stiffness',
      'B_Damping',
    ],
  },
  engine_risk: {
    label: 'Engine Risk Assessment',
    description: 'Comprehensive engine performance and risk analysis',
    modelName: 'engine_risk',
    requiredFields: [
      'Engine_ID',
      'Temperature',
      'Pressure',
      'Vibration',
      'RPM',
      'Fuel_Flow',
    ],
  },
}

function Tools() {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState(null)
  const [selectedAnalysisType, setSelectedAnalysisType] = useState('rul')
  const [availableModels, setAvailableModels] = useState([])
  const [modelsError, setModelsError] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isUploadSuccess, setIsUploadSuccess] = useState(false)
  const [showFieldsList, setShowFieldsList] = useState(false)

  const apiBaseUrl = useMemo(
    () => import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000',
    []
  )

  const currentAnalysis = ANALYSIS_TYPES[selectedAnalysisType] || ANALYSIS_TYPES.rul
  const requiredFieldsText = currentAnalysis.requiredFields.join(', ')

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
        if (models.length === 0) {
          setModelsError(
            'No models found on the backend. Add .pkl files in the Model folder and restart the server.'
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
  }, [apiBaseUrl])

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
    if (!selectedAnalysisType) return setErrorMessage('Please select an analysis type.')
    if (availableModels.length === 0) {
      return setErrorMessage(
        'No models are available on the backend. Add .pkl files in the Model folder.'
      )
    }

    try {
      setIsSubmitting(true)
      setIsUploadSuccess(false)

      const fd = new FormData()
      fd.append('file', selectedFile)

      const modelName = currentAnalysis.modelName
      const res = await axios.post(`${apiBaseUrl}/predict/${modelName}`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      sessionStorage.setItem('predictionResult', JSON.stringify(res.data))
      sessionStorage.setItem('analysisType', selectedAnalysisType)
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
              <h2 className="text-lg sm:text-xl font-semibold text-blue-900 mb-4">
                Analysis Types
              </h2>
              <div className="space-y-3 text-sm sm:text-base text-gray-700">
                <div className="border-l-4 border-blue-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">🔧 Remaining Useful Life (RUL)</p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    LSTM-based prediction for engine operational cycles remaining. Requires CMAPSS format data with sensor readings.
                  </p>
                </div>
                <div className="border-l-4 border-green-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">🏗️ Structural Integrity</p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    Material & design analysis for durability assessment. Uses 21 structural parameters.
                  </p>
                </div>
                <div className="border-l-4 border-yellow-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">⚙️ Landing Gear Fault</p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    Detects landing gear faults (gas leaks, seal wear, degradation). Includes RUL prediction.
                  </p>
                </div>
                <div className="border-l-4 border-red-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">⚡ Engine Risk (Coming Soon)</p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    Comprehensive engine performance and risk assessment tool.
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-blue-100 bg-white p-4 sm:p-6 shadow-sm">
              <h2 className="text-lg sm:text-xl font-semibold text-blue-900 mb-2">
                How it works
              </h2>
              <ul className="space-y-3 text-sm sm:text-base text-gray-700">
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 flex-shrink-0" />
                  <span>Select the analysis type based on your use case (RUL, Structural, Landing Gear, or Engine Risk).</span>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 flex-shrink-0" />
                  <span>Prepare a CSV with all required columns for the selected analysis type.</span>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 flex-shrink-0" />
                  <span>Upload the CSV file using the form on the right.</span>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 flex-shrink-0" />
                  <span>Click "Run Analysis" to process your data with the ML model.</span>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 flex-shrink-0" />
                  <span>View detailed results and risk assessments on the dashboard.</span>
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
                <Zap className="w-6 h-6 sm:w-7 sm:h-7" />
                <h2 className="text-xl sm:text-2xl font-semibold">Analysis Configuration</h2>
              </div>

              <label className="block">
                <span className="text-sm font-medium text-gray-700">Analysis Type *</span>
                <select
                  value={selectedAnalysisType}
                  onChange={(e) => {
                    setSelectedAnalysisType(e.target.value)
                    resetError()
                  }}
                  className="mt-2 w-full rounded-lg border border-blue-100 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(ANALYSIS_TYPES).map(([key, value]) => (
                    <option key={key} value={key}>
                      {value.label}
                    </option>
                  ))}
                </select>
                <p className="mt-2 text-xs text-gray-600">{currentAnalysis.description}</p>
              </label>

              <div className="rounded-lg bg-blue-50 border border-blue-100 p-3">
                <div className="flex gap-2 text-xs sm:text-sm">
                  <AlertCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-blue-900 mb-2">Required CSV Columns:</p>
                    <div className="flex flex-wrap gap-1">
                      {currentAnalysis.requiredFields.map((field, idx) => (
                        <span
                          key={idx}
                          className="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-mono"
                        >
                          {field}
                        </span>
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowFieldsList(!showFieldsList)}
                      className="mt-3 text-xs text-blue-600 hover:text-blue-800 underline"
                    >
                      {showFieldsList ? 'Hide full list' : 'View details'}
                    </button>
                    {showFieldsList && (
                      <div className="mt-3 p-2 bg-white rounded text-xs text-gray-700 max-h-40 overflow-y-auto border border-blue-200">
                        <p className="font-semibold mb-2">Mandatory fields for {currentAnalysis.label}:</p>
                        <ul className="list-disc list-inside space-y-1">
                          {currentAnalysis.requiredFields.map((field, idx) => (
                            <li key={idx}>{field}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <label className="block">
                <span className="text-sm font-medium text-gray-700">CSV File *</span>
                <div className="mt-3 rounded-xl border-2 border-dashed border-blue-200 bg-blue-50/60 p-4 sm:p-6 text-center">
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-gray-700 file:mr-4 file:rounded-lg file:border-0 file:bg-blue-700 file:px-4 file:py-2 file:text-white hover:file:bg-blue-800"
                  />
                  <p className="mt-2 text-xs text-gray-500">
                    Drag and drop is supported. CSV only. Max 50MB recommended.
                  </p>
                </div>
              </label>

              {selectedFile && (
                <div className="rounded-xl border border-blue-100 bg-blue-50 p-3 sm:p-4 text-sm text-blue-900">
                  <div className="flex justify-between gap-3">
                    <span className="font-semibold">Selected File:</span>
                    <span className="truncate text-gray-700">
                      {selectedFile.name}
                    </span>
                  </div>
                </div>
              )}

              {modelsError && (
                <div className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-700">
                  <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <span>{modelsError}</span>
                </div>
              )}

              {errorMessage && (
                <div className="flex gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                  <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <span>{errorMessage}</span>
                </div>
              )}

              {isUploadSuccess && (
                <div className="flex gap-2 rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-700">
                  <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <span>CSV uploaded successfully! Results are ready.</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isSubmitting || availableModels.length === 0 || !selectedFile}
                className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-6 py-3 text-white font-semibold shadow-md hover:bg-blue-900 disabled:opacity-70 disabled:cursor-not-allowed"
              >
                {isSubmitting ? 'Processing...' : 'Run Analysis'}
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
