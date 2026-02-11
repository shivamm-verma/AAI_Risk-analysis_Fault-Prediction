import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  UploadCloud,
  FileText,
  ShieldCheck,
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  Zap,
  AlertCircle,
  Loader,
  Download,
  Check,
  ExternalLink,
} from "lucide-react";

// Analysis type configurations with required fields
// These match the actual models deployed on the backend
const ANALYSIS_TYPES = {
  remaining_useful_life: {
    label: "Remaining Useful Life (RUL) - LSTM",
    description:
      "Predicts remaining operational cycles using LSTM deep learning. Requires CMAPSS format sensor data with minimum 30 rows.",
    modelName: "remaining_useful_life",
    requiredFields: [
      "engine_id",
      "cycle",
      "op1",
      "op2",
      "op3",
      "s1",
      "s2",
      "s3",
      "s4",
      "s5",
      "s6",
      "s7",
      "s8",
      "s9",
      "s10",
      "s11",
      "s12",
      "s13",
      "s14",
      "s15",
      "s16",
      "s17",
      "s18",
      "s19",
      "s20",
      "s21",
    ],
  },
  engine_maintenance: {
    label: "Engine Maintenance Risk",
    description:
      "Assesses engine maintenance requirements (Healthy/Maintenance/Replace). Uses custom feature extraction from a 30-cycle sensor window.",
    modelName: "engine_maintenance",
    requiredFields: [
      "engine_id",
      "cycle",
      "op1",
      "op2",
      "op3",
      "s1",
      "s2",
      "s3",
      "s4",
      "s5",
      "s6",
      "s7",
      "s8",
      "s9",
      "s10",
      "s11",
      "s12",
      "s13",
      "s14",
      "s15",
      "s16",
      "s17",
      "s18",
      "s19",
      "s20",
      "s21",
    ],
  },
  landing_gear_fault: {
    label: "Landing Gear Fault Prediction",
    description:
      "Detects landing gear faults (gas leaks, seal wear, degradation)",
    modelName: "landing_gear_fault",
    requiredFields: [
      "Max_Deflection",
      "Max_Velocity",
      "Settling_Time",
      "Mass",
      "K_Stiffness",
      "B_Damping",
    ],
  },
  landing_gear_rul: {
    label: "Landing Gear RUL",
    description:
      "Predicts remaining useful life cycles for landing gear components",
    modelName: "landing_gear_rul",
    requiredFields: [
      "Max_Deflection",
      "Max_Velocity",
      "Settling_Time",
      "Mass",
      "K_Stiffness",
      "B_Damping",
    ],
  },
  durability: {
    label: "Structural Durability Assessment",
    description:
      "Evaluates structural component durability and degradation status",
    modelName: "durability",
    requiredFields: [
      "Material_Type",
      "Temperature",
      "Pressure",
      "Stress_Level",
      "Cycle_Count",
      "Environmental_Factor",
    ],
  },
};

function Tools() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedAnalysisType, setSelectedAnalysisType] = useState("remaining_useful_life");
  const [availableModels, setAvailableModels] = useState([]);
  const [modelsError, setModelsError] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isUploadSuccess, setIsUploadSuccess] = useState(false);
  const [showFieldsList, setShowFieldsList] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking"); // "checking", "ready", "sleeping", "error"
  const [backendCheckAttempts, setBackendCheckAttempts] = useState(0);
  const [backendError, setBackendError] = useState("");

  const apiBaseUrl = useMemo(
    () => "https://aai-risk-analysis-fault-prediction.onrender.com" || import.meta.env.VITE_API_BASE_URL,
    [],
  );

  const currentAnalysis =
    ANALYSIS_TYPES[selectedAnalysisType] || ANALYSIS_TYPES.remaining_useful_life;
  const requiredFieldsText = currentAnalysis.requiredFields.join(", ");

  const resetError = () => setErrorMessage("");

  useEffect(() => {
    let isMounted = true;

    const checkBackendHealth = async (attempt = 0) => {
      try {
        if (attempt === 0) {
          setBackendStatus("checking");
          setBackendError("");
        }

        const res = await axios.get(`${apiBaseUrl}/health`, { timeout: 8000 });
        if (!isMounted) return;

        if (res.status === 200) {
          setBackendStatus("ready");
          console.log("✅ Backend health check passed");
          console.log("Backend URL:", apiBaseUrl);
          fetchModels();
        }
      } catch (e) {
        if (!isMounted) return;

        const errorMsg = e?.response?.status ? 
          `Status ${e.response.status}: ${e.response.statusText}` : 
          e?.message || "Connection timeout";
        
        console.error(`❌ Backend health check failed (attempt ${attempt + 1}/8):`, errorMsg);
        console.error("Backend URL:", apiBaseUrl);
        console.error("Full error:", e);

        // If backend is sleeping (Render cold start), retry with exponential backoff
        if (attempt < 8) {
          setBackendStatus("sleeping");
          setBackendCheckAttempts(attempt + 1);
          
          const delay = Math.min(1000 * Math.pow(1.5, attempt), 10000); // Max 10s delay
          setTimeout(() => {
            if (isMounted) checkBackendHealth(attempt + 1);
          }, delay);
        } else {
          setBackendStatus("error");
          setBackendError(
            `Connection failed after 8 attempts (${errorMsg}). ` +
            `Backend: ${apiBaseUrl}`
          );
          console.error("❌ Backend connection failed after all retries");
        }
      }
    };

    const fetchModels = async () => {
      try {
        setModelsError("");
        const res = await axios.get(`${apiBaseUrl}/models`);
        const models = res?.data?.models || [];
        if (!isMounted) return;
        setAvailableModels(models);
        if (models.length === 0) {
          setModelsError(
            "No models found on the backend. Add .pkl files in the Model folder and restart the server.",
          );
        }
      } catch (e) {
        if (!isMounted) return;
        setModelsError(
          e?.response?.data?.detail ||
            "Unable to fetch models from backend. Ensure the API is running.",
        );
      }
    };

    checkBackendHealth();

    return () => {
      isMounted = false;
    };
  }, [apiBaseUrl]);

  const manualRetryBackend = () => {
    setBackendStatus("checking");
    setBackendError("");
    setBackendCheckAttempts(0);
    // Trigger a retry by fetching health
    axios.get(`${apiBaseUrl}/health`, { timeout: 8000 })
      .then(() => {
        setBackendStatus("ready");
        const fetchModels = async () => {
          try {
            const res = await axios.get(`${apiBaseUrl}/models`);
            const models = res?.data?.models || [];
            setAvailableModels(models);
            if (models.length === 0) {
              setModelsError(
                "No models found on the backend. Add .pkl files in the Model folder and restart the server.",
              );
            }
          } catch (e) {
            setModelsError(
              e?.response?.data?.detail ||
                "Unable to fetch models from backend. Ensure the API is running.",
            );
          }
        };
        fetchModels();
      })
      .catch((e) => {
        const errorMsg = e?.response?.status ? 
          `Status ${e.response.status}: ${e.response.statusText}` : 
          e?.message || "Connection timeout";
        setBackendStatus("error");
        setBackendError(`Connection failed: ${errorMsg}. Backend: ${apiBaseUrl}`);
      });
  };

  const validateCsv = (file) => {
    if (!file) return "Please select a CSV file.";
    const ok =
      file.name.toLowerCase().endsWith(".csv") ||
      file.type === "text/csv" ||
      file.type === "application/vnd.ms-excel";
    return ok ? "" : "Only CSV files are allowed.";
  };

  const handleFileChange = (e) => {
    resetError();
    const file = e.target.files?.[0];
    const err = validateCsv(file);
    if (err) {
      setSelectedFile(null);
      setErrorMessage(err);
      e.target.value = "";
      return;
    }
    setSelectedFile(file);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    resetError();

    const err = validateCsv(selectedFile);
    if (err) return setErrorMessage(err);
    if (!selectedAnalysisType)
      return setErrorMessage("Please select an analysis type.");
    if (availableModels.length === 0) {
      return setErrorMessage(
        "No models are available on the backend. Add .pkl files in the Model folder.",
      );
    }

    try {
      setIsSubmitting(true);
      setIsUploadSuccess(false);

      // Read CSV file content for storage
      const fileContent = await selectedFile.text();
      sessionStorage.setItem("uploadedCsvData", fileContent);
      sessionStorage.setItem("uploadedFileName", selectedFile.name);

      const fd = new FormData();
      fd.append("file", selectedFile);

      const modelName = currentAnalysis.modelName;
      const res = await axios.post(`${apiBaseUrl}/predict/${modelName}`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      sessionStorage.setItem("predictionResult", JSON.stringify(res.data));
      sessionStorage.setItem("analysisType", selectedAnalysisType);
      setIsUploadSuccess(true);
    } catch (e) {
      setErrorMessage(
        e?.response?.data?.message ||
          e?.response?.data?.detail ||
          e?.message ||
          "Unable to process the CSV right now. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGoToDashboard = () => {
    const stored = sessionStorage.getItem("predictionResult");
    if (stored) {
      navigate("/dashboard", {
        state: { result: JSON.parse(stored), fileName: selectedFile.name },
      });
    }
  };

  return (
    <div className="bg-slate-50 min-h-screen">
      <div className="text-5xl text-center pt-4 font-bold text-blue-800 max-sm:text-3xl">
        Aerisk <div className="text-xl inline max-sm:text-sm max-sm:block">(Aviation Risk Analysis & Fault Prediction)</div>
      </div>
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
                  Upload a single CSV file to generate fault predictions and
                  risk insights.
                </p>
              </div>
            </div>

            <div className="rounded-2xl border border-blue-100 bg-white p-4 sm:p-6 shadow-sm">
              <h2 className="text-lg sm:text-xl font-semibold text-blue-900 mb-4">
                Analysis Types
              </h2>
              <div className="space-y-3 text-sm sm:text-base text-gray-700">
                <div className="border-l-4 border-blue-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">
                    🔧 Remaining Useful Life (LSTM)
                  </p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    LSTM deep learning model for predicting remaining operational cycles. Requires CMAPSS sensor data.
                  </p>
                </div>
                <div className="border-l-4 border-green-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">
                    ⚡ Engine Maintenance Risk
                  </p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    Predicts engine status (Healthy/Maintenance/Replace) using custom sensor feature extraction.
                  </p>
                </div>
                <div className="border-l-4 border-yellow-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">
                    ⚙️ Landing Gear Fault Detection
                  </p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    Detects landing gear faults (gas leaks, seal wear, degradation) from system parameters.
                  </p>
                </div>
                <div className="border-l-4 border-purple-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">
                    📊 Landing Gear RUL
                  </p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    Predicts remaining useful life cycles for landing gear components.
                  </p>
                </div>
                <div className="border-l-4 border-red-600 pl-3 py-2">
                  <p className="font-semibold text-blue-900">
                    🏗️ Structural Durability
                  </p>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    Evaluates structural component durability and degradation status.
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
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 shrink-0" />
                  <span>
                    Select the analysis type based on your use case (RUL,
                    Structural, Landing Gear, or Engine Risk).
                  </span>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 shrink-0" />
                  <span>
                    Prepare a CSV with all required columns for the selected
                    analysis type.
                  </span>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 shrink-0" />
                  <span>Upload the CSV file using the form on the right.</span>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 shrink-0" />
                  <span>
                    Click "Run Analysis" to process your data with the ML model.
                  </span>
                </li>
                <li className="flex gap-3">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-600 shrink-0" />
                  <span>
                    View detailed results and risk assessments on the dashboard.
                  </span>
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
                    Only one CSV file can be uploaded at a time. Maximum 50MB
                    recommended.
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
                <h2 className="text-xl sm:text-2xl font-semibold">
                  Analysis Configuration
                </h2>
              </div>

              {/* Backend Status Indicator */}
              {backendStatus !== "ready" && (
                <div className={`rounded-lg border-2 p-4 flex items-center gap-3 ${
                  backendStatus === "sleeping"
                    ? "border-yellow-300 bg-yellow-50"
                    : backendStatus === "error"
                    ? "border-red-300 bg-red-50"
                    : "border-blue-300 bg-blue-50"
                }`}>
                  {backendStatus === "checking" && (
                    <>
                      <Loader className="w-5 h-5 text-blue-600 animate-spin" />
                      <span className="text-sm text-blue-800 font-medium">
                        Checking backend status... Please wait.
                      </span>
                    </>
                  )}
                  {backendStatus === "sleeping" && (
                    <>
                      <Loader className="w-5 h-5 text-yellow-600 animate-spin" />
                      <div className="text-sm text-yellow-800">
                        <p className="font-medium">Backend is waking up...</p>
                        <p className="text-xs text-yellow-700">
                          This may take a few seconds. Attempt {backendCheckAttempts}/8
                        </p>
                      </div>
                    </>
                  )}
                  {backendStatus === "error" && (
                    <>
                      <AlertTriangle className="w-5 h-5 text-red-600" />
                      <div className="text-sm text-red-800 font-medium flex-1">
                        <p>Backend connection failed.</p>
                        {backendError && (
                          <p className="text-xs text-red-700 mt-1">{backendError}</p>
                        )}
                      </div>
                      <button
                        onClick={manualRetryBackend}
                        className="px-3 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 whitespace-nowrap font-medium"
                      >
                        Retry
                      </button>
                    </>
                  )}
                </div>
              )}

              {backendStatus === "ready" && (
                <div className="rounded-lg border-2 border-green-300 bg-green-50 p-4 flex items-center gap-3">
                  <Check className="w-5 h-5 text-green-600" />
                  <span className="text-sm text-green-800 font-medium">
                    Backend is ready! You can now run analysis.
                  </span>
                </div>
              )}

              <label className="block">
                <span className="text-sm font-medium text-gray-700">
                  Analysis Type *
                </span>
                <select
                  value={selectedAnalysisType}
                  onChange={(e) => {
                    setSelectedAnalysisType(e.target.value);
                    resetError();
                  }}
                  className="mt-2 w-full rounded-lg border border-blue-100 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(ANALYSIS_TYPES).map(([key, value]) => (
                    <option key={key} value={key}>
                      {value.label}
                    </option>
                  ))}
                </select>
                <p className="mt-2 text-xs text-gray-600">
                  {currentAnalysis.description}
                </p>
              </label>

              <div className="rounded-lg bg-blue-50 border border-blue-100 p-3">
                <div className="flex gap-2 text-xs sm:text-sm">
                  <AlertCircle className="h-5 w-5 text-blue-600 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-blue-900 mb-2">
                      Required CSV Columns:
                    </p>
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
                      {showFieldsList ? "Hide full list" : "View details"}
                    </button>
                    {showFieldsList && (
                      <div className="mt-3 p-2 bg-white rounded text-xs text-gray-700 max-h-40 overflow-y-auto border border-blue-200">
                        <p className="font-semibold mb-2">
                          Mandatory fields for {currentAnalysis.label}:
                        </p>
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
                <span className="text-sm font-medium text-gray-700">
                  CSV File *
                </span>
                <p className="mt-2 text-xs text-blue-600 hover:text-blue-800">
                  <a 
                    href="https://github.com/shivamm-verma/AAI_Risk-analysis_Fault-Prediction/tree/main/Model/sample_data" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 underline"
                  >
                    <Download className="w-3 h-3" />
                    Download sample data for testing
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </p>
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
                  <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>{modelsError}</span>
                </div>
              )}

              {errorMessage && (
                <div className="flex gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                  <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>{errorMessage}</span>
                </div>
              )}

              {isUploadSuccess && (
                <div className="flex gap-2 rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-700">
                  <CheckCircle className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>CSV uploaded successfully! Results are ready.</span>
                </div>
              )}

              <button
                type="submit"
                disabled={
                  isSubmitting || availableModels.length === 0 || !selectedFile
                }
                className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-blue-800 px-6 py-3 text-white font-semibold shadow-md hover:bg-blue-900 disabled:opacity-70 disabled:cursor-not-allowed"
              >
                {isSubmitting ? "Processing..." : "Run Analysis"}
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
  );
}

export default Tools;
