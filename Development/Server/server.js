import express from 'express'
import multer from 'multer'
import cors from 'cors'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import dotenv from 'dotenv'

dotenv.config()

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const app = express()
const PORT = process.env.PORT || 5000

// Middleware
app.use(cors())
app.use(express.json())

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads')
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true })
}

// Configure multer for CSV uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir)
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now()
    cb(null, `${timestamp}-${file.originalname}`)
  },
})

const fileFilter = (req, file, cb) => {
  // Only accept CSV files
  if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv')) {
    cb(null, true)
  } else {
    cb(new Error('Only CSV files are allowed'), false)
  }
}

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  },
})

// Routes

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'Server is running', timestamp: new Date().toISOString() })
})

// CSV Upload and Prediction endpoint
app.post('/predict', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No file uploaded',
      })
    }

    const filePath = req.file.path

    // Read the CSV file
    const csvContent = fs.readFileSync(filePath, 'utf-8')

    // TODO: Send CSV to ML model for prediction
    // For now, we'll mock the ML model response
    const mockPrediction = await mockMLModel(csvContent, req.file.originalname)

    // Optional: Clean up the uploaded file
    // fs.unlinkSync(filePath)

    res.json({
      success: true,
      message: 'Prediction completed successfully',
      fileName: req.file.originalname,
      fileSize: req.file.size,
      uploadedAt: new Date().toISOString(),
      ...mockPrediction,
    })
  } catch (error) {
    console.error('Error processing CSV:', error)
    res.status(500).json({
      success: false,
      message: error.message || 'Error processing CSV file',
    })
  }
})

// Mock ML Model function
// Replace this with actual ML model integration
async function mockMLModel(csvContent, fileName) {
  // Simulate ML model processing delay
  await new Promise((resolve) => setTimeout(resolve, 1000))

  // Parse CSV to count rows (simplified)
  const lines = csvContent.trim().split('\n')
  const rowCount = lines.length - 1 // Subtract header

  // Mock prediction response
  return {
    prediction: `Analysis of ${fileName} completed. Found ${rowCount} data points.`,
    risk_level: ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)],
    summary: `Processed ${rowCount} rows from your CSV file. The system analyzed the provided telemetry data and generated risk predictions based on the trained ML model.`,
    confidence_score: (Math.random() * 0.4 + 0.6).toFixed(3),
    faults_detected: Math.floor(Math.random() * 10),
    timestamp: new Date().toISOString(),
  }
}

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Error:', error)

  if (error instanceof multer.MulterError) {
    if (error.code === 'FILE_TOO_LARGE') {
      return res.status(413).json({
        success: false,
        message: 'File size exceeds 50MB limit',
      })
    }
    return res.status(400).json({
      success: false,
      message: 'File upload error: ' + error.message,
    })
  }

  if (error.message === 'Only CSV files are allowed') {
    return res.status(400).json({
      success: false,
      message: 'Only CSV files are allowed',
    })
  }

  res.status(500).json({
    success: false,
    message: error.message || 'Internal server error',
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Endpoint not found',
  })
})

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server is running on http://localhost:${PORT}`)
  console.log(`📁 Uploads directory: ${uploadsDir}`)
  console.log('🚀 Ready to receive CSV files for ML prediction')
})
