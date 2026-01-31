# Risk Analysis Backend

Express.js server for handling CSV uploads and ML model predictions.

## Setup

```bash
npm install
```

## Running

Development mode with auto-reload:
```bash
npm run dev
```

Production mode:
```bash
npm start
```

## API Endpoints

### POST /predict
Upload a CSV file for risk analysis prediction.

**Request:**
- Content-Type: multipart/form-data
- Parameter: `file` (CSV file)

**Response:**
```json
{
  "success": true,
  "message": "Prediction completed successfully",
  "fileName": "data.csv",
  "fileSize": 1024,
  "prediction": "Analysis results...",
  "risk_level": "Medium",
  "summary": "Summary of findings...",
  "confidence_score": "0.85",
  "faults_detected": 3,
  "timestamp": "2026-02-01T00:00:00.000Z"
}
```

### GET /health
Health check endpoint.

## Configuration

Edit `.env` file to configure:
- `PORT`: Server port (default: 5000)
- `CORS_ORIGIN`: Frontend origin for CORS
- `ML_MODEL_ENDPOINT`: Your ML model API endpoint
- `MAX_FILE_SIZE`: Maximum upload file size

## Integration with ML Model

Replace the `mockMLModel()` function in `server.js` with actual ML model API calls. The function currently returns mock data for testing.

## File Uploads

Uploaded CSV files are stored in the `uploads/` directory.
