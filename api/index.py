from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the parent directory to path to import the processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processor import process_real_estate_ad

app = FastAPI(title="Real Estate Ad Processor API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class RealEstateAdRequest(BaseModel):
    user_input: str
    api_key: str

@app.get("/")
async def root():
    return {"message": "Welcome to Real Estate Ad Processor API. Use POST /api/process-ad to analyze real estate listings."}

@app.post("/process-ad")
async def analyze_real_estate_ad(request: RealEstateAdRequest):
    try:
        if not request.user_input or not request.api_key:
            raise HTTPException(status_code=400, detail="Both user_input and api_key are required")
        
        result = process_real_estate_ad(request.user_input, request.api_key)
        
        # Check if there was an error
        if "error" in result and not any(result[key] for key in result if key != "error"):
            raise HTTPException(status_code=500, detail=f"Error processing input: {result['error']}")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")