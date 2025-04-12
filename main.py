from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

# Import the processor module
from processor import process_real_estate_ad

app = FastAPI(title="Real Estate Ad Processor API")

class RealEstateAdRequest(BaseModel):
    user_input: str
    api_key: str

@app.get("/")
async def root():
    return {"message": "Welcome to Real Estate Ad Processor API. Use POST /process-ad to analyze real estate listings."}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)