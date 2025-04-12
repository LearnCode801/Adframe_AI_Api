from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from typing import Dict, Any
# Import the processor module
from processor import process_real_estate_ad

app = FastAPI(title="Real Estate Ad Processor API")

class RealEstateAdRequest(BaseModel):
    user_input: str

@app.get("/")
async def root():
    return {"message": "Welcome to Real Estate Ad Processor API. Use POST /process-ad to analyze real estate listings."}

@app.post("/process-ad")
async def analyze_real_estate_ad(data: Dict[str, Any] = Body(...)):
    try:
        if not data:
            raise HTTPException(status_code=400, detail="JSON data required")
        
        print("\n\n=============================================")
        print(type(data))
        print(data)
        
        print("==================================")
        
        # Pass the JSON data directly to your processor
        result = process_real_estate_ad(data)
        
        # Check if there was an error
        if "error" in result and not any(result[key] for key in result if key != "error"):
            raise HTTPException(status_code=500, detail=f"Error processing input: {result['error']}")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)