from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from typing import Dict, Any

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

@app.get("/")
async def root():
    return {"message": "Welcome to Real Estate Ad Processor API. Use POST /api/process-ad to analyze real estate listings."}


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
    
# @app.post("/process-ad")
# async def analyze_real_estate_ad(request: RealEstateAdRequest):
#     try:
#         if not request.user_input:
#             raise HTTPException(status_code=400, detail="user_input required")
        
#         print("\n\n=============================================")
#         print(request.user_input)
#         print("==================================")
#         result = process_real_estate_ad(request.user_input)
        
#         # Check if there was an error
#         if "error" in result and not any(result[key] for key in result if key != "error"):
#             raise HTTPException(status_code=500, detail=f"Error processing input: {result['error']}")
            
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")