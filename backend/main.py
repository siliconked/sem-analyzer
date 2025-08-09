from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import traceback

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import your existing code
from backend.sem_analyzer import SEMConfig, EnhancedSEMAnalyzer  # Your existing SEM analyzer
from scraper import scrape_website  # Your existing scraper

app = FastAPI(title="SEM Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SEMAnalysisRequest(BaseModel):
    brand_website: str
    competitor_website: str
    service_locations: List[str]
    shopping_budget: float
    search_budget: float
    pmax_budget: float
    conversion_rate: Optional[float] = 0.02
    min_search_volume: Optional[int] = 500
    target_roas: Optional[float] = 4.0
    avg_order_value: Optional[float] = 150.0

class AnalysisResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

@app.get("/")
async def root():
    return {"message": "SEM Analyzer API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_sem_strategy(request: SEMAnalysisRequest):
    try:
        print(f"🚀 Starting analysis for {request.brand_website}")
        
        # Create config using your existing SEMConfig
        config = SEMConfig(
            brand_website=request.brand_website,
            competitor_website=request.competitor_website,
            service_locations=request.service_locations,
            shopping_budget=request.shopping_budget,
            search_budget=request.search_budget,
            pmax_budget=request.pmax_budget,
            conversion_rate=request.conversion_rate,
            min_search_volume=request.min_search_volume,
            target_roas=request.target_roas,
            avg_order_value=request.avg_order_value
        )
        
        # Use your scraper to get content
        print("📥 Scraping websites...")
        try:
            brand_data = scrape_website(request.brand_website)
            brand_content = brand_data['all_text'] if brand_data else "Fallback brand content"
        except Exception as e:
            print(f"⚠️ Brand scraping failed: {e}")
            brand_content = f"Website: {request.brand_website}. Professional business website."
        
        try:
            competitor_data = scrape_website(request.competitor_website)
            competitor_content = competitor_data['all_text'] if competitor_data else "Fallback competitor content"
        except Exception as e:
            print(f"⚠️ Competitor scraping failed: {e}")
            competitor_content = f"Competitor: {request.competitor_website}. Business competitor."
        
        # Use your existing SEM analyzer
        print("🔍 Running SEM analysis...")
        analyzer = EnhancedSEMAnalyzer(config)
        strategy = analyzer.generate_sem_strategy(brand_content, competitor_content)
        
        return AnalysisResponse(
            status="success",
            message="SEM analysis completed successfully!",
            data=strategy
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return AnalysisResponse(
            status="error",
            message=f"Analysis failed: {str(e)}",
            data=None
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
