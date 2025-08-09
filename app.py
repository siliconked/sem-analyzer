import streamlit as st
import requests
import pandas as pd
import json
import os
import time
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Configure Streamlit
st.set_page_config(
    page_title="SEM Strategy Analyzer",
    page_icon="📊",
    layout="wide"
)

# Get API keys from Streamlit secrets
try:
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    SERPER_API_KEY = ""
    GEMINI_API_KEY = ""

@dataclass
class SEMConfig:
    brand_website: str
    competitor_website: str
    service_locations: List[str]
    shopping_budget: float
    search_budget: float
    pmax_budget: float
    conversion_rate: float = 0.02
    min_search_volume: int = 500
    target_roas: float = 4.0
    avg_order_value: float = 150.0

@dataclass
class KeywordData:
    keyword: str
    search_volume: int
    competition: str
    cpc_low: float
    cpc_high: float
    keyword_difficulty: int = 0
    search_intent: str = "commercial"
    relevance_score: float = 0.0
    estimated_cpa: float = 0.0
    projected_roas: float = 0.0

class SerperKeywordResearch:
    def __init__(self):
        self.api_key = SERPER_API_KEY
        self.base_url = "https://google.serper.dev"
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def get_search_volume_data(self, keywords: List[str], location: str = "us") -> Dict[str, Dict]:
        keyword_data = {}
        for i, keyword in enumerate(keywords[:10]):  # Limit for free deployment
            try:
                if i > 0:
                    time.sleep(1)
                search_data = self._search_keyword(keyword, location)
                if search_data:
                    keyword_data[keyword] = self._analyze_search_results(keyword, search_data)
            except Exception as e:
                continue
        return keyword_data
    
    def _search_keyword(self, keyword: str, location: str = "us") -> Dict:
        try:
            payload = {"q": keyword, "gl": location, "hl": "en", "num": 10}
            response = requests.post(f"{self.base_url}/search", json=payload, headers=self.headers, timeout=10)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def _analyze_search_results(self, keyword: str, search_data: Dict) -> Dict:
        total_results = search_data.get('searchInformation', {}).get('totalResults', '0')
        total_results = int(total_results.replace(',', '')) if isinstance(total_results, str) else 0
        
        ads_results = search_data.get('ads', [])
        organic_results = search_data.get('organic', [])
        
        # Estimate metrics
        search_volume = max(500, int(total_results / 1000 * (1 + len(ads_results) * 0.5)))
        competition = "HIGH" if len(ads_results) >= 4 else "MEDIUM" if len(ads_results) >= 2 else "LOW"
        cpc_low = 0.5 + len(ads_results) * 0.3
        cpc_high = cpc_low * 2.5
        
        return {
            'search_volume': search_volume,
            'competition': competition,
            'cpc_low': cpc_low,
            'cpc_high': cpc_high,
            'keyword_difficulty': min(20 + len(ads_results) * 15, 100),
            'search_intent': 'commercial'
        }

class EnhancedSEMAnalyzer:
    def __init__(self, config: SEMConfig):
        self.config = config
        self.setup_gemini()
        self.serper = SerperKeywordResearch()
    
    def setup_gemini(self):
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        except:
            self.gemini_model = None
    
    def analyze_website_content(self, content: str, url: str) -> Dict:
        if not self.gemini_model:
            return self._fallback_analysis(content, url)
        
        prompt = f"""
        Analyze this website content for SEM keyword research:
        WEBSITE: {url}
        CONTENT: {content[:4000]}
        
        Return JSON:
        {{
            "business_type": "specific business category",
            "primary_services": ["service 1", "service 2", "service 3"],
            "target_audience": "target audience description",
            "seed_keywords": ["keyword 1", "keyword 2", "keyword 3", "keyword 4", "keyword 5"]
        }}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return self._fallback_analysis(content, url)
    
    def _fallback_analysis(self, content: str, url: str) -> Dict:
        return {
            "business_type": "Professional Services",
            "primary_services": ["Business Service", "Professional Solution", "Consulting"],
            "target_audience": "Business professionals and companies",
            "seed_keywords": ["professional service", "business solution", "consulting", "expert service", "business support"]
        }
    
    def generate_sem_strategy(self, brand_content: str, competitor_content: str) -> Dict:
        # Analyze content
        brand_analysis = self.analyze_website_content(brand_content, self.config.brand_website)
        competitor_analysis = self.analyze_website_content(competitor_content, self.config.competitor_website)
        
        # Get keywords
        all_keywords = brand_analysis.get('seed_keywords', []) + competitor_analysis.get('seed_keywords', [])
        keyword_data = self.serper.get_search_volume_data(list(set(all_keywords)))
        
        # Create keyword objects
        keywords = []
        for kw, data in keyword_data.items():
            keyword_obj = KeywordData(
                keyword=kw,
                search_volume=data['search_volume'],
                competition=data['competition'],
                cpc_low=data['cpc_low'],
                cpc_high=data['cpc_high'],
                keyword_difficulty=data['keyword_difficulty'],
                search_intent=data['search_intent']
            )
            # Calculate ROAS
            avg_cpc = (keyword_obj.cpc_low + keyword_obj.cpc_high) / 2
            keyword_obj.estimated_cpa = avg_cpc / self.config.conversion_rate
            keyword_obj.projected_roas = self.config.avg_order_value / keyword_obj.estimated_cpa if keyword_obj.estimated_cpa > 0 else 0
            keywords.append(keyword_obj)
        
        # Sort by ROAS
        keywords.sort(key=lambda x: x.projected_roas, reverse=True)
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "brand_analysis": brand_analysis,
            "competitor_analysis": competitor_analysis,
            "keyword_research_summary": {
                "total_keywords_researched": len(all_keywords),
                "final_optimized_keywords": len(keywords),
                "avg_search_volume": int(sum(k.search_volume for k in keywords) / len(keywords)) if keywords else 0,
                "avg_projected_roas": round(sum(k.projected_roas for k in keywords) / len(keywords), 2) if keywords else 0
            },
            "deliverables": {
                "search_campaigns": self._create_ad_groups(keywords),
                "pmax_themes": self._create_pmax_themes(brand_analysis)
            }
        }
    
    def _create_ad_groups(self, keywords: List[KeywordData]) -> List[Dict]:
        if not keywords:
            return []
        
        return [{
            "ad_group_name": "High-Intent Keywords",
            "theme": "Primary commercial keywords",
            "keywords": [
                {
                    "keyword": kw.keyword,
                    "suggested_cpc": round((kw.cpc_low + kw.cpc_high) / 2, 2),
                    "search_volume": kw.search_volume,
                    "competition": kw.competition,
                    "projected_roas": round(kw.projected_roas, 2)
                }
                for kw in keywords[:15]
            ]
        }]
    
    def _create_pmax_themes(self, analysis: Dict) -> List[Dict]:
        services = analysis.get('primary_services', ['Service'])
        return [{
            "theme_name": f"{services[0]} Solutions",
            "asset_group_focus": f"Complete {services[0].lower()} for businesses",
            "target_audience": analysis.get('target_audience', 'Business professionals')
        }]

def scrape_website_simple(url):
    """Simple web scraping for cloud deployment"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=10, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return ' '.join(chunk for chunk in chunks if chunk)[:3000]
        
        return f"Website: {url}. Professional business website."
    except Exception as e:
        st.warning(f"Could not scrape {url}: {str(e)}")
        return f"Website: {url}. Professional business website."

def main():
    st.title("🚀 Universal SEM Strategy Analyzer")
    st.markdown("Generate comprehensive Google Ads strategies using AI and real market data")
    
    # Check API keys
    if not SERPER_API_KEY or not GEMINI_API_KEY:
        st.error("⚠️ API keys not configured. This is a demo version.")
        st.info("To use this tool, you need Serper.dev and Google Gemini API keys.")
    
    # Usage limit
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
    
    st.sidebar.info(f"Free analyses: {3 - st.session_state.usage_count}/3")
    
    if st.session_state.usage_count >= 3:
        st.warning("Daily limit reached. Refresh to reset counter.")
        if st.button("Reset Counter"):
            st.session_state.usage_count = 0
            st.rerun()
        return
    
    # Input form
    with st.form("sem_analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            brand_website = st.text_input("Your Website", placeholder="https://example.com")
            search_budget = st.number_input("Search Budget ($)", value=5000, min_value=100)
        
        with col2:
            competitor_website = st.text_input("Competitor Website", placeholder="https://competitor.com")
            target_roas = st.slider("Target ROAS", 2.0, 8.0, 4.0)
        
        locations = st.text_area("Target Locations", value="United States\nCanada\nUnited Kingdom")
        
        submit = st.form_submit_button("🔍 Generate SEM Strategy", type="primary")
    
    if submit:
        if not brand_website or not competitor_website:
            st.error("Please enter both websites")
            return
        
        if not SERPER_API_KEY or not GEMINI_API_KEY:
            st.error("API keys required for analysis")
            return
        
        st.session_state.usage_count += 1
        
        with st.spinner("Analyzing websites and generating strategy..."):
            try:
                # Create config
                service_locations = [loc.strip() for loc in locations.split('\n') if loc.strip()]
                config = SEMConfig(
                    brand_website=brand_website,
                    competitor_website=competitor_website,
                    service_locations=service_locations,
                    search_budget=search_budget,
                    pmax_budget=search_budget * 0.6,
                    shopping_budget=search_budget * 0.4,
                    target_roas=target_roas
                )
                
                # Scrape and analyze
                brand_content = scrape_website_simple(brand_website)
                competitor_content = scrape_website_simple(competitor_website)
                
                analyzer = EnhancedSEMAnalyzer(config)
                results = analyzer.generate_sem_strategy(brand_content, competitor_content)
                
                # Display results
                st.success("🎉 Analysis Complete!")
                
                summary = results.get("keyword_research_summary", {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Keywords", summary.get("total_keywords_researched", 0))
                with col2:
                    st.metric("Optimized", summary.get("final_optimized_keywords", 0))
                with col3:
                    st.metric("Avg Volume", f"{summary.get('avg_search_volume', 0):,}")
                with col4:
                    st.metric("Avg ROAS", f"{summary.get('avg_projected_roas', 0):.1f}x")
                
                # Show campaigns
                campaigns = results.get("deliverables", {}).get("search_campaigns", [])
                if campaigns:
                    st.subheader("🎯 Recommended Keywords")
                    for campaign in campaigns:
                        with st.expander(f"📁 {campaign['ad_group_name']}"):
                            if campaign.get('keywords'):
                                df = pd.DataFrame(campaign['keywords'])
                                st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.usage_count -= 1

if __name__ == "__main__":
    main()
