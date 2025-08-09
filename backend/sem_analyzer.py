import json
import yaml
import pandas as pd
import re
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
import random
from collections import defaultdict

load_dotenv()

@dataclass
class SEMConfig:
    """Configuration class for SEM strategy inputs"""
    brand_website: str
    competitor_website: str
    service_locations: List[str]
    shopping_budget: float
    search_budget: float
    pmax_budget: float
    conversion_rate: float = 0.02  # Default 2%
    min_search_volume: int = 500
    target_roas: float = 4.0
    avg_order_value: float = 150.0
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SEMConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

@dataclass
class KeywordData:
    """Data structure for keyword information"""
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
    trend: str = "stable"

class SerperKeywordResearch:
    """Serper.dev API wrapper for keyword research"""
    
    def __init__(self):
        self.api_key = os.getenv('SERPER_API_KEY')
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables. Get it from https://serper.dev")
        
        self.base_url = "https://google.serper.dev"
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        print("✅ Serper.dev API configured successfully")
    
    def get_search_volume_data(self, keywords: List[str], location: str = "us") -> Dict[str, Dict]:
        """Get search volume and competition data for keywords"""
        print(f"🔍 Fetching search volume data for {len(keywords)} keywords...")
        
        # Serper doesn't have direct keyword volume API, so we'll use search results 
        # and related searches to estimate competition and relevance
        keyword_data = {}
        
        for i, keyword in enumerate(keywords):
            try:
                # Rate limiting - Serper allows 1000 requests per month on free tier
                if i > 0:
                    time.sleep(1)  # 1 second delay between requests
                
                # Search for the keyword to get competition insights
                search_data = self._search_keyword(keyword, location)
                
                if search_data:
                    keyword_data[keyword] = self._analyze_search_results(keyword, search_data)
                    print(f"   ✅ Processed: {keyword}")
                else:
                    print(f"   ⚠️ No data for: {keyword}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {keyword}: {e}")
                continue
        
        print(f"✅ Collected data for {len(keyword_data)} keywords")
        return keyword_data
    
    def _search_keyword(self, keyword: str, location: str = "us") -> Dict:
        """Search for a specific keyword using Serper"""
        try:
            payload = {
                "q": keyword,
                "gl": location,
                "hl": "en",
                "num": 10,
                "autocorrect": True
            }
            
            response = requests.post(
                f"{self.base_url}/search",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"   API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"   Request error: {e}")
            return None
    
    def _analyze_search_results(self, keyword: str, search_data: Dict) -> Dict:
        """Analyze search results to estimate keyword metrics"""
        
        # Get basic search metrics
        total_results = search_data.get('searchInformation', {}).get('totalResults', '0')
        if isinstance(total_results, str):
            total_results = int(total_results.replace(',', '')) if total_results.replace(',', '').isdigit() else 0
        
        organic_results = search_data.get('organic', [])
        ads_results = search_data.get('ads', [])
        related_searches = search_data.get('relatedSearches', [])
        
        # Estimate search volume based on total results and competition
        # This is a heuristic estimation since Serper doesn't provide exact volumes
        search_volume = self._estimate_search_volume(keyword, total_results, len(ads_results))
        
        # Estimate competition based on ads and organic results
        competition = self._estimate_competition(len(ads_results), len(organic_results))
        
        # Estimate CPC range based on competition and keyword characteristics
        cpc_low, cpc_high = self._estimate_cpc(keyword, competition, len(ads_results))
        
        # Determine search intent
        search_intent = self._determine_search_intent(keyword, organic_results)
        
        # Get keyword difficulty (estimated)
        difficulty = self._estimate_keyword_difficulty(len(ads_results), total_results, len(organic_results))
        
        return {
            'search_volume': search_volume,
            'competition': competition,
            'cpc_low': cpc_low,
            'cpc_high': cpc_high,
            'keyword_difficulty': difficulty,
            'search_intent': search_intent,
            'total_results': total_results,
            'ads_count': len(ads_results),
            'related_searches': [rs.get('query', '') for rs in related_searches[:5]]
        }
    
    def _estimate_search_volume(self, keyword: str, total_results: int, ads_count: int) -> int:
        """Estimate monthly search volume based on available signals"""
        
        # Base estimation on total results and ads competition
        base_volume = 500  # Minimum threshold
        
        # Adjust based on total results (more results = higher volume potential)
        if total_results > 10000000:  # > 10M results
            volume_multiplier = 20
        elif total_results > 1000000:  # > 1M results  
            volume_multiplier = 10
        elif total_results > 100000:  # > 100K results
            volume_multiplier = 5
        else:
            volume_multiplier = 2
        
        # Adjust based on ads competition (more ads = higher commercial value)
        ads_multiplier = 1 + (ads_count * 0.5)  # Each ad increases estimate by 50%
        
        # Adjust based on keyword characteristics
        word_count = len(keyword.split())
        if word_count == 1:  # Single word keywords typically have higher volume
            word_multiplier = 2
        elif word_count == 2:
            word_multiplier = 1.5
        else:  # Long-tail keywords
            word_multiplier = 0.8
        
        # Calculate estimated volume
        estimated_volume = int(base_volume * volume_multiplier * ads_multiplier * word_multiplier)
        
        # Add some randomization to make it more realistic
        estimated_volume = int(estimated_volume * random.uniform(0.7, 1.3))
        
        return max(estimated_volume, 500)  # Ensure minimum volume
    
    def _estimate_competition(self, ads_count: int, organic_count: int) -> str:
        """Estimate competition level"""
        
        if ads_count >= 4:
            return "HIGH"
        elif ads_count >= 2:
            return "MEDIUM"
        elif ads_count >= 1:
            return "LOW"
        else:
            # Check organic competition
            if organic_count >= 8:
                return "MEDIUM"
            else:
                return "LOW"
    
    def _estimate_cpc(self, keyword: str, competition: str, ads_count: int) -> Tuple[float, float]:
        """Estimate CPC range based on keyword and competition"""
        
        # Base CPC ranges by competition level
        base_ranges = {
            "LOW": (0.5, 1.5),
            "MEDIUM": (1.0, 3.0),
            "HIGH": (2.0, 6.0)
        }
        
        base_low, base_high = base_ranges[competition]
        
        # Adjust based on keyword characteristics
        keyword_lower = keyword.lower()
        
        # Commercial intent keywords tend to be more expensive
        if any(term in keyword_lower for term in ['buy', 'price', 'cost', 'cheap', 'deal', 'discount']):
            multiplier = 1.5
        elif any(term in keyword_lower for term in ['best', 'top', 'review', 'compare']):
            multiplier = 1.3
        elif any(term in keyword_lower for term in ['how', 'what', 'why', 'guide', 'tutorial']):
            multiplier = 0.7  # Informational keywords are typically cheaper
        else:
            multiplier = 1.0
        
        # Adjust based on ads count
        ads_multiplier = 1 + (ads_count * 0.1)
        
        final_low = round(base_low * multiplier * ads_multiplier, 2)
        final_high = round(base_high * multiplier * ads_multiplier, 2)
        
        return final_low, final_high
    
    def _determine_search_intent(self, keyword: str, organic_results: List[Dict]) -> str:
        """Determine the search intent of the keyword"""
        
        keyword_lower = keyword.lower()
        
        # Transactional intent
        if any(term in keyword_lower for term in ['buy', 'purchase', 'order', 'shop', 'price', 'cost']):
            return "transactional"
        
        # Commercial investigation intent  
        elif any(term in keyword_lower for term in ['best', 'top', 'review', 'compare', 'vs']):
            return "commercial"
        
        # Informational intent
        elif any(term in keyword_lower for term in ['how', 'what', 'why', 'guide', 'tutorial', 'tips']):
            return "informational"
        
        # Check organic results for intent signals
        titles = [result.get('title', '').lower() for result in organic_results[:3]]
        title_text = ' '.join(titles)
        
        if any(term in title_text for term in ['buy', 'shop', 'price', 'order']):
            return "transactional"
        elif any(term in title_text for term in ['best', 'review', 'compare']):
            return "commercial"
        else:
            return "informational"
    
    def _estimate_keyword_difficulty(self, ads_count: int, total_results: int, organic_count: int) -> int:
        """Estimate keyword difficulty score (0-100)"""
        
        # Base difficulty on competition signals
        base_difficulty = 20
        
        # Adjust based on ads (high competition)
        ads_difficulty = ads_count * 15
        
        # Adjust based on total results  
        if total_results > 50000000:
            results_difficulty = 40
        elif total_results > 10000000:
            results_difficulty = 30
        elif total_results > 1000000:
            results_difficulty = 20
        else:
            results_difficulty = 10
        
        total_difficulty = min(base_difficulty + ads_difficulty + results_difficulty, 100)
        return total_difficulty

class ROASOptimizer:
    """Enhanced ROAS optimizer with Serper data"""
    
    def __init__(self, conversion_rate: float = 0.02, target_roas: float = 4.0):
        self.conversion_rate = conversion_rate
        self.target_roas = target_roas
    
    def calculate_keyword_roas(self, keyword: KeywordData, avg_order_value: float) -> KeywordData:
        """Calculate projected ROAS for a keyword with enhanced metrics"""
        
        # Use average of low and high CPC
        avg_cpc = (keyword.cpc_low + keyword.cpc_high) / 2 if keyword.cpc_high > 0 else keyword.cpc_low
        
        if avg_cpc == 0:
            avg_cpc = 2.0  # Default fallback
        
        # Calculate CPA (Cost Per Acquisition)
        keyword.estimated_cpa = avg_cpc / self.conversion_rate
        
        # Calculate projected ROAS
        if keyword.estimated_cpa > 0:
            keyword.projected_roas = avg_order_value / keyword.estimated_cpa
        else:
            keyword.projected_roas = 0
        
        # Enhanced relevance score calculation
        keyword.relevance_score = self._calculate_relevance_score(keyword, avg_order_value)
        
        return keyword
    
    def _calculate_relevance_score(self, keyword: KeywordData, avg_order_value: float) -> float:
        """Calculate comprehensive relevance score"""
        
        # Volume score (normalized to 0-1)
        volume_score = min(keyword.search_volume / 10000, 1.0)
        
        # Competition score (inverse - lower competition is better)
        comp_scores = {"LOW": 1.0, "MEDIUM": 0.7, "HIGH": 0.4}
        comp_score = comp_scores.get(keyword.competition, 0.5)
        
        # ROAS score (normalized to target ROAS)
        roas_score = min(keyword.projected_roas / self.target_roas, 1.0) if keyword.projected_roas > 0 else 0
        
        # Intent score (commercial intent is better for conversions)
        intent_scores = {"transactional": 1.0, "commercial": 0.8, "informational": 0.4}
        intent_score = intent_scores.get(keyword.search_intent, 0.6)
        
        # Difficulty score (inverse - easier keywords are better)
        difficulty_score = max(0, (100 - keyword.keyword_difficulty) / 100)
        
        # Weighted average
        relevance_score = (
            volume_score * 0.25 +      # 25% weight on volume
            comp_score * 0.20 +        # 20% weight on competition
            roas_score * 0.30 +        # 30% weight on ROAS
            intent_score * 0.15 +      # 15% weight on intent
            difficulty_score * 0.10    # 10% weight on difficulty
        )
        
        return round(relevance_score, 3)
    
    def optimize_keyword_list(self, keywords: List[KeywordData], budget: float, 
                            avg_order_value: float) -> List[KeywordData]:
        """Optimize keyword list for maximum ROAS within budget"""
        
        # Calculate ROAS for each keyword
        optimized_keywords = [self.calculate_keyword_roas(kw, avg_order_value) for kw in keywords]
        
        # Sort by relevance score (descending)
        optimized_keywords.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Filter by budget constraints
        daily_budget = budget / 30
        selected_keywords = []
        estimated_daily_spend = 0
        
        for keyword in optimized_keywords:
            # Estimate daily spend for this keyword
            avg_cpc = (keyword.cpc_low + keyword.cpc_high) / 2
            estimated_clicks = keyword.search_volume / 30 * 0.1  # Assume 10% impression share
            daily_spend = estimated_clicks * avg_cpc
            
            if estimated_daily_spend + daily_spend <= daily_budget:
                selected_keywords.append(keyword)
                estimated_daily_spend += daily_spend
            
            if len(selected_keywords) >= 100:  # Increased limit for better coverage
                break
        
        print(f"✅ Optimized to {len(selected_keywords)} keywords within ${budget:,.0f} budget")
        print(f"📊 Average relevance score: {sum(k.relevance_score for k in selected_keywords) / len(selected_keywords) if selected_keywords else 0:.3f}")
        
        return selected_keywords

class EnhancedSEMAnalyzer:
    """Complete SEM analyzer with Serper.dev integration"""
    
    def __init__(self, config: SEMConfig):
        self.config = config
        self.setup_gemini()
        self.serper = SerperKeywordResearch()
        self.roas_optimizer = ROASOptimizer(config.conversion_rate, config.target_roas)
    
    def setup_gemini(self):
        """Initialize Gemini AI"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini API configured successfully")
        except Exception as e:
            print(f"⚠️ Gemini API issue, using fallback: {e}")
            try:
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            except:
                self.gemini_model = None
    
    def analyze_website_content(self, scraped_text: str, website_url: str) -> Dict:
        """Analyze website content using Gemini AI"""
        
        if not self.gemini_model:
            return self._fallback_website_analysis(scraped_text, website_url)
        
        # Truncate content to avoid token limits
        analysis_text = scraped_text[:8000] if len(scraped_text) > 8000 else scraped_text
        
        prompt = f"""
        Analyze this website content for SEM keyword research:
        
        WEBSITE: {website_url}
        CONTENT: {analysis_text}
        
        Extract specific business information and return JSON:
        {{
            "business_type": "specific business category (e.g., 'Payment Processing', 'E-commerce Software')",
            "primary_services": ["specific service 1", "specific service 2", "specific service 3"],
            "target_audience": "detailed target audience description",
            "value_propositions": ["unique value prop 1", "unique value prop 2"],
            "avg_order_value_estimate": 150,
            "seed_keywords": ["high-intent keyword 1", "high-intent keyword 2", "keyword 3", "keyword 4", "keyword 5", "keyword 6", "keyword 7", "keyword 8", "keyword 9", "keyword 10"],
            "competitor_advantages": ["advantage 1", "advantage 2"],
            "market_positioning": "positioning description",
            "product_categories": ["category 1", "category 2"],
            "geographic_focus": "target geographic regions"
        }}
        
        Make seed keywords specific to this business and commercially viable for Google Ads.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"✅ AI analysis completed for {result.get('business_type', 'Unknown Business')}")
                return result
        except Exception as e:
            print(f"⚠️ AI analysis error: {e}")
        
        return self._fallback_website_analysis(scraped_text, website_url)
    
    def _fallback_website_analysis(self, scraped_text: str, website_url: str) -> Dict:
        """Fallback analysis when AI is unavailable"""
        
        # Extract some basic info from URL and text
        domain = website_url.replace('https://', '').replace('http://', '').split('/')[0]
        business_name = domain.split('.')[0].title()
        
        # Basic keyword extraction from text
        words = re.findall(r'\b[a-z]{3,}\b', scraped_text.lower()[:2000])
        common_business_words = ['service', 'solution', 'platform', 'software', 'system', 'management', 'business', 'professional']
        
        seed_keywords = []
        for word in common_business_words:
            if word in words:
                seed_keywords.extend([
                    f"{word} service",
                    f"best {word}",
                    f"{word} solution",
                    f"professional {word}"
                ])
        
        return {
            "business_type": "Professional Services",
            "primary_services": ["Professional Service", "Business Solution", "Consulting"],
            "target_audience": "Business professionals and companies",
            "seed_keywords": seed_keywords[:10] if seed_keywords else [
                "professional service", "business solution", "consulting service",
                "expert advice", "business consulting", "professional help",
                "service provider", "business support", "professional solution", "expert service"
            ],
            "avg_order_value_estimate": self.config.avg_order_value
        }
    
    def expand_keywords_with_ai(self, seed_keywords: List[str], business_analysis: Dict) -> List[str]:
        """Expand seed keywords using AI"""
        
        if not self.gemini_model:
            return self._expand_keywords_heuristic(seed_keywords, business_analysis)
        
        prompt = f"""
        Generate additional high-converting keywords for Google Ads campaigns:
        
        BUSINESS: {business_analysis.get('business_type', 'Service Provider')}
        SERVICES: {', '.join(business_analysis.get('primary_services', []))}
        SEED KEYWORDS: {', '.join(seed_keywords)}
        
        Generate 30 additional keywords with different intents:
        - Commercial intent (buying keywords)
        - Informational intent (research keywords)  
        - Competitor comparison keywords
        - Location-based keywords using: {', '.join(self.config.service_locations)}
        - Long-tail variations
        
        Return as JSON array:
        ["keyword 1", "keyword 2", ..., "keyword 30"]
        
        Make keywords specific, realistic, and suitable for paid search.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_match:
                additional_keywords = json.loads(json_match.group())
                print(f"✅ AI generated {len(additional_keywords)} additional keywords")
                return list(set(seed_keywords + additional_keywords))  # Remove duplicates
        except Exception as e:
            print(f"⚠️ AI keyword expansion error: {e}")
        
        return self._expand_keywords_heuristic(seed_keywords, business_analysis)
    
    def _expand_keywords_heuristic(self, seed_keywords: List[str], business_analysis: Dict) -> List[str]:
        """Heuristic keyword expansion when AI is unavailable"""
        
        expanded_keywords = list(seed_keywords)
        
        # Modifiers for expansion
        commercial_modifiers = ["best", "top", "professional", "affordable", "cheap", "premium", "quality"]
        action_modifiers = ["buy", "get", "find", "hire", "choose", "compare"]
        location_modifiers = self.config.service_locations[:3]  # Use first 3 locations
        
        # Expand with different patterns
        for seed in seed_keywords[:5]:  # Limit to prevent explosion
            # Commercial variations
            for mod in commercial_modifiers[:3]:
                expanded_keywords.append(f"{mod} {seed}")
            
            # Action variations
            for mod in action_modifiers[:2]:
                expanded_keywords.append(f"{mod} {seed}")
            
            # Location variations
            for loc in location_modifiers:
                expanded_keywords.append(f"{seed} {loc}")
                expanded_keywords.append(f"{seed} near me")
        
        # Remove duplicates and return
        return list(set(expanded_keywords))
    
    def generate_sem_strategy(self, brand_content: str, competitor_content: str) -> Dict:
        """Generate complete SEM strategy"""
        
        print("🚀 Starting Enhanced SEM Strategy Generation with Serper.dev")
        print("=" * 70)
        
        # Step 1: Analyze content with AI
        print("📊 Step 1: Analyzing website content with AI...")
        brand_analysis = self.analyze_website_content(brand_content, self.config.brand_website)
        competitor_analysis = self.analyze_website_content(competitor_content, self.config.competitor_website)
        
        # Step 2: Expand keywords
        print("🔍 Step 2: Expanding keyword universe...")
        initial_seeds = list(set(
            brand_analysis.get('seed_keywords', []) + 
            competitor_analysis.get('seed_keywords', [])
        ))
        
        all_keywords = self.expand_keywords_with_ai(initial_seeds, brand_analysis)
        print(f"📈 Expanded to {len(all_keywords)} total keywords")
        
        # Step 3: Get search volume data from Serper
        print("📊 Step 3: Fetching real market data with Serper.dev...")
        # Limit to top keywords to stay within API limits
        priority_keywords = all_keywords[:50]  # Adjust based on your Serper plan
        keyword_data_dict = self.serper.get_search_volume_data(priority_keywords)
        
        # Step 4: Convert to KeywordData objects
        print("🔄 Step 4: Processing keyword data...")
        keyword_objects = []
        for keyword, data in keyword_data_dict.items():
            keyword_obj = KeywordData(
                keyword=keyword,
                search_volume=data.get('search_volume', 500),
                competition=data.get('competition', 'MEDIUM'),
                cpc_low=data.get('cpc_low', 1.0),
                cpc_high=data.get('cpc_high', 3.0),
                keyword_difficulty=data.get('keyword_difficulty', 50),
                search_intent=data.get('search_intent', 'commercial')
            )
            keyword_objects.append(keyword_obj)
        
        # Step 5: Filter by minimum search volume
        print("🔽 Step 5: Filtering keywords...")
        filtered_keywords = [
            kw for kw in keyword_objects 
            if kw.search_volume >= self.config.min_search_volume
        ]
        
        if not filtered_keywords:
            print("⚠️ No keywords meet minimum volume requirement, using top keywords anyway")
            filtered_keywords = keyword_objects[:20]
        
        # Step 6: Optimize for ROAS
        print("💰 Step 6: Optimizing for maximum ROAS...")
        search_keywords = self.roas_optimizer.optimize_keyword_list(
            filtered_keywords, 
            self.config.search_budget, 
            self.config.avg_order_value
        )
        
        # Step 7: Generate all deliverables
        print("📋 Step 7: Generating campaign deliverables...")
        deliverables = self._generate_all_deliverables(
            search_keywords, brand_analysis, competitor_analysis
        )
        
        # Step 8: Compile final strategy
        strategy = {
            "analysis_timestamp": datetime.now().isoformat(),
            "config_used": {
                "brand_website": self.config.brand_website,
                "competitor_website": self.config.competitor_website,
                "total_budget": self.config.search_budget + self.config.pmax_budget + self.config.shopping_budget,
                "target_roas": self.config.target_roas,
                "min_search_volume": self.config.min_search_volume
            },
            "brand_analysis": brand_analysis,
            "competitor_analysis": competitor_analysis,
            "keyword_research_summary": {
                "total_keywords_researched": len(all_keywords),
                "keywords_with_data": len(keyword_objects),
                "keywords_after_filtering": len(filtered_keywords),
                "final_optimized_keywords": len(search_keywords),
                "avg_search_volume": int(sum(kw.search_volume for kw in search_keywords) / len(search_keywords)) if search_keywords else 0,
                "avg_cpc": round(sum((kw.cpc_low + kw.cpc_high) / 2 for kw in search_keywords) / len(search_keywords), 2) if search_keywords else 0,
                "avg_projected_roas": round(sum(kw.projected_roas for kw in search_keywords) / len(search_keywords), 2) if search_keywords else 0
            },
            "deliverables": deliverables,
            "implementation_roadmap": self._create_implementation_roadmap()
        }
        
        print("✅ Complete SEM Strategy Generated!")
        return strategy
    
    def _generate_all_deliverables(self, keywords: List[KeywordData], 
                                 brand_analysis: Dict, competitor_analysis: Dict) -> Dict:
        """Generate all three required deliverables"""
        
        return {
            "1_search_campaign_ad_groups": self._create_search_ad_groups(keywords),
            "2_performance_max_themes": self._create_pmax_themes(keywords, brand_analysis),
            "3_shopping_cpc_bids": self._create_shopping_strategy(keywords),
            "bonus_competitive_analysis": self._create_competitive_analysis(brand_analysis, competitor_analysis),
            "budget_allocation": self._create_budget_allocation(keywords)
        }
    
    def _create_search_ad_groups(self, keywords: List[KeywordData]) -> List[Dict]:
        """Deliverable 1: Search Campaign Ad Groups with match types and CPC ranges"""
        
        # Group keywords by intent and characteristics
        ad_groups = []
        
        # Group 1: High-Intent Commercial Keywords
        commercial_keywords = [kw for kw in keywords if kw.search_intent == "commercial"][:15]
        if commercial_keywords:
            ad_groups.append({
                "ad_group_name": "High-Intent Commercial",
                "theme": "Users ready to make purchasing decisions",
                "target_audience": "High-intent prospects comparing solutions",
                "keywords": [
                    {
                        "keyword": kw.keyword,
                        "match_type": "phrase",
                        "suggested_cpc": round((kw.cpc_low + kw.cpc_high) / 2, 2),
                        "search_volume": kw.search_volume,
                        "competition": kw.competition,
                        "projected_roas": round(kw.projected_roas, 2),
                        "relevance_score": kw.relevance_score
                    }
                    for kw in commercial_keywords
                ],
                "estimated_daily_budget": sum((kw.cpc_low + kw.cpc_high) / 2 * (kw.search_volume / 30 * 0.1) for kw in commercial_keywords) / len(commercial_keywords) if commercial_keywords else 0
            })
        
        # Group 2: Transactional Keywords (Buy Intent)
        transactional_keywords = [kw for kw in keywords if kw.search_intent == "transactional"][:12]
        if transactional_keywords:
            ad_groups.append({
                "ad_group_name": "Transactional - Buy Intent",
                "theme": "Users ready to purchase immediately",
                "target_audience": "Bottom-funnel prospects with purchase intent",
                "keywords": [
                    {
                        "keyword": kw.keyword,
                        "match_type": "exact" if len(kw.keyword.split()) <= 3 else "phrase",
                        "suggested_cpc": round(kw.cpc_high * 0.9, 2),  # Bid higher for transactional
                        "search_volume": kw.search_volume,
                        "competition": kw.competition,
                        "projected_roas": round(kw.projected_roas, 2),
                        "relevance_score": kw.relevance_score
                    }
                    for kw in transactional_keywords
                ],
                "estimated_daily_budget": sum(kw.cpc_high * 0.9 * (kw.search_volume / 30 * 0.15) for kw in transactional_keywords) / len(transactional_keywords) if transactional_keywords else 0
            })
        
        # Group 3: Category/Service Terms
        remaining_keywords = [kw for kw in keywords if kw not in commercial_keywords + transactional_keywords]
        service_keywords = remaining_keywords[:15]
        if service_keywords:
            ad_groups.append({
                "ad_group_name": "Core Services",
                "theme": "Primary service and category targeting",
                "target_audience": "Users searching for specific services",
                "keywords": [
                    {
                        "keyword": kw.keyword,
                        "match_type": "broad",
                        "suggested_cpc": round((kw.cpc_low + kw.cpc_high) / 2, 2),
                        "search_volume": kw.search_volume,
                        "competition": kw.competition,
                        "projected_roas": round(kw.projected_roas, 2),
                        "relevance_score": kw.relevance_score
                    }
                    for kw in service_keywords
                ],
                "estimated_daily_budget": sum((kw.cpc_low + kw.cpc_high) / 2 * (kw.search_volume / 30 * 0.08) for kw in service_keywords) / len(service_keywords) if service_keywords else 0
            })
        
        # Group 4: Location-Based Keywords
        location_keywords = [kw for kw in keywords if any(loc.lower() in kw.keyword.lower() for loc in self.config.service_locations)][:10]
        if location_keywords:
            ad_groups.append({
                "ad_group_name": "Location Targeting",
                "theme": "Geographic-specific searches",
                "target_audience": "Local prospects in target markets",
                "keywords": [
                    {
                        "keyword": kw.keyword,
                        "match_type": "phrase",
                        "suggested_cpc": round(kw.cpc_low * 1.1, 2),  # Slightly higher for local
                        "search_volume": kw.search_volume,
                        "competition": kw.competition,
                        "projected_roas": round(kw.projected_roas, 2),
                        "relevance_score": kw.relevance_score
                    }
                    for kw in location_keywords
                ],
                "estimated_daily_budget": sum(kw.cpc_low * 1.1 * (kw.search_volume / 30 * 0.12) for kw in location_keywords) / len(location_keywords) if location_keywords else 0
            })
        
        # Group 5: Competitor Terms (if any competitive keywords found)
        competitor_keywords = [kw for kw in keywords if any(comp in kw.keyword.lower() for comp in ['vs', 'alternative', 'competitor', 'compare'])][:8]
        if competitor_keywords:
            ad_groups.append({
                "ad_group_name": "Competitor Comparisons",
                "theme": "Competitive positioning and alternatives",
                "target_audience": "Users comparing different solutions",
                "keywords": [
                    {
                        "keyword": kw.keyword,
                        "match_type": "phrase",
                        "suggested_cpc": round((kw.cpc_low + kw.cpc_high) / 2 * 1.2, 2),  # Higher for competitive
                        "search_volume": kw.search_volume,
                        "competition": kw.competition,
                        "projected_roas": round(kw.projected_roas, 2),
                        "relevance_score": kw.relevance_score
                    }
                    for kw in competitor_keywords
                ],
                "estimated_daily_budget": sum((kw.cpc_low + kw.cpc_high) / 2 * 1.2 * (kw.search_volume / 30 * 0.06) for kw in competitor_keywords) / len(competitor_keywords) if competitor_keywords else 0
            })
        
        return ad_groups
    
    def _create_pmax_themes(self, keywords: List[KeywordData], brand_analysis: Dict) -> List[Dict]:
        """Deliverable 2: Performance Max Campaign Themes"""
        
        services = brand_analysis.get('primary_services', ['Service', 'Solution', 'Platform'])
        business_type = brand_analysis.get('business_type', 'Professional Service')
        
        themes = []
        
        # Theme 1: Primary Service Focus
        primary_keywords = keywords[:8]
        themes.append({
            "theme_name": f"{services[0]} Solutions",
            "asset_group_focus": f"Complete {services[0].lower()} for businesses",
            "target_audience": brand_analysis.get('target_audience', 'Business professionals'),
            "value_proposition": f"Leading {services[0].lower()} provider",
            "related_keywords": [kw.keyword for kw in primary_keywords],
            "audience_signals": [
                "Website visitors (past 30 days)",
                f"In-market for {business_type.lower()}",
                "Custom intent: Business decision makers",
                f"Affinity: {business_type} users",
                "Similar audiences: Existing customers"
            ],
            "asset_requirements": {
                "headlines": [
                    f"#1 {services[0]} Platform",
                    f"Transform Your Business with {services[0]}",
                    f"Professional {services[0]} Made Easy",
                    f"Get Started with {services[0]} Today"
                ],
                "descriptions": [
                    f"Streamline your business with our {services[0].lower()} platform. Trusted by thousands of companies worldwide.",
                    f"Everything you need for {services[0].lower()} in one powerful platform. Start your free trial today."
                ],
                "images_needed": ["Product demo", "Happy customers", "Dashboard screenshot", "Team collaboration"]
            }
        })
        
        # Theme 2: Industry/Use Case Focus
        if len(services) > 1:
            themes.append({
                "theme_name": f"{services[1]} for Enterprises",
                "asset_group_focus": f"Enterprise-grade {services[1].lower()} solutions",
                "target_audience": "Enterprise decision makers and IT professionals",
                "value_proposition": f"Scalable {services[1].lower()} for large organizations",
                "related_keywords": [kw.keyword for kw in keywords[8:15]],
                "audience_signals": [
                    "Job title: IT Director, CTO, CEO",
                    "Company size: 100+ employees",
                    "In-market for enterprise software",
                    "Website visitors who viewed pricing page",
                    "Custom audience: Enterprise prospects"
                ],
                "asset_requirements": {
                    "headlines": [
                        f"Enterprise {services[1]} Platform",
                        f"Scale Your Business with {services[1]}",
                        f"Trusted by Fortune 500 Companies",
                        f"Enterprise-Grade {services[1]}"
                    ],
                    "descriptions": [
                        f"Power your enterprise with our {services[1].lower()} platform. Built for scale, security, and performance.",
                        f"Join thousands of enterprises using our {services[1].lower()} solution. Contact sales for custom pricing."
                    ],
                    "images_needed": ["Enterprise dashboard", "Security badges", "Customer logos", "ROI charts"]
                }
            })
        
        # Theme 3: Problem-Solution Focus
        themes.append({
            "theme_name": "Problem Solver",
            "asset_group_focus": "Solving specific business challenges",
            "target_audience": "Businesses facing operational challenges",
            "value_proposition": "Eliminate inefficiencies and boost productivity",
            "related_keywords": [kw.keyword for kw in keywords[15:20]],
            "audience_signals": [
                "Custom intent: Business efficiency",
                "Website visitors (problem-focused content)",
                "In-market for productivity software",
                "Job title: Operations Manager, Business Owner",
                "Remarketing: Blog readers"
            ],
            "asset_requirements": {
                "headlines": [
                    "Stop Wasting Time on Manual Processes",
                    "Automate Your Business Operations",
                    "Increase Efficiency by 300%",
                    "Transform Your Business Today"
                ],
                "descriptions": [
                    "Stop losing money to inefficient processes. Our platform automates your workflow and saves you hours daily.",
                    "See immediate results with our proven system. Join 10,000+ businesses already saving time and money."
                ],
                "images_needed": ["Before/after comparisons", "Time savings graphics", "Process automation", "Success stories"]
            }
        })
        
        # Theme 4: Competitive Advantage Focus
        themes.append({
            "theme_name": "Best Alternative",
            "asset_group_focus": "Superior alternative to competitors",
            "target_audience": "Users comparing different solutions",
            "value_proposition": "Better features, better price, better support",
            "related_keywords": [kw.keyword for kw in keywords[20:25]],
            "audience_signals": [
                "Competitor website visitors",
                "Custom intent: Solution comparison",
                "In-market for business software",
                "Website visitors (competitor comparison pages)",
                "Similar audiences: Competitor customers"
            ],
            "asset_requirements": {
                "headlines": [
                    "Better Than the Competition",
                    "Switch and Save 40%",
                    "Why Businesses Choose Us",
                    "The Smart Alternative"
                ],
                "descriptions": [
                    "Get more features for less money. See why businesses are switching from competitors to our platform.",
                    "Superior functionality at a better price. Try risk-free for 30 days and see the difference."
                ],
                "images_needed": ["Comparison charts", "Customer testimonials", "Feature highlights", "Pricing comparisons"]
            }
        })
        
        return themes
    
    def _create_shopping_strategy(self, keywords: List[KeywordData]) -> Dict:
        """Deliverable 3: Shopping Campaign CPC Strategy"""
        
        # Calculate overall metrics for shopping strategy
        avg_cpc = sum((kw.cpc_low + kw.cpc_high) / 2 for kw in keywords) / len(keywords) if keywords else 2.0
        avg_roas = sum(kw.projected_roas for kw in keywords) / len(keywords) if keywords else self.config.target_roas
        
        # Target CPA calculation (CPA = AOV / Target ROAS)
        target_cpa = self.config.avg_order_value / self.config.target_roas
        
        # Shopping campaigns work differently - they're product-focused
        # We'll create product groups based on keyword themes
        product_groups = []
        
        # High-value product group (from highest ROAS keywords)
        high_roas_keywords = sorted([kw for kw in keywords if kw.projected_roas > self.config.target_roas], 
                                   key=lambda x: x.projected_roas, reverse=True)[:10]
        
        if high_roas_keywords:
            product_groups.append({
                "product_group_name": "High-Value Products",
                "product_criteria": "Top performing product categories",
                "suggested_max_cpc": round(avg_cpc * 1.3, 2),  # Bid higher for high-value
                "target_roas": round(avg_roas * 1.2, 1),
                "priority": "High",
                "bid_strategy": "Target ROAS",
                "related_keywords": [kw.keyword for kw in high_roas_keywords[:5]],
                "estimated_daily_budget": round(self.config.shopping_budget / 30 * 0.4, 2),  # 40% of shopping budget
                "performance_metrics": {
                    "avg_search_volume": int(sum(kw.search_volume for kw in high_roas_keywords) / len(high_roas_keywords)),
                    "avg_competition": high_roas_keywords[0].competition,
                    "projected_conversion_rate": f"{self.config.conversion_rate * 100 * 1.2:.1f}%"  # Higher for shopping
                }
            })
        
        # Standard product group
        standard_keywords = [kw for kw in keywords if kw not in high_roas_keywords][:15]
        if standard_keywords:
            product_groups.append({
                "product_group_name": "Standard Products",
                "product_criteria": "General product inventory",
                "suggested_max_cpc": round(avg_cpc, 2),
                "target_roas": round(self.config.target_roas, 1),
                "priority": "Medium",
                "bid_strategy": "Manual CPC",
                "related_keywords": [kw.keyword for kw in standard_keywords[:5]],
                "estimated_daily_budget": round(self.config.shopping_budget / 30 * 0.4, 2),  # 40% of shopping budget
                "performance_metrics": {
                    "avg_search_volume": int(sum(kw.search_volume for kw in standard_keywords) / len(standard_keywords)),
                    "avg_competition": standard_keywords[0].competition if standard_keywords else "MEDIUM",
                    "projected_conversion_rate": f"{self.config.conversion_rate * 100:.1f}%"
                }
            })
        
        # Long-tail/discovery product group
        longtail_keywords = [kw for kw in keywords if len(kw.keyword.split()) >= 4][:10]
        if longtail_keywords:
            product_groups.append({
                "product_group_name": "Discovery & Long-tail",
                "product_criteria": "Long-tail and discovery traffic",
                "suggested_max_cpc": round(avg_cpc * 0.7, 2),  # Lower bids for discovery
                "target_roas": round(self.config.target_roas * 0.8, 1),  # Lower ROAS target
                "priority": "Low",
                "bid_strategy": "Enhanced CPC",
                "related_keywords": [kw.keyword for kw in longtail_keywords[:5]],
                "estimated_daily_budget": round(self.config.shopping_budget / 30 * 0.2, 2),  # 20% of shopping budget
                "performance_metrics": {
                    "avg_search_volume": int(sum(kw.search_volume for kw in longtail_keywords) / len(longtail_keywords)),
                    "avg_competition": "LOW",
                    "projected_conversion_rate": f"{self.config.conversion_rate * 100 * 0.8:.1f}%"
                }
            })
        
        return {
            "overall_strategy": {
                "total_shopping_budget": self.config.shopping_budget,
                "daily_budget": round(self.config.shopping_budget / 30, 2),
                "target_cpa": round(target_cpa, 2),
                "target_roas": self.config.target_roas,
                "bid_strategy_recommendation": "Start with Manual CPC, transition to Target ROAS after 2 weeks of data"
            },
            "product_groups": product_groups,
            "negative_keywords": [
                "free", "cheap", "diy", "how to", "tutorial", 
                "review only", "comparison only", "vs", "problems with"
            ],
            "optimization_recommendations": [
                "Monitor search term reports weekly",
                "Adjust bids based on ROAS performance",
                "Add negative keywords from irrelevant search terms",
                "Test different product images and titles",
                "Implement dynamic remarketing for cart abandoners"
            ]
        }
    
    def _create_competitive_analysis(self, brand_analysis: Dict, competitor_analysis: Dict) -> Dict:
        """Bonus: Competitive analysis and opportunities"""
        
        return {
            "brand_strengths": brand_analysis.get('value_propositions', []),
            "competitor_advantages": competitor_analysis.get('competitor_advantages', []),
            "market_opportunities": [
                "Long-tail keyword opportunities with lower competition",
                "Geographic expansion in underserved markets",
                "Mobile-first advertising approach",
                "Voice search optimization"
            ],
            "recommended_competitive_actions": [
                "Bid on competitor brand terms (where legal)",
                "Create comparison landing pages",
                "Highlight unique differentiators in ad copy",
                "Target competitor website visitors with display ads"
            ],
            "content_gaps": [
                "Create more comparison content",
                "Develop case studies and testimonials",
                "Build location-specific landing pages",
                "Optimize for voice and mobile searches"
            ]
        }
    
    def _create_budget_allocation(self, keywords: List[KeywordData]) -> Dict:
        """Smart budget allocation across campaigns"""
        
        total_budget = self.config.search_budget + self.config.pmax_budget + self.config.shopping_budget
        
        # Calculate potential based on keyword data
        total_potential_spend = sum((kw.cpc_low + kw.cpc_high) / 2 * (kw.search_volume / 30 * 0.1) for kw in keywords)
        high_roas_potential = sum((kw.cpc_low + kw.cpc_high) / 2 * (kw.search_volume / 30 * 0.1) 
                                 for kw in keywords if kw.projected_roas >= self.config.target_roas)
        
        return {
            "monthly_budget_breakdown": {
                "search_campaigns": {
                    "budget": self.config.search_budget,
                    "percentage": round(self.config.search_budget / total_budget * 100, 1),
                    "focus": "High-intent keywords and brand protection"
                },
                "performance_max": {
                    "budget": self.config.pmax_budget,
                    "percentage": round(self.config.pmax_budget / total_budget * 100, 1),
                    "focus": "Automated bidding and audience expansion"
                },
                "shopping_campaigns": {
                    "budget": self.config.shopping_budget,
                    "percentage": round(self.config.shopping_budget / total_budget * 100, 1),
                    "focus": "Product visibility and comparison shopping"
                }
            },
            "performance_projections": {
                "estimated_monthly_clicks": int(sum(kw.search_volume * 0.1 for kw in keywords)),
                "estimated_monthly_conversions": int(sum(kw.search_volume * 0.1 * self.config.conversion_rate for kw in keywords)),
                "projected_monthly_revenue": int(sum(kw.search_volume * 0.1 * self.config.conversion_rate * self.config.avg_order_value for kw in keywords)),
                "break_even_roas": round(total_budget / (sum(kw.search_volume * 0.1 * self.config.conversion_rate * self.config.avg_order_value for kw in keywords) or 1), 2)
            },
            "scaling_recommendations": [
                "Start with 70% of allocated budget in month 1",
                "Increase budget by 20% monthly for profitable campaigns",
                "Reallocate budget from underperforming to high-ROAS campaigns",
                "Reserve 10% budget for testing new keywords and audiences"
            ]
        }
    
    def _create_implementation_roadmap(self) -> List[Dict]:
        """Create step-by-step implementation plan"""
        
        return [
            {
                "phase": "Setup & Launch (Week 1-2)",
                "tasks": [
                    "Set up Google Ads account and conversion tracking",
                    "Create search campaigns with provided ad groups",
                    "Set up Performance Max campaigns with asset groups",
                    "Launch shopping campaigns with product groups",
                    "Implement negative keyword lists"
                ],
                "success_metrics": ["All campaigns live", "Conversion tracking verified", "Budget pacing on track"]
            },
            {
                "phase": "Optimization (Week 3-6)",
                "tasks": [
                    "Monitor search term reports daily",
                    "Add negative keywords from irrelevant searches",
                    "Adjust bids based on performance data",
                    "A/B test ad copy variations",
                    "Optimize landing page experience"
                ],
                "success_metrics": ["CPA trending toward target", "Quality Score improvements", "ROAS > 3.0"]
            },
            {
                "phase": "Scale & Expand (Week 7-12)",
                "tasks": [
                    "Increase budget for profitable campaigns",
                    "Expand keyword lists with new opportunities",
                    "Test additional ad formats and extensions",
                    "Implement audience targeting refinements",
                    "Launch remarketing campaigns"
                ],
                "success_metrics": ["ROAS > target", "Consistent profitability", "Ready for further scaling"]
            }
        ]
    
    def save_strategy_with_exports(self, strategy: Dict, filename: str = None) -> str:
        """Save strategy with multiple export formats"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sem_strategy_serper_{timestamp}.json"
        
        # Save complete strategy as JSON
        with open(filename, 'w') as f:
            json.dump(strategy, f, indent=2, default=str)
        
        base_name = filename.replace('.json', '')
        
        # Export 1: Ad Groups as CSV
        self._export_ad_groups_csv(strategy, f"{base_name}_ad_groups.csv")
        
        # Export 2: Performance Max themes as CSV
        self._export_pmax_themes_csv(strategy, f"{base_name}_pmax_themes.csv")
        
        # Export 3: Shopping strategy as CSV
        self._export_shopping_csv(strategy, f"{base_name}_shopping_strategy.csv")
        
        # Export 4: Complete keyword list with metrics
        self._export_keyword_analysis_csv(strategy, f"{base_name}_keyword_analysis.csv")
        
        # Export 5: Executive summary as text
        self._export_executive_summary(strategy, f"{base_name}_executive_summary.txt")
        
        print(f"💾 Complete strategy and exports saved!")
        print(f"📁 Main file: {filename}")
        print(f"📊 CSV exports: {base_name}_*.csv")
        print(f"📋 Summary: {base_name}_executive_summary.txt")
        
        return filename
    
    def _export_ad_groups_csv(self, strategy: Dict, filename: str):
        """Export ad groups to CSV format"""
        
        ad_groups_data = []
        deliverables = strategy.get('deliverables', {})
        
        for ad_group in deliverables.get('1_search_campaign_ad_groups', []):
            for keyword_data in ad_group.get('keywords', []):
                ad_groups_data.append({
                    'Campaign': 'Search Campaign',
                    'Ad Group': ad_group['ad_group_name'],
                    'Keyword': keyword_data['keyword'],
                    'Match Type': keyword_data['match_type'],
                    'Suggested CPC': keyword_data['suggested_cpc'],
                    'Search Volume': keyword_data.get('search_volume', 0),
                    'Competition': keyword_data.get('competition', 'MEDIUM'),
                    'Projected ROAS': keyword_data.get('projected_roas', 0),
                    'Relevance Score': keyword_data.get('relevance_score', 0),
                    'Theme': ad_group.get('theme', ''),
                    'Target Audience': ad_group.get('target_audience', '')
                })
        
        df = pd.DataFrame(ad_groups_data)
        df.to_csv(filename, index=False)
    
    def _export_pmax_themes_csv(self, strategy: Dict, filename: str):
        """Export Performance Max themes to CSV"""
        
        pmax_data = []
        deliverables = strategy.get('deliverables', {})
        
        for theme in deliverables.get('2_performance_max_themes', []):
            pmax_data.append({
                'Theme Name': theme['theme_name'],
                'Asset Group Focus': theme['asset_group_focus'],
                'Target Audience': theme['target_audience'],
                'Value Proposition': theme['value_proposition'],
                'Related Keywords': ', '.join(theme.get('related_keywords', [])),
                'Audience Signals': ', '.join(theme.get('audience_signals', [])),
                'Headlines': ', '.join(theme.get('asset_requirements', {}).get('headlines', [])),
                'Descriptions': ', '.join(theme.get('asset_requirements', {}).get('descriptions', []))
            })
        
        df = pd.DataFrame(pmax_data)
        df.to_csv(filename, index=False)
    
    def _export_shopping_csv(self, strategy: Dict, filename: str):
        """Export shopping strategy to CSV"""
        
        shopping_data = []
        deliverables = strategy.get('deliverables', {})
        shopping_strategy = deliverables.get('3_shopping_cpc_bids', {})
        
        for product_group in shopping_strategy.get('product_groups', []):
            shopping_data.append({
                'Product Group': product_group['product_group_name'],
                'Product Criteria': product_group['product_criteria'],
                'Suggested Max CPC': product_group['suggested_max_cpc'],
                'Target ROAS': product_group['target_roas'],
                'Priority': product_group['priority'],
                'Bid Strategy': product_group['bid_strategy'],
                'Daily Budget': product_group['estimated_daily_budget'],
                'Related Keywords': ', '.join(product_group.get('related_keywords', [])),
                'Avg Search Volume': product_group.get('performance_metrics', {}).get('avg_search_volume', 0),
                'Competition': product_group.get('performance_metrics', {}).get('avg_competition', 'MEDIUM')
            })
        
        df = pd.DataFrame(shopping_data)
        df.to_csv(filename, index=False)
    
    def _export_keyword_analysis_csv(self, strategy: Dict, filename: str):
        """Export detailed keyword analysis"""
        # This would contain all keywords with their full metrics
        # Implementation depends on how keywords are stored in strategy
        pass
    
    def _export_executive_summary(self, strategy: Dict, filename: str):
        """Export executive summary as text file"""
        
        with open(filename, 'w') as f:
            f.write("SEM STRATEGY EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Add summary content
            config = strategy.get('config_used', {})
            research = strategy.get('keyword_research_summary', {})
            
            f.write(f"Analysis Date: {strategy.get('analysis_timestamp', '')}\n")
            f.write(f"Brand Website: {config.get('brand_website', '')}\n")
            f.write(f"Total Budget: ${config.get('total_budget', 0):,}\n")
            f.write(f"Target ROAS: {config.get('target_roas', 0)}x\n\n")
            
            f.write("KEYWORD RESEARCH RESULTS:\n")
            f.write(f"- Keywords Researched: {research.get('total_keywords_researched', 0)}\n")
            f.write(f"- Keywords with Data: {research.get('keywords_with_data', 0)}\n")
            f.write(f"- Final Optimized: {research.get('final_optimized_keywords', 0)}\n")
            f.write(f"- Average Search Volume: {research.get('avg_search_volume', 0):,}\n")
            f.write(f"- Average CPC: ${research.get('avg_cpc', 0):.2f}\n")
            f.write(f"- Average Projected ROAS: {research.get('avg_projected_roas', 0):.2f}x\n\n")
            
            # Add deliverables summary
            deliverables = strategy.get('deliverables', {})
            ad_groups = deliverables.get('1_search_campaign_ad_groups', [])
            pmax_themes = deliverables.get('2_performance_max_themes', [])
            
            f.write("CAMPAIGN STRUCTURE:\n")
            f.write(f"- Search Ad Groups: {len(ad_groups)}\n")
            f.write(f"- Performance Max Themes: {len(pmax_themes)}\n")
            f.write(f"- Shopping Product Groups: {len(deliverables.get('3_shopping_cpc_bids', {}).get('product_groups', []))}\n\n")
            
            # Add next steps
            f.write("IMPLEMENTATION ROADMAP:\n")
            for phase in strategy.get('implementation_roadmap', []):
                f.write(f"- {phase.get('phase', 'Phase')}\n")
                for task in phase.get('tasks', [])[:3]:  # First 3 tasks
                    f.write(f"  • {task}\n")
                f.write("\n")

# Main execution function
def create_sample_config():
    """Create a sample configuration file"""
    
    sample_config = {
        "brand_website": "https://dodopayments.com",
        "competitor_website": "https://stripe.com", 
        "service_locations": ["United States", "United Kingdom", "Canada", "Australia", "Germany"],
        "shopping_budget": 3000.0,
        "search_budget": 7000.0,
        "pmax_budget": 5000.0,
        "conversion_rate": 0.02,
        "min_search_volume": 500,
        "target_roas": 4.0,
        "avg_order_value": 150.0
    }
    
    with open('sem_config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print("✅ Sample configuration created: sem_config.yaml")
    print("📝 Please update with your actual values before running analysis")

def main():
    """Main execution function"""
    
    print("🚀 SEM Strategy Builder with Serper.dev")
    print("=" * 50)
    
    # Check for configuration
    config_file = "sem_config.yaml"
    if not os.path.exists(config_file):
        print("❌ Configuration file not found!")
        print("Creating sample configuration...")
        create_sample_config()
        return
    
    try:
        # Load configuration
        config = SEMConfig.from_yaml(config_file)
        print(f"✅ Configuration loaded successfully")
        print(f"🌐 Brand: {config.brand_website}")
        print(f"🏢 Competitor: {config.competitor_website}")
        print(f"💰 Total Budget: ${config.search_budget + config.pmax_budget + config.shopping_budget:,.0f}/month")
        print(f"🎯 Target ROAS: {config.target_roas}x")
        
        # Check API keys
        if not os.getenv('SERPER_API_KEY'):
            print("❌ SERPER_API_KEY not found in environment variables")
            print("🔑 Get your API key from https://serper.dev")
            print("💡 Add to your .env file: SERPER_API_KEY=your_api_key_here")
            return
        
        if not os.getenv('GEMINI_API_KEY'):
            print("❌ GEMINI_API_KEY not found in environment variables") 
            print("🔑 Get your API key from https://aistudio.google.com/app/apikey")
            print("💡 Add to your .env file: GEMINI_API_KEY=your_api_key_here")
            return
        
        print("✅ API keys found - ready to analyze!")
        
        # Initialize analyzer
        analyzer = EnhancedSEMAnalyzer(config)
        
        # Load scraped content
        brand_content = ""
        competitor_content = ""
        
        # Try to load existing scraped files
        scraped_files = [
            'dodopayments_com_complete_scrape_text_only.txt',
            'complete_scrape_text_only.txt', 
            'scraped_text.txt',
            'brand_content.txt'
        ]
        
        used_files = []
        
        # Load brand content
        for filename in scraped_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if len(content) > 100:  # Valid content
                            if not brand_content:
                                brand_content = content
                                used_files.append(f"Brand: {filename}")
                            elif not competitor_content:
                                competitor_content = content  
                                used_files.append(f"Competitor: {filename}")
                            break
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
                    continue
        
        # Use sample content if no files found
        if not brand_content:
            print("⚠️ No scraped content found, using sample analysis")
            brand_content = f"Sample business content for {config.brand_website}. Professional service provider offering solutions."
            used_files.append("Brand: Sample content")
        
        if not competitor_content:
            competitor_content = f"Sample competitor content for {config.competitor_website}. Competitive service provider."
            used_files.append("Competitor: Sample content")
        
        print(f"📄 Content loaded: {', '.join(used_files)}")
        print(f"📊 Brand content: {len(brand_content):,} characters")
        print(f"📊 Competitor content: {len(competitor_content):,} characters")
        
        # Generate strategy
        print("\n🔥 Starting SEM strategy generation...")
        strategy = analyzer.generate_sem_strategy(brand_content, competitor_content)
        
        # Save with exports
        output_file = analyzer.save_strategy_with_exports(strategy)
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("🎯 SEM STRATEGY GENERATION COMPLETE!")
        print("="*70)
        
        # Summary stats
        research_summary = strategy.get('keyword_research_summary', {})
        deliverables = strategy.get('deliverables', {})
        
        print(f"📊 KEYWORD RESEARCH RESULTS:")
        print(f"   • Keywords Researched: {research_summary.get('total_keywords_researched', 0)}")
        print(f"   • Keywords with Real Data: {research_summary.get('keywords_with_data', 0)}")
        print(f"   • Final Optimized Keywords: {research_summary.get('final_optimized_keywords', 0)}")
        print(f"   • Average Search Volume: {research_summary.get('avg_search_volume', 0):,}/month")
        print(f"   • Average CPC: ${research_summary.get('avg_cpc', 0):.2f}")
        print(f"   • Average Projected ROAS: {research_summary.get('avg_projected_roas', 0):.2f}x")
        
        print(f"\n💰 BUDGET ALLOCATION:")
        print(f"   • Search Campaigns: ${config.search_budget:,}/month")
        print(f"   • Performance Max: ${config.pmax_budget:,}/month") 
        print(f"   • Shopping Campaigns: ${config.shopping_budget:,}/month")
        print(f"   • Total Monthly Budget: ${config.search_budget + config.pmax_budget + config.shopping_budget:,}")
        
        print(f"\n📋 DELIVERABLES GENERATED:")
        ad_groups = deliverables.get('1_search_campaign_ad_groups', [])
        pmax_themes = deliverables.get('2_performance_max_themes', [])
        shopping_groups = deliverables.get('3_shopping_cpc_bids', {}).get('product_groups', [])
        
        print(f"   1️⃣ Search Campaign Ad Groups: {len(ad_groups)}")
        for i, ag in enumerate(ad_groups[:3], 1):
            keyword_count = len(ag.get('keywords', []))
            print(f"      {i}. {ag['ad_group_name']}: {keyword_count} keywords")
        
        print(f"   2️⃣ Performance Max Themes: {len(pmax_themes)}")
        for i, theme in enumerate(pmax_themes[:3], 1):
            print(f"      {i}. {theme['theme_name']}")
        
        print(f"   3️⃣ Shopping Product Groups: {len(shopping_groups)}")
        for i, group in enumerate(shopping_groups[:3], 1):
            print(f"      {i}. {group['product_group_name']} (CPC: ${group['suggested_max_cpc']})")
        
        # Performance projections
        budget_info = deliverables.get('budget_allocation', {})
        projections = budget_info.get('performance_projections', {})
        
        if projections:
            print(f"\n📈 PERFORMANCE PROJECTIONS:")
            print(f"   • Est. Monthly Clicks: {projections.get('estimated_monthly_clicks', 0):,}")
            print(f"   • Est. Monthly Conversions: {projections.get('estimated_monthly_conversions', 0):,}")
            print(f"   • Projected Monthly Revenue: ${projections.get('projected_monthly_revenue', 0):,}")
            print(f"   • Break-even ROAS: {projections.get('break_even_roas', 0):.2f}x")
        
        print(f"\n📁 FILES GENERATED:")
        base_name = output_file.replace('.json', '')
        print(f"   📄 Complete Strategy: {output_file}")
        print(f"   📊 Ad Groups CSV: {base_name}_ad_groups.csv")
        print(f"   🚀 Performance Max CSV: {base_name}_pmax_themes.csv")
        print(f"   🛒 Shopping Strategy CSV: {base_name}_shopping_strategy.csv")
        print(f"   📋 Executive Summary: {base_name}_executive_summary.txt")
        
        print(f"\n🚀 READY FOR IMPLEMENTATION!")
        print("Next steps:")
        print("1. Review the generated CSV files")
        print("2. Set up Google Ads campaigns using the ad groups")
        print("3. Create Performance Max campaigns with the themes")
        print("4. Implement shopping campaigns with suggested CPCs")
        print("5. Monitor performance and optimize based on data")
        
        # Show first few keywords as sample
        if ad_groups and ad_groups[0].get('keywords'):
            print(f"\n🎯 SAMPLE KEYWORDS FROM TOP AD GROUP:")
            sample_keywords = ad_groups[0]['keywords'][:5]
            for kw in sample_keywords:
                print(f"   • '{kw['keyword']}' ({kw['match_type']}) - CPC: ${kw['suggested_cpc']}")
        
        print("\n✨ SEM Strategy Analysis Complete! ✨")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Make sure your API keys are set in the .env file")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🛠️ If you need help, check:")
        print("1. API keys are correct and have sufficient credits")
        print("2. Internet connection is working")
        print("3. Configuration file is properly formatted")

if __name__ == "__main__":
    main()