import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

st.set_page_config(
    page_title="SEM Strategy Analyzer",
    page_icon="📊",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    st.title("🚀 SEM Strategy Analyzer MVP")
    st.markdown("Generate comprehensive Google Ads strategies using AI and real market data")
    
    # Check API status
    if check_api_health():
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.info("Start FastAPI server: `uvicorn backend.main:app --reload`")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Website inputs
        st.subheader("🌐 Websites")
        brand_website = st.text_input(
            "Brand Website", 
            value="https://stripe.com",
            help="Your website URL"
        )
        
        competitor_website = st.text_input(
            "Competitor Website", 
            value="https://square.com",
            help="Competitor website URL"
        )
        
        # Service locations
        st.subheader("📍 Target Locations")
        locations_text = st.text_area(
            "Service Locations (one per line)",
            value="United States\nCanada\nUnited Kingdom"
        )
        service_locations = [loc.strip() for loc in locations_text.split('\n') if loc.strip()]
        
        # Budget settings
        st.subheader("💰 Budget Settings")
        col1, col2 = st.columns(2)
        with col1:
            search_budget = st.number_input("Search Budget ($)", value=5000.0, min_value=100.0)
            pmax_budget = st.number_input("PMax Budget ($)", value=3000.0, min_value=100.0)
        with col2:
            shopping_budget = st.number_input("Shopping Budget ($)", value=2000.0, min_value=100.0)
            avg_order_value = st.number_input("Avg Order Value ($)", value=150.0, min_value=1.0)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            conversion_rate = st.slider("Conversion Rate (%)", 0.5, 10.0, 2.0, 0.1) / 100
            target_roas = st.slider("Target ROAS", 2.0, 8.0, 4.0, 0.1)
            min_search_volume = st.number_input("Min Search Volume", value=500, min_value=100)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analyze Strategy", type="primary", use_container_width=True)
    
    with col1:
        total_budget = search_budget + pmax_budget + shopping_budget
        st.write(f"**Total Budget:** ${total_budget:,.0f}")
        st.write(f"**Target Locations:** {len(service_locations)} markets")
        st.write(f"**Expected ROAS:** {target_roas:.1f}x")
    
    # Analysis section
    if analyze_button:
        if not brand_website or not competitor_website:
            st.error("Please enter both websites")
            return
        
        with st.spinner("🔍 Analyzing websites and generating SEM strategy..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            payload = {
                "brand_website": brand_website,
                "competitor_website": competitor_website,
                "service_locations": service_locations,
                "shopping_budget": shopping_budget,
                "search_budget": search_budget,
                "pmax_budget": pmax_budget,
                "conversion_rate": conversion_rate,
                "min_search_volume": min_search_volume,
                "target_roas": target_roas,
                "avg_order_value": avg_order_value
            }
            
            try:
                status_text.text("🌐 Scraping websites...")
                progress_bar.progress(20)
                
                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(50)
                
                response = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=300)
                
                status_text.text("📊 Generating results...")
                progress_bar.progress(80)
                
                if response.status_code == 200:
                    result = response.json()
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed!")
                    
                    if result["status"] == "success":
                        display_results(result["data"])
                    else:
                        st.error(f"Analysis failed: {result['message']}")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Analysis timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API. Make sure FastAPI server is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def display_results(strategy_data):
    """Display the SEM strategy results"""
    
    st.success("🎉 SEM Strategy Generated Successfully!")
    
    # Overview metrics
    st.header("📊 Strategy Overview")
    
    summary = strategy_data.get("keyword_research_summary", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Keywords Analyzed", summary.get("total_keywords_researched", 0))
    with col2:
        st.metric("Final Keywords", summary.get("final_optimized_keywords", 0))
    with col3:
        st.metric("Avg Search Volume", f"{summary.get('avg_search_volume', 0):,}")
    with col4:
        st.metric("Avg Projected ROAS", f"{summary.get('avg_projected_roas', 0):.1f}x")
    
    # Display deliverables in tabs
    deliverables = strategy_data.get("deliverables", {})
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Search Campaigns", "⚡ Performance Max", "🛍️ Shopping", "🔍 Analysis"])
    
    with tab1:
        display_search_campaigns(deliverables.get("1_search_campaign_ad_groups", []))
    
    with tab2:
        display_pmax_themes(deliverables.get("2_performance_max_themes", []))
    
    with tab3:
        display_shopping_strategy(deliverables.get("3_shopping_cpc_bids", {}))
    
    with tab4:
        display_analysis_data(strategy_data)

def display_search_campaigns(ad_groups):
    """Display search campaign ad groups"""
    
    st.subheader("🎯 Search Campaign Ad Groups")
    
    if not ad_groups:
        st.warning("No ad groups generated")
        return
    
    for group in ad_groups:
        with st.expander(f"📁 {group.get('ad_group_name', 'Ad Group')} ({len(group.get('keywords', []))} keywords)"):
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Theme:** {group.get('theme', 'N/A')}")
                st.write(f"**Audience:** {group.get('target_audience', 'N/A')}")
            with col2:
                budget = group.get('estimated_daily_budget', 0)
                st.metric("Est. Daily Budget", f"${budget:.2f}")
            
            # Keywords table
            if group.get('keywords'):
                keywords_data = []
                for kw in group['keywords']:
                    keywords_data.append({
                        'Keyword': kw.get('keyword', ''),
                        'Match Type': kw.get('match_type', ''),
                        'CPC': f"${kw.get('suggested_cpc', 0):.2f}",
                        'Volume': f"{kw.get('search_volume', 0):,}",
                        'Competition': kw.get('competition', ''),
                        'ROAS': f"{kw.get('projected_roas', 0):.1f}x"
                    })
                
                df = pd.DataFrame(keywords_data)
                st.dataframe(df, use_container_width=True)

def display_pmax_themes(pmax_themes):
    """Display Performance Max themes"""
    
    st.subheader("⚡ Performance Max Campaign Themes")
    
    if not pmax_themes:
        st.warning("No Performance Max themes generated")
        return
    
    for theme in pmax_themes:
        with st.expander(f"🎨 {theme.get('theme_name', 'Theme')}"):
            st.write(f"**Focus:** {theme.get('asset_group_focus', 'N/A')}")
            st.write(f"**Audience:** {theme.get('target_audience', 'N/A')}")
            if theme.get('value_proposition'):
                st.write(f"**Value Prop:** {theme.get('value_proposition')}")

def display_shopping_strategy(shopping_data):
    """Display shopping strategy"""
    
    st.subheader("🛍️ Shopping Campaign Strategy")
    
    if not shopping_data:
        st.info("Shopping strategy will be displayed here")
        return
    
    st.json(shopping_data)

def display_analysis_data(strategy_data):
    """Display analysis insights"""
    
    st.subheader("🔍 AI Analysis Results")
    
    # Brand analysis
    brand_analysis = strategy_data.get("brand_analysis", {})
    if brand_analysis:
        st.write("### 🏢 Brand Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Business Type:** {brand_analysis.get('business_type', 'N/A')}")
            st.write(f"**Target Audience:** {brand_analysis.get('target_audience', 'N/A')}")
        with col2:
            services = brand_analysis.get('primary_services', [])
            st.write(f"**Services:** {', '.join(services[:3]) if services else 'N/A'}")
    
    # Competitor analysis
    competitor_analysis = strategy_data.get("competitor_analysis", {})
    if competitor_analysis:
        st.write("### 🏆 Competitor Analysis")
        st.write(f"**Business Type:** {competitor_analysis.get('business_type', 'N/A')}")
    
    # Show raw data option
    if st.checkbox("Show Raw Data"):
        st.json(strategy_data)

if __name__ == "__main__":
    main()
