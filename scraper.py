from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse

def setup_driver():
    """Set up Chrome driver with options"""
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run without GUI
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # You might need to specify ChromeDriver path
    # service = Service('/path/to/chromedriver')  # Update this path
    # driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # If chromedriver is in PATH, use this:
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def scrape_page_content(url):
    """Scrape all content from a single page"""
    driver = setup_driver()
    
    try:
        print(f"🌐 Loading: {url}")
        driver.get(url)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Scroll to load any dynamic content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Get page source after JavaScript execution
        html_content = driver.page_source
        print("✅ Page loaded successfully")
        
        return html_content
    
    except Exception as e:
        print(f"❌ Error loading page: {str(e)}")
        return None
    
    finally:
        driver.quit()

def parse_html_content(html_content, url):
    """Parse HTML content using BeautifulSoup"""
    if not html_content:
        return None
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    
    # Extract structured data
    data = {
        'url': url,
        'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'page_info': {},
        'content': {},
        'structure': {},
        'all_text': '',
        'html_content': str(soup)  # Full HTML for reference
    }
    
    # Basic page information
    data['page_info'] = {
        'title': soup.find('title').get_text().strip() if soup.find('title') else 'No title',
        'description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else 'No description',
        'keywords': soup.find('meta', attrs={'name': 'keywords'})['content'] if soup.find('meta', attrs={'name': 'keywords'}) else 'No keywords',
        'canonical': soup.find('link', attrs={'rel': 'canonical'})['href'] if soup.find('link', attrs={'rel': 'canonical'}) else None
    }
    
    # Extract all headings
    headings = []
    for i in range(1, 7):  # h1 to h6
        for heading in soup.find_all(f'h{i}'):
            headings.append({
                'level': f'h{i}',
                'text': heading.get_text().strip(),
                'id': heading.get('id', ''),
                'class': ' '.join(heading.get('class', []))
            })
    data['structure']['headings'] = headings
    
    # Extract all paragraphs
    paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
    data['content']['paragraphs'] = paragraphs
    
    # Extract all links
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text().strip()
        if text and href:
            full_url = urljoin(url, href)
            links.append({
                'text': text,
                'href': href,
                'full_url': full_url,
                'is_internal': urlparse(full_url).netloc == urlparse(url).netloc
            })
    data['structure']['links'] = links
    
    # Extract lists
    lists = []
    for ul in soup.find_all(['ul', 'ol']):
        list_items = [li.get_text().strip() for li in ul.find_all('li') if li.get_text().strip()]
        if list_items:
            lists.append({
                'type': ul.name,
                'items': list_items
            })
    data['content']['lists'] = lists
    
    # Extract images
    images = []
    for img in soup.find_all('img'):
        src = img.get('src', '')
        alt = img.get('alt', '')
        if src:
            images.append({
                'src': urljoin(url, src),
                'alt': alt,
                'title': img.get('title', '')
            })
    data['content']['images'] = images
    
    # Extract forms (useful for understanding lead capture)
    forms = []
    for form in soup.find_all('form'):
        form_data = {
            'action': form.get('action', ''),
            'method': form.get('method', 'get'),
            'inputs': []
        }
        for input_tag in form.find_all(['input', 'textarea', 'select']):
            form_data['inputs'].append({
                'type': input_tag.get('type', input_tag.name),
                'name': input_tag.get('name', ''),
                'placeholder': input_tag.get('placeholder', ''),
                'required': input_tag.has_attr('required')
            })
        forms.append(form_data)
    data['structure']['forms'] = forms
    
    # Extract all visible text
    text_content = soup.get_text()
    # Clean up whitespace
    lines = (line.strip() for line in text_content.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    data['all_text'] = ' '.join(chunk for chunk in chunks if chunk)
    
    # Extract pricing information
    pricing_patterns = [
        r'\$[\d,]+\.?\d*',  # $99, $1,299.99
        r'\d+\.?\d*\s*%',   # 2.9%, 15%
        r'€[\d,]+\.?\d*',   # €99
        r'£[\d,]+\.?\d*',   # £99
        r'\d+\.?\d*\s*(dollars?|cents?|euro?|pounds?)',
        r'(free|trial|demo)',
        r'(starting\s+at|from|only|just)\s*\$?\d+',
        r'(monthly|yearly|annual|per\s+month|per\s+year)',
        r'(subscription|pricing|cost|fee|charge)'
    ]
    
    pricing_info = []
    for pattern in pricing_patterns:
        matches = re.finditer(pattern, data['all_text'], re.IGNORECASE)
        for match in matches:
            context_start = max(0, match.start() - 50)
            context_end = min(len(data['all_text']), match.end() + 50)
            context = data['all_text'][context_start:context_end].strip()
            pricing_info.append({
                'match': match.group(),
                'context': context,
                'position': match.start()
            })
    
    data['content']['pricing_info'] = pricing_info[:20]  # Limit to first 20 matches
    
    # Extract business/service information
    business_keywords = [
        'services', 'products', 'solutions', 'features', 'benefits',
        'about us', 'contact', 'phone', 'email', 'address',
        'testimonials', 'reviews', 'clients', 'customers',
        'industries', 'case studies', 'portfolio'
    ]
    
    business_info = {}
    for keyword in business_keywords:
        pattern = rf'.{{0,100}}{re.escape(keyword)}.{{0,100}}'
        matches = re.finditer(pattern, data['all_text'], re.IGNORECASE)
        contexts = [match.group().strip() for match in matches]
        if contexts:
            business_info[keyword] = contexts[:5]  # Top 5 matches
    
    data['content']['business_info'] = business_info
    
    # Content statistics
    data['statistics'] = {
        'total_words': len(data['all_text'].split()),
        'total_characters': len(data['all_text']),
        'headings_count': len(headings),
        'paragraphs_count': len(paragraphs),
        'links_count': len(links),
        'images_count': len(images),
        'forms_count': len(forms),
        'pricing_mentions': len(pricing_info)
    }
    
    return data

def scrape_website(url, output_filename=None):
    """Main function to scrape website and save results"""
    print(f"🚀 Starting complete scrape of: {url}")
    print("="*60)
    
    # Scrape HTML content
    html_content = scrape_page_content(url)
    if not html_content:
        print("❌ Failed to scrape website")
        return None
    
    # Parse content with BeautifulSoup
    print("🔍 Parsing HTML content with BeautifulSoup...")
    parsed_data = parse_html_content(html_content, url)
    
    if not parsed_data:
        print("❌ Failed to parse HTML content")
        return None
    
    # Print summary
    print("\n📊 SCRAPING RESULTS SUMMARY")
    print("="*60)
    print(f"🏷️  Title: {parsed_data['page_info']['title']}")
    print(f"📝 Description: {parsed_data['page_info']['description'][:100]}...")
    print(f"📈 Statistics:")
    for key, value in parsed_data['statistics'].items():
        print(f"   • {key.replace('_', ' ').title()}: {value:,}")
    
    print(f"\n🔤 First 200 characters of text:")
    print(f"   {parsed_data['all_text'][:200]}...")
    
    print(f"\n📋 Top 5 headings:")
    for heading in parsed_data['structure']['headings'][:5]:
        print(f"   • {heading['level'].upper()}: {heading['text']}")
    
    if parsed_data['content']['pricing_info']:
        print(f"\n💰 Pricing information found:")
        for price in parsed_data['content']['pricing_info'][:3]:
            print(f"   • {price['match']} - {price['context'][:50]}...")
    
    # Save to file
    if not output_filename:
        domain = urlparse(url).netloc.replace('www.', '').replace('.', '_')
        output_filename = f"{domain}_complete_scrape.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Complete data saved to: {output_filename}")
    
    # Also save just the clean text
    text_filename = output_filename.replace('.json', '_text_only.txt')
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(f"Website: {url}\n")
        f.write(f"Scraped: {parsed_data['scraped_at']}\n")
        f.write("="*80 + "\n\n")
        f.write(parsed_data['all_text'])
    
    print(f"📄 Clean text saved to: {text_filename}")
    
    return parsed_data

if __name__ == "__main__":
    # Example usage
    url = "https://dodopayments.com/"
    
    try:
        result = scrape_website(url)
        if result:
            print("\n🎉 Scraping completed successfully!")
            print(f"Total text extracted: {len(result['all_text']):,} characters")
        else:
            print("❌ Scraping failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure you have Chrome and chromedriver installed!")