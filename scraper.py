import asyncio
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright
import requests
from bs4 import BeautifulSoup
from loguru import logger
import pdfplumber
import pytesseract
from PIL import Image
import io

from backend.utils import retry_async, sanitize_text


class WebScraper:
    """Advanced web scraper with OCR and PDF capabilities"""

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self):
        """Start Playwright browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

    async def close(self):
        """Close browser"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def scrape_page(self, url: str, wait_for: Optional[str] = None) -> str:
        """Scrape web page content"""

        async def _scrape():
            page = await self.context.new_page()
            try:
                await page.goto(url, wait_until="networkidle")

                if wait_for:
                    await page.wait_for_selector(wait_for, timeout=10000)

                # Get page content
                content = await page.content()

                # Extract text with BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                text = soup.get_text(separator=' ', strip=True)
                return sanitize_text(text)

            finally:
                await page.close()

        return await retry_async(_scrape, max_retries=3)

    async def scrape_multiple_pages(self, urls: list) -> Dict[str, str]:
        """Scrape multiple pages concurrently"""
        tasks = [self.scrape_page(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped_data = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to scrape {url}: {result}")
                scraped_data[url] = ""
            else:
                scraped_data[url] = result

        return scraped_data

    def extract_tables(self, html: str) -> list:
        """Extract tables from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        tables = []

        for table in soup.find_all('table'):
            table_data = []
            headers = []

            # Extract headers
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

            # Extract rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:
                    table_data.append(row_data)

            if table_data:
                tables.append({
                    "headers": headers,
                    "data": table_data
                })

        return tables

    async def scrape_pdf(self, url: str) -> str:
        """Scrape and extract text from PDF"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                    # Extract tables from page
                    tables = page.extract_tables()
                    for table in tables:
                        text += "\n[Table]\n"
                        for row in table:
                            text += " | ".join(str(cell) for cell in row if cell) + "\n"

                return sanitize_text(text)

        except Exception as e:
            logger.error(f"PDF scraping failed for {url}: {e}")

            # Fallback: Download and try OCR
            try:
                return await self._ocr_pdf(url)
            except Exception as ocr_error:
                logger.error(f"OCR also failed: {ocr_error}")
                return ""

    async def _ocr_pdf(self, url: str) -> str:
        """Fallback OCR for PDFs"""
        response = requests.get(url, timeout=30)

        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            text = ""
            for page in pdf.pages:
                # Convert page to image
                im = page.to_image(resolution=150).original

                # OCR the image
                page_text = pytesseract.image_to_string(im)
                text += page_text + "\n"

            return sanitize_text(text)

    def extract_links(self, html: str, base_url: str) -> list:
        """Extract all links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            # Convert relative URLs to absolute
            if href.startswith('/'):
                href = base_url + href
            elif not href.startswith(('http://', 'https://')):
                href = base_url + '/' + href

            links.append({
                "url": href,
                "text": a.get_text(strip=True)
            })

        return links