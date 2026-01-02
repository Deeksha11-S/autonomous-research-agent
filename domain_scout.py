from typing import List, Dict, Any
import asyncio
from datetime import datetime, timedelta
import re
from loguru import logger

from backend.agents.base_agent import BaseAgent, AgentMessage
from backend.tools.scraper import WebScraper
from backend.tools.arxiv_client import ArxivClient
from backend.config import settings


class DomainScoutAgent(BaseAgent):
    """Discovers emerging scientific domains post-2024"""

    def __init__(self, agent_id: str = "scout_001"):
        capabilities = [
            "web_search", "trend_analysis", "domain_discovery",
            "academic_search", "github_trending", "patent_analysis"
        ]
        super().__init__(agent_id, "domain_scout", capabilities)

        # Initialize tools
        self.scraper = WebScraper()
        self.arxiv = ArxivClient()
        self.setup_tools()

    def setup_tools(self):
        """Setup domain discovery tools"""
        self.add_tool(
            name="search_emerging_trends",
            tool_func=self.search_emerging_trends,
            description="Search for emerging scientific trends using multiple sources"
        )

        self.add_tool(
            name="analyze_arxiv_categories",
            tool_func=self.analyze_arxiv_categories,
            description="Analyze new ArXiv categories and publication trends"
        )

        self.add_tool(
            name="check_github_trending",
            tool_func=self.check_github_trending,
            description="Check trending repositories in scientific domains"
        )

        self.add_tool(
            name="search_twitter_threads",
            tool_func=self.search_twitter_threads,
            description="Search for scientific discussions on Twitter/X"
        )

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process domain discovery request"""
        self.log(f"Received message: {message.message_type}")

        if message.message_type == "command" and message.content.get("action") == "discover_domains":
            try:
                domains = await self.discover_emerging_domains()

                return AgentMessage(
                    sender=self.agent_id,
                    recipient=message.sender,
                    content={
                        "domains": domains,
                        "total_domains": len(domains),
                        "discovery_date": datetime.now().isoformat()
                    },
                    message_type="result",
                    confidence=self.calculate_domain_confidence(domains)
                )
            except Exception as e:
                self.log(f"Domain discovery failed: {e}", "error")
                raise

        return await super().process(message)

    async def discover_emerging_domains(self) -> List[Dict[str, Any]]:
        """Discover emerging scientific domains post-2024"""
        self.log("Starting domain discovery...")

        # Parallel search across multiple sources
        tasks = [
            self.search_emerging_trends(),
            self.analyze_arxiv_categories(),
            self.check_github_trending(),
            self.search_twitter_threads()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process and merge results
        all_domains = {}

        for result in results:
            if isinstance(result, Exception):
                self.log(f"Source failed: {result}", "warning")
                continue

            for domain in result:
                domain_name = domain["name"]
                if domain_name not in all_domains:
                    all_domains[domain_name] = {
                        "name": domain_name,
                        "sources": [],
                        "evidence": [],
                        "first_seen": domain.get("first_seen"),
                        "momentum": 0
                    }

                all_domains[domain_name]["sources"].append(domain["source"])
                all_domains[domain_name]["evidence"].extend(domain.get("evidence", []))
                all_domains[domain_name]["momentum"] += domain.get("momentum", 1)

        # Filter for post-2024 emergence
        filtered_domains = []
        current_year = datetime.now().year

        for domain in all_domains.values():
            # Check if domain is emerging (based on evidence recency)
            is_emerging = self._is_emerging_domain(domain)

            if is_emerging and len(domain["sources"]) >= settings.min_domain_sources:
                # Calculate confidence score
                confidence = self._calculate_domain_confidence(domain)

                filtered_domains.append({
                    **domain,
                    "confidence": confidence,
                    "is_emerging": True,
                    "discovery_score": self._calculate_discovery_score(domain)
                })

        # Sort by discovery score
        filtered_domains.sort(key=lambda x: x["discovery_score"], reverse=True)

        # Store in memory
        for domain in filtered_domains[:10]:  # Store top 10
            self.memory.store(
                f"Emerging domain: {domain['name']}",
                metadata={
                    "type": "domain",
                    "confidence": domain["confidence"],
                    "sources": domain["sources"],
                    "momentum": domain["momentum"]
                }
            )

        self.log(f"Discovered {len(filtered_domains)} emerging domains")
        return filtered_domains[:5]  # Return top 5

    async def search_emerging_trends(self) -> List[Dict]:
        """Search for emerging trends using SerpAPI"""
        import requests

        self.log("Searching for emerging trends...")

        queries = [
            "emerging scientific fields 2024 2025",
            "new research areas after 2024",
            "cutting-edge science 2024",
            "scientific breakthroughs 2024 2025",
            "future of science 2025"
        ]

        domains = []

        for query in queries:
            try:
                params = {
                    'q': query,
                    'api_key': settings.serper_api_key,
                    'num': 10
                }

                response = requests.get(
                    'https://google.serper.dev/search',
                    params=params
                )
                response.raise_for_status()

                data = response.json()

                for item in data.get('organic', []):
                    title = item.get('title', '')
                    snippet = item.get('snippet', '')

                    # Extract potential domain names
                    extracted = self._extract_domains_from_text(f"{title} {snippet}")

                    for domain_name in extracted:
                        domains.append({
                            "name": domain_name,
                            "source": "serpapi",
                            "evidence": [snippet[:200]],
                            "momentum": 2,
                            "first_seen": datetime.now().isoformat()
                        })

            except Exception as e:
                self.log(f"SerpAPI search failed for '{query}': {e}", "warning")
                continue

        return domains

    async def analyze_arxiv_categories(self) -> List[Dict]:
        """Analyze ArXiv for new categories and trends"""
        self.log("Analyzing ArXiv categories...")

        try:
            # Search for recent papers in new categories
            categories = await self.arxiv.get_new_categories()

            domains = []
            for category in categories:
                # Get recent papers in this category
                papers = await self.arxiv.search(
                    query=f"cat:{category}",
                    max_results=5,
                    sort_by="submittedDate",
                    sort_order="descending"
                )

                if papers:
                    paper_titles = [p["title"] for p in papers[:3]]

                    domains.append({
                        "name": category.replace(".", " ").title(),
                        "source": "arxiv",
                        "evidence": paper_titles,
                        "momentum": len(papers),
                        "first_seen": papers[0]["published"]
                    })

            return domains

        except Exception as e:
            self.log(f"ArXiv analysis failed: {e}", "warning")
            return []

    async def check_github_trending(self) -> List[Dict]:
        """Check GitHub trending repositories for scientific domains"""
        self.log("Checking GitHub trending...")

        try:
            # Use GitHub API or scrape trending page
            trending_url = "https://github.com/trending"
            html = await self.scraper.scrape_page(trending_url)

            # Extract repository information
            repos = self._extract_github_repos(html)

            domains = []
            for repo in repos:
                # Analyze repo description for scientific domains
                description = repo.get("description", "").lower()

                scientific_keywords = [
                    "quantum", "ai", "machine learning", "bioinformatics",
                    "neuroscience", "genomics", "climate", "energy",
                    "material science", "cryptography", "robotics"
                ]

                for keyword in scientific_keywords:
                    if keyword in description:
                        domains.append({
                            "name": keyword.title(),
                            "source": "github",
                            "evidence": [repo["description"]],
                            "momentum": repo.get("stars", 1),
                            "first_seen": datetime.now().isoformat()
                        })
                        break

            return domains

        except Exception as e:
            self.log(f"GitHub trending check failed: {e}", "warning")
            return []

    async def search_twitter_threads(self) -> List[Dict]:
        """Search for scientific discussions on Twitter/X"""
        self.log("Searching Twitter threads...")

        # Note: Twitter API requires v2 access. For free tier, we'll use web scraping
        try:
            search_url = "https://twitter.com/search?q=scientific%20breakthrough%202024&f=live"
            html = await self.scraper.scrape_page(search_url)

            # Extract tweets (simplified)
            tweets = self._extract_tweets(html)

            domains = []
            for tweet in tweets:
                text = tweet.get("text", "").lower()

                # Look for emerging domain mentions
                emerging_indicators = [
                    "new field", "emerging", "breakthrough in",
                    "recent discovery", "novel approach", "paradigm shift"
                ]

                for indicator in emerging_indicators:
                    if indicator in text:
                        # Extract potential domain
                        domain_name = self._extract_domain_from_context(text)
                        if domain_name:
                            domains.append({
                                "name": domain_name,
                                "source": "twitter",
                                "evidence": [text[:200]],
                                "momentum": tweet.get("likes", 1),
                                "first_seen": datetime.now().isoformat()
                            })
                        break

            return domains

        except Exception as e:
            self.log(f"Twitter search failed: {e}", "warning")
            return []

    def _is_emerging_domain(self, domain: Dict) -> bool:
        """Check if domain is truly emerging (post-2024)"""
        # Check evidence recency
        if domain.get("first_seen"):
            first_seen = datetime.fromisoformat(domain["first_seen"].replace('Z', '+00:00'))
            cutoff_date = datetime(2024, 1, 1)

            if first_seen >= cutoff_date:
                return True

        # Check for emerging indicators in evidence
        emerging_keywords = [
            "emerging", "new", "novel", "recent", "cutting-edge",
            "breakthrough", "pioneering", "groundbreaking"
        ]

        evidence_text = " ".join(str(e) for e in domain.get("evidence", []))
        evidence_lower = evidence_text.lower()

        keyword_count = sum(1 for keyword in emerging_keywords if keyword in evidence_lower)

        return keyword_count >= 2 or domain.get("momentum", 0) > 5

    def _calculate_domain_confidence(self, domain: Dict) -> float:
        """Calculate confidence score for a domain"""
        base_score = 0.5

        # Source diversity bonus
        source_count = len(set(domain.get("sources", [])))
        base_score += min(source_count * 0.1, 0.3)

        # Evidence quality bonus
        evidence_count = len(domain.get("evidence", []))
        base_score += min(evidence_count * 0.05, 0.2)

        # Momentum bonus
        momentum = domain.get("momentum", 0)
        base_score += min(momentum * 0.02, 0.2)

        return min(base_score, 1.0)

    def _calculate_discovery_score(self, domain: Dict) -> float:
        """Calculate overall discovery score"""
        confidence = domain.get("confidence", 0.5)
        momentum = domain.get("momentum", 0) / 100  # Normalize
        source_diversity = len(set(domain.get("sources", []))) / 5  # Max 5 sources

        return (confidence * 0.4) + (momentum * 0.3) + (source_diversity * 0.3)

    def _extract_domains_from_text(self, text: str) -> List[str]:
        """Extract potential domain names from text"""
        # Common domain patterns
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:science|field|research|domain)\b',
            r'\b(?:emerging|new|novel)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+for\s+[A-Z][a-z]+\b'
        ]

        domains = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            domains.update(matches)

        return list(domains)[:5]  # Return max 5 domains

    def calculate_domain_confidence(self, domains: List[Dict]) -> float:
        """Calculate overall confidence for domain discovery"""
        if not domains:
            return 0.0

        avg_confidence = sum(d.get("confidence", 0) for d in domains) / len(domains)
        domain_diversity = len(set(d["name"] for d in domains)) / len(domains)

        return (avg_confidence * 0.7) + (domain_diversity * 0.3)