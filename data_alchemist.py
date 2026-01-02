from typing import List, Dict, Any, Tuple
import asyncio
from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np
from io import StringIO

from backend.agents.base_agent import BaseAgent, AgentMessage
from backend.tools.scraper import WebScraper
from backend.tools.arxiv_client import ArxivClient
from backend.tools.data_cleaner import DataCleaner
from backend.config import settings


class DataAlchemistAgent(BaseAgent):
    """Fetches and cleans data from multiple disparate sources"""

    def __init__(self, agent_id: str = "data_001"):
        capabilities = [
            "web_scraping", "data_cleaning", "data_integration",
            "api_integration", "pdf_parsing", "table_extraction"
        ]
        super().__init__(agent_id, "data_alchemist", capabilities)

        # Initialize tools
        self.scraper = WebScraper()
        self.arxiv = ArxivClient()
        self.cleaner = DataCleaner()
        self.setup_tools()

    def setup_tools(self):
        """Setup data acquisition tools"""
        self.add_tool(
            name="search_data_sources",
            tool_func=self.search_data_sources_tool,
            description="Search for data sources relevant to research question"
        )

        self.add_tool(
            name="fetch_and_clean_data",
            tool_func=self.fetch_and_clean_data_tool,
            description="Fetch data from sources and clean it"
        )

        self.add_tool(
            name="integrate_data_sources",
            tool_func=self.integrate_data_sources_tool,
            description="Integrate data from multiple sources"
        )

        self.add_tool(
            name="assess_data_quality",
            tool_func=self.assess_data_quality_tool,
            description="Assess quality of collected data"
        )

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process data acquisition request"""
        self.log(f"Received message: {message.message_type}")

        if message.message_type == "command" and message.content.get("action") == "acquire_data":
            try:
                research_question = message.content.get("research_question", {})
                domain_info = message.content.get("domain_info", {})

                if not research_question:
                    raise ValueError("No research question provided")

                # Search for data sources
                sources = await self.search_data_sources(research_question, domain_info)

                # Fetch and clean data from each source
                all_data = []
                for source in sources[:settings.min_data_sources]:  # Limit to minimum required
                    try:
                        data = await self.fetch_and_clean_data(source, research_question)
                        if data:
                            all_data.append(data)
                            self.log(f"Successfully acquired data from {source['type']}")
                    except Exception as e:
                        self.log(f"Failed to fetch from {source['type']}: {e}", "warning")

                # Integrate data sources
                if len(all_data) >= settings.min_data_sources:
                    integrated_data = await self.integrate_data_sources(all_data, research_question)

                    # Assess data quality
                    quality_metrics = await self.assess_data_quality(integrated_data)

                    return AgentMessage(
                        sender=self.agent_id,
                        recipient=message.sender,
                        content={
                            "data": integrated_data,
                            "sources": [d["source_info"] for d in all_data],
                            "quality_metrics": quality_metrics,
                            "total_sources": len(all_data),
                            "acquisition_date": datetime.now().isoformat()
                        },
                        message_type="result",
                        confidence=quality_metrics.get("overall_quality", 0.5)
                    )
                else:
                    raise ValueError(
                        f"Insufficient data sources. Found {len(all_data)}, need at least {settings.min_data_sources}")

            except Exception as e:
                self.log(f"Data acquisition failed: {e}", "error")
                raise

        return await super().process(message)

    async def search_data_sources_tool(self, research_question: Dict[str, Any],
                                       domain_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for data sources relevant to research question"""
        question_text = research_question.get("question", "")
        domain_name = domain_info.get("name", "")

        prompt = f"""
        Find data sources for this research question. Provide at least 5 potential sources.

        Research Question: {question_text}
        Domain: {domain_name}

        For each source, provide:
        - Type: [Academic Paper, Public Dataset, API, Web Content, Government Data, etc.]
        - Description: [Brief description of what data it contains]
        - URL/Identifier: [If known]
        - Accessibility: [Open Access, Requires API Key, Free Tier Available, etc.]
        - Relevance: [High, Medium, Low]

        Prioritize sources that are:
        1. Open access and freely available
        2. Recent (last 5 years)
        3. From reputable organizations
        4. In machine-readable formats (CSV, JSON, API)
        """

        try:
            response = await self.llm.ainvoke(prompt)
            sources_text = response.content

            # Parse sources
            sources = self._parse_sources_from_text(sources_text)

            # Add search-based sources
            search_sources = await self._search_web_for_sources(question_text, domain_name)
            sources.extend(search_sources)

            # Remove duplicates
            unique_sources = []
            seen_urls = set()

            for source in sources:
                url = source.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(source)
                elif not url:
                    unique_sources.append(source)  # Keep sources without URLs

            self.log(f"Found {len(unique_sources)} potential data sources")
            return unique_sources[:10]  # Return top 10

        except Exception as e:
            self.log(f"Source search failed: {e}", "error")
            return []

    async def search_data_sources(self, research_question: Dict[str, Any],
                                  domain_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for data sources with multiple strategies"""
        sources = []

        # Strategy 1: LLM-based search
        llm_sources = await self.search_data_sources_tool(research_question, domain_info)
        sources.extend(llm_sources)

        # Strategy 2: ArXiv search
        try:
            arxiv_sources = await self._search_arxiv_sources(research_question)
            sources.extend(arxiv_sources)
        except Exception as e:
            self.log(f"ArXiv search failed: {e}", "warning")

        # Strategy 3: Dataset search
        try:
            dataset_sources = await self._search_dataset_sources(research_question)
            sources.extend(dataset_sources)
        except Exception as e:
            self.log(f"Dataset search failed: {e}", "warning")

        # Score and rank sources
        scored_sources = []
        for source in sources:
            score = self._calculate_source_score(source, research_question)
            scored_sources.append({
                **source,
                "relevance_score": score
            })

        # Sort by relevance
        scored_sources.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Store in memory
        for source in scored_sources[:5]:
            self.memory.store(
                f"Data source: {source.get('type', 'Unknown')} - {source.get('description', '')}",
                metadata={
                    "type": "data_source",
                    "relevance": source.get("relevance_score", 0),
                    "accessibility": source.get("accessibility", "Unknown")
                }
            )

        return scored_sources

    async def fetch_and_clean_data_tool(self, source: Dict[str, Any],
                                        research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch and clean data from a source"""
        source_type = source.get("type", "").lower()
        url = source.get("url", "")
        description = source.get("description", "")

        self.log(f"Fetching data from {source_type}: {url}")

        data = {
            "raw_data": None,
            "cleaned_data": None,
            "metadata": {},
            "source_info": source,
            "fetch_timestamp": datetime.now().isoformat()
        }

        try:
            # Fetch based on source type
            if "arxiv" in source_type or "paper" in source_type or "pdf" in source_type:
                if url:
                    raw_data = await self.scraper.scrape_pdf(url)
                    data["raw_data"] = raw_data

                    # Extract structured information
                    cleaned_data = self.cleaner.extract_from_paper(raw_data, research_question)
                    data["cleaned_data"] = cleaned_data

                    # Extract metadata
                    data["metadata"] = {
                        "document_type": "research_paper",
                        "pages": len(raw_data.split('\n')) // 50,  # Estimate
                        "has_tables": "table" in raw_data.lower(),
                        "has_figures": "figure" in raw_data.lower()
                    }

            elif "csv" in source_type or "dataset" in source_type:
                if url:
                    import requests
                    response = requests.get(url, timeout=30)

                    if response.status_code == 200:
                        raw_data = response.text
                        data["raw_data"] = raw_data

                        # Parse CSV
                        try:
                            df = pd.read_csv(StringIO(raw_data))
                            cleaned_data = self.cleaner.clean_dataframe(df, research_question)
                            data["cleaned_data"] = cleaned_data

                            data["metadata"] = {
                                "rows": len(df),
                                "columns": len(df.columns),
                                "data_types": str(df.dtypes.to_dict()),
                                "missing_values": df.isnull().sum().sum()
                            }
                        except:
                            # Try tab-separated
                            try:
                                df = pd.read_csv(StringIO(raw_data), sep='\t')
                                cleaned_data = self.cleaner.clean_dataframe(df, research_question)
                                data["cleaned_data"] = cleaned_data
                            except:
                                data["cleaned_data"] = {"error": "Could not parse as CSV/TSV"}

            elif "api" in source_type:
                if url:
                    import requests
                    response = requests.get(url, timeout=30)

                    if response.status_code == 200:
                        raw_data = response.json()
                        data["raw_data"] = raw_data

                        # Convert to structured format
                        cleaned_data = self.cleaner.clean_json_data(raw_data, research_question)
                        data["cleaned_data"] = cleaned_data

                        data["metadata"] = {
                            "data_format": "json",
                            "size_kb": len(str(raw_data)) / 1024
                        }

            elif "web" in source_type or "html" in source_type:
                if url:
                    raw_data = await self.scraper.scrape_page(url)
                    data["raw_data"] = raw_data

                    # Extract structured information
                    cleaned_data = self.cleaner.extract_from_html(raw_data, research_question)
                    data["cleaned_data"] = cleaned_data

                    # Extract tables if present
                    tables = self.scraper.extract_tables(raw_data)
                    if tables:
                        data["metadata"]["tables"] = len(tables)
                        data["cleaned_data"]["tables"] = tables

            else:
                # Generic text processing
                if description:
                    data["raw_data"] = description
                    data["cleaned_data"] = {
                        "text": description,
                        "keywords": self.cleaner.extract_keywords(description)
                    }

            # Add quality indicators
            if data["cleaned_data"]:
                data["quality_indicators"] = self._assess_data_quality_indicators(data)

            return data

        except Exception as e:
            self.log(f"Failed to fetch from {url}: {e}", "warning")
            data["error"] = str(e)
            return data

    async def fetch_and_clean_data(self, source: Dict[str, Any],
                                   research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data with fallback strategies"""
        try:
            data = await self.fetch_and_clean_data_tool(source, research_question)

            # If fetching failed, try alternative approach
            if not data.get("cleaned_data") or "error" in data:
                self.log(f"Primary fetch failed, trying alternative for {source.get('type')}")

                # Try extracting information from description
                description = source.get("description", "")
                if description and len(description) > 100:
                    keywords = self.cleaner.extract_keywords(description)

                    data["cleaned_data"] = {
                        "text": description,
                        "keywords": keywords,
                        "entities": self.cleaner.extract_entities(description),
                        "summary": description[:500] + "..." if len(description) > 500 else description
                    }
                    data["metadata"] = {"source": "description_extraction"}

            return data

        except Exception as e:
            self.log(f"Complete fetch failed for {source.get('type')}: {e}", "error")
            return {
                "source_info": source,
                "error": str(e),
                "fetch_timestamp": datetime.now().isoformat()
            }

    async def integrate_data_sources_tool(self, data_list: List[Dict[str, Any]],
                                          research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate data from multiple sources"""
        self.log(f"Integrating data from {len(data_list)} sources")

        integrated_data = {
            "sources": [],
            "unified_data": {},
            "integration_method": "semantic",
            "integration_timestamp": datetime.now().isoformat(),
            "research_question": research_question.get("question", "")
        }

        # Collect all cleaned data
        all_cleaned_data = []
        for data_item in data_list:
            if data_item.get("cleaned_data"):
                all_cleaned_data.append(data_item["cleaned_data"])
                integrated_data["sources"].append(data_item["source_info"])

        if not all_cleaned_data:
            return integrated_data

        # Strategy 1: Text-based integration (for papers, web content)
        text_data = []
        table_data = []
        structured_data = []

        for data in all_cleaned_data:
            if isinstance(data, dict):
                if "text" in data:
                    text_data.append(data["text"])
                if "tables" in data:
                    table_data.extend(data["tables"])
                if "structured" in data:
                    structured_data.append(data["structured"])
            elif isinstance(data, pd.DataFrame):
                table_data.append(data)
            elif isinstance(data, str) and len(data) > 100:
                text_data.append(data)

        # Integrate text data
        if text_data:
            integrated_text = self.cleaner.integrate_text_data(text_data, research_question)
            integrated_data["unified_data"]["text"] = integrated_text

        # Integrate table data
        if table_data:
            integrated_tables = self.cleaner.integrate_tables(table_data, research_question)
            integrated_data["unified_data"]["tables"] = integrated_tables

        # Integrate structured data
        if structured_data:
            integrated_structured = self.cleaner.integrate_structured_data(structured_data, research_question)
            integrated_data["unified_data"]["structured"] = integrated_structured

        # Extract key insights
        if integrated_data["unified_data"].get("text"):
            insights = self.cleaner.extract_insights(
                integrated_data["unified_data"]["text"],
                research_question
            )
            integrated_data["key_insights"] = insights

        # Create summary
        summary = self.cleaner.create_data_summary(integrated_data, research_question)
        integrated_data["summary"] = summary

        return integrated_data

    async def integrate_data_sources(self, data_list: List[Dict[str, Any]],
                                     research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate data with quality checks"""
        integrated_data = await self.integrate_data_sources_tool(data_list, research_question)

        # Add integration metrics
        source_count = len(data_list)
        successful_sources = sum(1 for d in data_list if d.get("cleaned_data"))

        integration_metrics = {
            "total_sources": source_count,
            "successful_sources": successful_sources,
            "success_rate": successful_sources / max(source_count, 1),
            "data_types": self._identify_data_types(integrated_data),
            "integration_complexity": self._calculate_integration_complexity(data_list)
        }

        integrated_data["integration_metrics"] = integration_metrics

        # Store in memory
        self.memory.store(
            f"Integrated data for: {research_question.get('question', '')[:100]}",
            metadata={
                "type": "integrated_data",
                "source_count": source_count,
                "success_rate": integration_metrics["success_rate"]
            }
        )

        return integrated_data

    async def assess_data_quality_tool(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of collected data"""
        prompt = f"""
        Assess the quality of this integrated dataset for research purposes.

        Research Question: {data.get('research_question', 'Unknown')}
        Data Summary: {data.get('summary', 'No summary')}

        Provide scores (0-1) for:
        1. Relevance to research question
        2. Completeness of data
        3. Data accuracy/credibility
        4. Data diversity (multiple perspectives)
        5. Timeliness/recency

        Also provide:
        - Overall quality score (0-1)
        - Major limitations
        - Suggestions for improvement
        """

        try:
            response = await self.llm.ainvoke(prompt)
            assessment_text = response.content

            # Parse assessment
            quality_metrics = self._parse_quality_metrics(assessment_text)

            # Add quantitative metrics
            if "integration_metrics" in data:
                metrics = data["integration_metrics"]
                quality_metrics.update({
                    "source_diversity": metrics.get("success_rate", 0),
                    "data_completeness": self._calculate_data_completeness(data),
                    "structured_ratio": self._calculate_structured_ratio(data)
                })

            # Calculate overall quality
            if "relevance" in quality_metrics and "completeness" in quality_metrics:
                overall = (
                        quality_metrics.get("relevance", 0.5) * 0.3 +
                        quality_metrics.get("completeness", 0.5) * 0.2 +
                        quality_metrics.get("accuracy", 0.5) * 0.2 +
                        quality_metrics.get("diversity", 0.5) * 0.15 +
                        quality_metrics.get("timeliness", 0.5) * 0.15
                )
                quality_metrics["overall_quality"] = min(overall, 1.0)

            return quality_metrics

        except Exception as e:
            self.log(f"Quality assessment failed: {e}", "warning")
            return {
                "overall_quality": 0.5,
                "error": str(e)
            }

    async def assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality with multiple methods"""
        # Method 1: LLM assessment
        llm_quality = await self.assess_data_quality_tool(data)

        # Method 2: Quantitative assessment
        quant_quality = self._quantitative_quality_assessment(data)

        # Method 3: Source credibility assessment
        credibility_score = self._assess_source_credibility(data.get("sources", []))

        # Combine assessments
        final_quality = {
            **llm_quality,
            **quant_quality,
            "source_credibility": credibility_score,
            "combined_quality": (
                    llm_quality.get("overall_quality", 0.5) * 0.5 +
                    quant_quality.get("quantitative_score", 0.5) * 0.3 +
                    credibility_score * 0.2
            )
        }

        return final_quality

    async def _search_web_for_sources(self, question: str, domain: str) -> List[Dict[str, Any]]:
        """Search web for data sources"""
        import requests

        sources = []

        # Search queries
        queries = [
            f"{question} dataset",
            f"{domain} data",
            f"{question.split()[0]} {domain} CSV",
            f"{domain} open data",
            f"{question} API"
        ]

        for query in queries[:3]:  # Limit to 3 queries
            try:
                params = {
                    'q': query,
                    'api_key': settings.serper_api_key,
                    'num': 5
                }

                response = requests.get(
                    'https://google.serper.dev/search',
                    params=params,
                    timeout=10
                )
                response.raise_for_status()

                data = response.json()

                for item in data.get('organic', []):
                    title = item.get('title', '')
                    snippet = item.get('snippet', '')
                    link = item.get('link', '')

                    # Determine source type
                    source_type = "web"
                    if any(ext in link.lower() for ext in ['.csv', '.json', '.xlsx', '.xls']):
                        source_type = "dataset"
                    elif 'api' in link.lower() or 'api' in title.lower():
                        source_type = "api"
                    elif any(ext in link.lower() for ext in ['.pdf', '.doc', '.docx']):
                        source_type = "document"

                    sources.append({
                        "type": source_type,
                        "description": f"{title}: {snippet}",
                        "url": link,
                        "accessibility": "open",
                        "relevance": "medium"
                    })

            except Exception as e:
                self.log(f"Web search failed for '{query}': {e}", "warning")
                continue

        return sources

    async def _search_arxiv_sources(self, research_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search ArXiv for relevant papers"""
        question_keywords = research_question.get("question", "").split()[:5]
        query = " ".join(question_keywords)

        try:
            papers = await self.arxiv.search(
                query=query,
                max_results=5,
                sort_by="relevance"
            )

            sources = []
            for paper in papers:
                sources.append({
                    "type": "academic_paper",
                    "description": f"ArXiv paper: {paper.get('title', '')}",
                    "url": paper.get("pdf_url", ""),
                    "accessibility": "open_access",
                    "relevance": "high",
                    "metadata": {
                        "authors": paper.get("authors", []),
                        "published": paper.get("published", ""),
                        "categories": paper.get("categories", [])
                    }
                })

            return sources

        except Exception as e:
            self.log(f"ArXiv search failed: {e}", "warning")
            return []

    async def _search_dataset_sources(self, research_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for public datasets"""
        # This is a simplified version - in practice, you'd use Kaggle API, Google Dataset Search, etc.
        question_text = research_question.get("question", "").lower()

        # Common dataset repositories
        dataset_sites = [
            "https://www.kaggle.com/datasets",
            "https://data.gov",
            "https://data.world",
            "https://registry.opendata.aws",
            "https://datasetsearch.research.google.com"
        ]

        sources = []
        for site in dataset_sites[:2]:  # Limit to 2 sites
            sources.append({
                "type": "dataset_repository",
                "description": f"Search datasets on {site}",
                "url": site,
                "accessibility": "open",
                "relevance": "medium"
            })

        return sources

    def _calculate_source_score(self, source: Dict[str, Any], research_question: Dict[str, Any]) -> float:
        """Calculate relevance score for a source"""
        score = 0.5  # Base score

        # Type bonus
        type_bonus = {
            "academic_paper": 0.3,
            "dataset": 0.4,
            "api": 0.3,
            "government_data": 0.3,
            "web": 0.1
        }

        source_type = source.get("type", "").lower()
        for key, bonus in type_bonus.items():
            if key in source_type:
                score += bonus
                break

        # Accessibility bonus
        accessibility = source.get("accessibility", "").lower()
        if "open" in accessibility or "free" in accessibility:
            score += 0.2

        # URL presence bonus
        if source.get("url"):
            score += 0.1

        # Keyword matching with research question
        question_text = research_question.get("question", "").lower()
        description = source.get("description", "").lower()

        question_words = set(question_text.split())
        description_words = set(description.split())

        overlap = len(question_words.intersection(description_words))
        match_score = overlap / max(len(question_words), 1)

        score += match_score * 0.2

        return min(score, 1.0)

    def _assess_data_quality_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess basic data quality indicators"""
        indicators = {}

        cleaned_data = data.get("cleaned_data", {})

        # Size indicator
        if isinstance(cleaned_data, dict):
            indicators["size"] = len(str(cleaned_data))
        elif isinstance(cleaned_data, str):
            indicators["size"] = len(cleaned_data)
        else:
            indicators["size"] = 0

        # Structure indicator
        if isinstance(cleaned_data, dict) and len(cleaned_data) > 1:
            indicators["structure"] = "high"
        elif isinstance(cleaned_data, str) and len(cleaned_data) > 500:
            indicators["structure"] = "medium"
        else:
            indicators["structure"] = "low"

        # Completeness indicator
        raw_data = data.get("raw_data")
        if raw_data:
            if isinstance(raw_data, str) and len(raw_data) > 1000:
                indicators["completeness"] = "high"
            else:
                indicators["completeness"] = "medium"
        else:
            indicators["completeness"] = "low"

        return indicators

    def _parse_quality_metrics(self, text: str) -> Dict[str, float]:
        """Parse quality metrics from LLM response"""
        metrics = {}

        # Look for scores
        import re

        patterns = {
            "relevance": r'relevance.*?(\d+\.\d+|\d+)',
            "completeness": r'completeness.*?(\d+\.\d+|\d+)',
            "accuracy": r'accuracy.*?(\d+\.\d+|\d+)',
            "diversity": r'diversity.*?(\d+\.\d+|\d+)',
            "timeliness": r'timeliness.*?(\d+\.\d+|\d+)',
            "overall": r'overall.*?quality.*?(\d+\.\d+|\d+)'
        }

        text_lower = text.lower()

        for key, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    metrics[key] = float(match.group(1))
                except:
                    metrics[key] = 0.5

        return metrics

    def _quantitative_quality_assessment(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Quantitative quality assessment"""
        scores = {}

        unified_data = data.get("unified_data", {})

        # Data volume score
        total_size = 0
        for key, value in unified_data.items():
            if isinstance(value, str):
                total_size += len(value)
            elif isinstance(value, (list, dict)):
                total_size += len(str(value))

        volume_score = min(total_size / 10000, 1.0)  # 10KB = perfect score
        scores["volume_score"] = volume_score

        # Diversity score
        source_count = len(data.get("sources", []))
        diversity_score = min(source_count / 5, 1.0)  # 5 sources = perfect score
        scores["diversity_score"] = diversity_score

        # Structure score
        structured_count = sum(1 for key in unified_data.keys()
                               if key in ["tables", "structured", "dataframe"])
        structure_score = min(structured_count / 3, 1.0)  # 3 structured sources = perfect
        scores["structure_score"] = structure_score

        # Overall quantitative score
        scores["quantitative_score"] = (
                volume_score * 0.3 +
                diversity_score * 0.4 +
                structure_score * 0.3
        )

        return scores

    def _assess_source_credibility(self, sources: List[Dict[str, Any]]) -> float:
        """Assess credibility of sources"""
        if not sources:
            return 0.5

        credibility_scores = []
        credible_domains = [
            ".gov", ".edu", ".ac.", ".org", "arxiv", "research", "university",
            "nih.gov", "nsf.gov", "science.gov"
        ]

        for source in sources:
            url = source.get("url", "").lower()
            source_type = source.get("type", "").lower()

            score = 0.5  # Base

            # Domain credibility
            for domain in credible_domains:
                if domain in url:
                    score += 0.3
                    break

            # Type credibility
            if "academic" in source_type or "paper" in source_type:
                score += 0.2
            elif "government" in source_type:
                score += 0.2
            elif "dataset" in source_type and "kaggle" in url:
                score += 0.1

            credibility_scores.append(min(score, 1.0))

        return sum(credibility_scores) / len(credibility_scores)

    def _calculate_data_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        unified_data = data.get("unified_data", {})

        if not unified_data:
            return 0.0

        # Count non-empty data sections
        non_empty = 0
        total = len(unified_data)

        for key, value in unified_data.items():
            if value:
                if isinstance(value, (str, list, dict)):
                    if (isinstance(value, str) and len(value.strip()) > 0) or \
                            (isinstance(value, (list, dict)) and len(value) > 0):
                        non_empty += 1

        return non_empty / max(total, 1)

    def _calculate_structured_ratio(self, data: Dict[str, Any]) -> float:
        """Calculate ratio of structured to unstructured data"""
        unified_data = data.get("unified_data", {})

        structured_keys = ["tables", "structured", "dataframe", "json"]
        unstructured_keys = ["text", "summary", "insights"]

        structured_count = sum(1 for key in unified_data.keys() if key in structured_keys)
        unstructured_count = sum(1 for key in unified_data.keys() if key in unstructured_keys)

        total = structured_count + unstructured_count
        if total == 0:
            return 0.5  # Neutral

        return structured_count / total

    def _identify_data_types(self, data: Dict[str, Any]) -> List[str]:
        """Identify types of data in integrated dataset"""
        unified_data = data.get("unified_data", {})
        data_types = []

        for key, value in unified_data.items():
            if key == "text" and value:
                data_types.append("textual")
            elif key == "tables" and value:
                data_types.append("tabular")
            elif key == "structured" and value:
                data_types.append("structured")
            elif key == "json" and value:
                data_types.append("json")

        return list(set(data_types))

    def _calculate_integration_complexity(self, data_list: List[Dict[str, Any]]) -> float:
        """Calculate complexity of data integration"""
        if len(data_list) <= 1:
            return 0.2

        # Count different data types
        data_types = []
        for data in data_list:
            source_type = data.get("source_info", {}).get("type", "").lower()
            if source_type:
                data_types.append(source_type)

        unique_types = len(set(data_types))

        # Complexity based on number of unique types
        if unique_types == 1:
            return 0.3
        elif unique_types == 2:
            return 0.5
        elif unique_types == 3:
            return 0.7
        else:
            return 0.9

    def _parse_sources_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse sources from LLM response text"""
        sources = []

        lines = text.strip().split('\n')
        current_source = {}

        for line in lines:
            line = line.strip()

            if line.startswith('- Type:'):
                if current_source:
                    sources.append(current_source)
                current_source = {"type": line.replace('- Type:', '').strip()}

            elif line.startswith('Description:'):
                current_source["description"] = line.replace('Description:', '').strip()

            elif line.startswith('URL/Identifier:'):
                current_source["url"] = line.replace('URL/Identifier:', '').strip()

            elif line.startswith('Accessibility:'):
                current_source["accessibility"] = line.replace('Accessibility:', '').strip()

            elif line.startswith('Relevance:'):
                current_source["relevance"] = line.replace('Relevance:', '').strip()

        if current_source:
            sources.append(current_source)

        return sources