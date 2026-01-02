import arxiv
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from backend.utils import retry_async


class ArxivClient:
    """Client for interacting with ArXiv API"""

    def __init__(self):
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )

    async def search(self, query: str, max_results: int = 10,
                     sort_by: str = "relevance",
                     sort_order: str = "descending") -> List[Dict[str, Any]]:
        """Search ArXiv for papers"""

        async def _search():
            try:
                # Configure search
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate if sort_by == "submittedDate" else arxiv.SortCriterion.Relevance,
                    sort_order=arxiv.SortOrder.Descending if sort_order == "descending" else arxiv.SortOrder.Ascending
                )

                papers = []
                for paper in self.client.results(search):
                    papers.append({
                        "id": paper.entry_id,
                        "title": paper.title,
                        "authors": [str(author) for author in paper.authors],
                        "summary": paper.summary,
                        "published": paper.published.isoformat(),
                        "updated": paper.updated.isoformat() if paper.updated else None,
                        "categories": paper.categories,
                        "pdf_url": paper.pdf_url,
                        "doi": paper.doi,
                        "comment": paper.comment,
                        "journal_ref": paper.journal_ref,
                        "primary_category": paper.primary_category
                    })

                    if len(papers) >= max_results:
                        break

                logger.info(f"Found {len(papers)} papers for query: {query}")
                return papers

            except Exception as e:
                logger.error(f"ArXiv search failed: {e}")
                raise

        return await retry_async(_search, max_retries=3)

    async def get_recent_papers(self, category: str = None,
                                days: int = 30,
                                max_results: int = 20) -> List[Dict[str, Any]]:
        """Get recent papers from ArXiv"""
        date_cutoff = datetime.now() - timedelta(days=days)
        date_str = date_cutoff.strftime("%Y%m%d")

        query = f"submittedDate:[{date_str} TO *]"
        if category:
            query = f"cat:{category} AND {query}"

        return await self.search(query, max_results, "submittedDate", "descending")

    async def get_new_categories(self, days: int = 90) -> List[str]:
        """Get new or trending categories on ArXiv"""
        try:
            # Get recent papers
            recent_papers = await self.get_recent_papers(days=days, max_results=50)

            # Extract and count categories
            category_counts = {}
            for paper in recent_papers:
                categories = paper.get("categories", [])
                for category in categories:
                    # Clean category (remove version numbers)
                    base_category = category.split('.')[0] if '.' in category else category
                    category_counts[base_category] = category_counts.get(base_category, 0) + 1

            # Find categories with recent growth
            # For simplicity, return most frequent recent categories
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

            # Filter for potentially new categories (not in standard set)
            standard_categories = {
                "cs", "math", "physics", "q-bio", "q-fin", "stat", "eess", "econ"
            }

            new_categories = []
            for category, count in sorted_categories[:10]:  # Top 10
                if category not in standard_categories and count >= 3:
                    new_categories.append(category)

            return new_categories[:5]  # Return top 5

        except Exception as e:
            logger.error(f"Failed to get new categories: {e}")
            return []

    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get paper by ArXiv ID"""

        async def _get_paper():
            try:
                search = arxiv.Search(id_list=[paper_id.replace("arXiv:", "")])
                paper = next(self.client.results(search), None)

                if paper:
                    return {
                        "id": paper.entry_id,
                        "title": paper.title,
                        "authors": [str(author) for author in paper.authors],
                        "summary": paper.summary,
                        "published": paper.published.isoformat(),
                        "updated": paper.updated.isoformat() if paper.updated else None,
                        "categories": paper.categories,
                        "pdf_url": paper.pdf_url,
                        "doi": paper.doi,
                        "comment": paper.comment,
                        "journal_ref": paper.journal_ref,
                        "primary_category": paper.primary_category
                    }
                return None

            except Exception as e:
                logger.error(f"Failed to get paper {paper_id}: {e}")
                raise

        return await retry_async(_get_paper, max_retries=3)

    async def get_related_papers(self, paper_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get papers related to a given paper"""
        try:
            # Get the target paper
            target_paper = await self.get_paper_by_id(paper_id)
            if not target_paper:
                return []

            # Use paper's title and categories for related search
            title = target_paper.get("title", "")
            categories = target_paper.get("categories", [])

            # Extract key terms from title
            import re
            title_words = re.findall(r'\b\w{4,}\b', title)
            key_terms = title_words[:5]  # Use first 5 longer words

            # Build query
            query_terms = key_terms + categories[:3]
            query = " OR ".join(query_terms)

            # Search for related papers
            related = await self.search(query, max_results, "relevance", "descending")

            # Remove the target paper from results
            related = [p for p in related if p["id"] != paper_id]

            return related[:max_results]

        except Exception as e:
            logger.error(f"Failed to get related papers for {paper_id}: {e}")
            return []

    async def get_citation_network(self, paper_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get citation network for a paper (simplified)"""
        # Note: ArXiv API doesn't provide citation data directly
        # This is a simplified version that uses related papers
        try:
            network = {
                "paper": await self.get_paper_by_id(paper_id),
                "cited_by": [],
                "cites": []
            }

            if depth > 0 and network["paper"]:
                # Get papers that cite this one (simulated via related papers)
                title = network["paper"].get("title", "")
                query = f'ti:"{title}"'

                citing_papers = await self.search(query, 10, "submittedDate", "descending")
                network["cited_by"] = [p for p in citing_papers if p["id"] != paper_id][:5]

                # Get papers this paper might cite (simulated via references in summary)
                # This is very simplified
                summary = network["paper"].get("summary", "")

                # Extract potential references (papers mentioned with arXiv IDs)
                import re
                arxiv_refs = re.findall(r'arXiv:\d+\.\d+v?\d*', summary)

                for ref_id in arxiv_refs[:3]:  # Limit to 3
                    try:
                        ref_paper = await self.get_paper_by_id(ref_id)
                        if ref_paper:
                            network["cites"].append(ref_paper)
                    except:
                        continue

            return network

        except Exception as e:
            logger.error(f"Failed to get citation network for {paper_id}: {e}")
            return {"paper": None, "cited_by": [], "cites": []}