import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import re
from datetime import datetime
from loguru import logger
import json


class DataCleaner:
    """Cleans and processes data from various sources"""

    def __init__(self):
        self.nlp = self._initialize_nlp()

    def _initialize_nlp(self):
        """Initialize NLP tools if available"""
        try:
            import spacy
            # Try to load a small model
            return spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy not available, using simple text processing")
            return None

    def extract_from_paper(self, paper_text: str, research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured information from research paper text"""
        cleaned = {
            "text": paper_text[:5000],  # Limit length
            "keywords": self.extract_keywords(paper_text),
            "entities": self.extract_entities(paper_text),
            "sections": self.extract_paper_sections(paper_text),
            "tables": self.extract_tables_from_text(paper_text),
            "references": self.extract_references(paper_text),
            "summary": self.summarize_text(paper_text, research_question)
        }

        # Extract numerical data
        cleaned["numerical_data"] = self.extract_numerical_data(paper_text)

        # Extract methodology mentions
        cleaned["methodology"] = self.extract_methodology(paper_text)

        return cleaned

    def clean_dataframe(self, df: pd.DataFrame, research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and process dataframe"""
        if df.empty:
            return {"error": "Empty dataframe"}

        cleaned = {
            "original_shape": df.shape,
            "cleaned_shape": None,
            "cleaned_df": None,
            "summary_stats": {},
            "column_info": {},
            "cleaning_steps": []
        }

        try:
            # Make a copy
            df_clean = df.copy()
            cleaning_steps = []

            # 1. Handle missing values
            missing_before = df_clean.isnull().sum().sum()
            df_clean = self._handle_missing_values(df_clean)
            missing_after = df_clean.isnull().sum().sum()

            if missing_before > 0:
                cleaning_steps.append(f"Handled missing values: {missing_before} -> {missing_after}")

            # 2. Convert data types
            df_clean = self._convert_data_types(df_clean)
            cleaning_steps.append("Converted data types where possible")

            # 3. Remove duplicates
            duplicates = df_clean.duplicated().sum()
            if duplicates > 0:
                df_clean = df_clean.drop_duplicates()
                cleaning_steps.append(f"Removed {duplicates} duplicate rows")

            # 4. Handle outliers (for numerical columns)
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                outlier_count = self._handle_outliers(df_clean, numerical_cols)
                if outlier_count > 0:
                    cleaning_steps.append(f"Addressed outliers in {len(numerical_cols)} numerical columns")

            # 5. Normalize column names
            df_clean.columns = self._normalize_column_names(df_clean.columns)
            cleaning_steps.append("Normalized column names")

            # 6. Extract relevant columns based on research question
            relevant_cols = self._identify_relevant_columns(df_clean, research_question)
            if relevant_cols:
                df_clean = df_clean[relevant_cols]
                cleaning_steps.append(f"Selected {len(relevant_cols)} relevant columns")

            # Update cleaned data
            cleaned["cleaned_shape"] = df_clean.shape
            cleaned["cleaned_df"] = df_clean.to_dict(orient="records")
            cleaned["summary_stats"] = self._calculate_summary_stats(df_clean)
            cleaned["column_info"] = self._get_column_info(df_clean)
            cleaned["cleaning_steps"] = cleaning_steps

            return cleaned

        except Exception as e:
            logger.error(f"Dataframe cleaning failed: {e}")
            return {"error": str(e), "original_data": df.to_dict(orient="records")}

    def clean_json_data(self, json_data: Union[dict, list], research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and process JSON data"""
        cleaned = {
            "original_structure": str(type(json_data)),
            "cleaned_data": None,
            "extracted_tables": [],
            "key_insights": []
        }

        try:
            if isinstance(json_data, dict):
                # Flatten nested dictionaries
                flattened = self._flatten_dict(json_data)
                cleaned["cleaned_data"] = flattened

                # Try to extract tabular data
                tables = self._extract_tables_from_dict(json_data)
                if tables:
                    cleaned["extracted_tables"] = tables

            elif isinstance(json_data, list):
                # Check if list can be converted to dataframe
                if all(isinstance(item, dict) for item in json_data) and len(json_data) > 0:
                    # Try to create dataframe
                    df = pd.DataFrame(json_data)
                    df_clean = self.clean_dataframe(df, research_question)
                    cleaned["cleaned_data"] = df_clean
                else:
                    cleaned["cleaned_data"] = json_data

            # Extract insights
            json_str = json.dumps(json_data)
            cleaned["key_insights"] = self.extract_insights(json_str, research_question)

            return cleaned

        except Exception as e:
            logger.error(f"JSON cleaning failed: {e}")
            return {"error": str(e), "original_data": json_data}

    def extract_from_html(self, html_text: str, research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured information from HTML"""
        cleaned = {
            "text": self._extract_text_from_html(html_text)[:5000],
            "links": self._extract_links_from_html(html_text),
            "headings": self._extract_headings_from_html(html_text),
            "tables": self._extract_html_tables(html_text),
            "metadata": self._extract_html_metadata(html_text),
            "summary": self.summarize_text(html_text, research_question)
        }

        # Extract entities and keywords
        cleaned["keywords"] = self.extract_keywords(cleaned["text"])
        cleaned["entities"] = self.extract_entities(cleaned["text"])

        return cleaned

    def integrate_text_data(self, text_list: List[str], research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate multiple text sources"""
        if not text_list:
            return {"error": "No text data to integrate"}

        integrated = {
            "combined_text": " ".join(text_list)[:10000],  # Limit length
            "source_count": len(text_list),
            "key_themes": self._extract_key_themes(text_list, research_question),
            "conflicting_info": self._identify_conflicts(text_list),
            "consensus_points": self._identify_consensus(text_list, research_question),
            "integrated_summary": self.summarize_text(" ".join(text_list), research_question)
        }

        # Extract entities across all texts
        all_entities = []
        for text in text_list:
            entities = self.extract_entities(text)
            if entities:
                all_entities.extend(entities)

        integrated["common_entities"] = self._find_common_items(all_entities, threshold=2)

        return integrated

    def integrate_tables(self, tables: List[Any], research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate multiple tables"""
        if not tables:
            return {"error": "No tables to integrate"}

        integrated = {
            "table_count": len(tables),
            "merged_tables": [],
            "schema_alignment": {},
            "integration_issues": []
        }

        try:
            # Convert all tables to dataframes
            dfs = []
            for i, table in enumerate(tables):
                if isinstance(table, pd.DataFrame):
                    dfs.append(table)
                elif isinstance(table, dict) and "data" in table:
                    # Try to create dataframe from dict
                    try:
                        df = pd.DataFrame(table["data"])
                        if "headers" in table:
                            df.columns = table["headers"][:len(df.columns)]
                        dfs.append(df)
                    except:
                        integrated["integration_issues"].append(f"Table {i}: Could not convert to dataframe")
                elif isinstance(table, list) and len(table) > 0:
                    # Try to create dataframe from list
                    try:
                        df = pd.DataFrame(table)
                        dfs.append(df)
                    except:
                        integrated["integration_issues"].append(f"Table {i}: Could not convert list to dataframe")

            if not dfs:
                return integrated

            # Try to merge dataframes
            if len(dfs) == 1:
                integrated["merged_tables"] = dfs[0].to_dict(orient="records")
                integrated["schema_alignment"] = {"single_table": True}
            else:
                # Find common columns
                all_columns = [set(df.columns) for df in dfs]
                common_columns = set.intersection(*all_columns) if all_columns else set()

                if common_columns:
                    # Merge on common columns
                    merged = pd.concat(dfs, ignore_index=True)
                    integrated["merged_tables"] = merged[list(common_columns)].to_dict(orient="records")
                    integrated["schema_alignment"] = {
                        "common_columns": list(common_columns),
                        "total_rows": len(merged)
                    }
                else:
                    # Can't merge, keep as separate
                    integrated["merged_tables"] = [df.to_dict(orient="records") for df in dfs]
                    integrated["schema_alignment"] = {
                        "no_common_columns": True,
                        "table_shapes": [df.shape for df in dfs]
                    }

            return integrated

        except Exception as e:
            logger.error(f"Table integration failed: {e}")
            integrated["integration_issues"].append(f"Integration error: {str(e)}")
            return integrated

    def integrate_structured_data(self, structured_list: List[Dict[str, Any]],
                                  research_question: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate multiple structured data sources"""
        if not structured_list:
            return {"error": "No structured data to integrate"}

        integrated = {
            "source_count": len(structured_list),
            "integrated_dict": {},
            "key_value_alignment": {},
            "conflicts": []
        }

        # Collect all keys
        all_keys = set()
        for data in structured_list:
            if isinstance(data, dict):
                all_keys.update(data.keys())

        # Integrate values by key
        for key in all_keys:
            values = []
            for data in structured_list:
                if isinstance(data, dict) and key in data:
                    values.append(data[key])

            if values:
                # Check for conflicts
                unique_values = set(str(v) for v in values)
                if len(unique_values) > 1:
                    integrated["conflicts"].append({
                        "key": key,
                        "values": list(unique_values)[:5]  # Limit display
                    })

                # Use most common value or average if numeric
                integrated["integrated_dict"][key] = self._resolve_values(values)

        # Calculate alignment score
        if all_keys:
            conflict_count = len(integrated["conflicts"])
            integrated["key_value_alignment"] = {
                "total_keys": len(all_keys),
                "conflict_keys": conflict_count,
                "alignment_score": 1.0 - (conflict_count / len(all_keys))
            }

        return integrated

    def extract_insights(self, text: str, research_question: Dict[str, Any]) -> List[str]:
        """Extract key insights from text"""
        insights = []

        # Simple insight extraction based on patterns
        question_keywords = research_question.get("question", "").lower().split()

        # Look for sentences that contain question keywords
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 500:
                continue

            sentence_lower = sentence.lower()

            # Check for insight indicators
            insight_indicators = [
                "finding", "result", "conclusion", "shows that", "indicates that",
                "suggests that", "demonstrates", "evidence for", "support for",
                "contrary to", "unexpected", "surprisingly", "important", "significant"
            ]

            has_indicator = any(indicator in sentence_lower for indicator in insight_indicators)
            has_keyword = any(keyword in sentence_lower for keyword in question_keywords)

            if has_indicator and has_keyword:
                insights.append(sentence)

        # Limit number of insights
        return insights[:10]

    def create_data_summary(self, integrated_data: Dict[str, Any],
                            research_question: Dict[str, Any]) -> str:
        """Create summary of integrated data"""
        summary_parts = []

        # Basic info
        if "source_count" in integrated_data:
            summary_parts.append(f"Integrated data from {integrated_data['source_count']} sources.")

        # Text data summary
        if "text" in integrated_data.get("unified_data", {}):
            text_data = integrated_data["unified_data"]["text"]
            if isinstance(text_data, dict) and "combined_text" in text_data:
                text_length = len(text_data["combined_text"])
                summary_parts.append(f"Combined text data: {text_length} characters.")

        # Table data summary
        if "tables" in integrated_data.get("unified_data", {}):
            table_data = integrated_data["unified_data"]["tables"]
            if isinstance(table_data, dict) and "table_count" in table_data:
                summary_parts.append(f"Integrated {table_data['table_count']} tables.")

        # Key insights
        if "key_insights" in integrated_data:
            insights = integrated_data["key_insights"]
            if insights and len(insights) > 0:
                summary_parts.append(f"Extracted {len(insights)} key insights.")

        # Conflicts
        if "conflicts" in integrated_data.get("unified_data", {}):
            conflicts = integrated_data["unified_data"].get("conflicts", [])
            if conflicts:
                summary_parts.append(f"Found {len(conflicts)} data conflicts requiring resolution.")

        # If no specific info, create generic summary
        if not summary_parts:
            summary_parts.append("Data successfully integrated from multiple sources.")
            summary_parts.append(f"Relevant to research question: {research_question.get('question', '')[:100]}...")

        return " ".join(summary_parts)

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text"""
        if not text or len(text) < 50:
            return []

        # Simple keyword extraction
        words = re.findall(r'\b\w{4,}\b', text.lower())

        # Remove common stop words
        stop_words = {
            'that', 'with', 'this', 'from', 'have', 'which', 'their', 'would',
            'there', 'about', 'when', 'were', 'they', 'some', 'what', 'than',
            'other', 'into', 'could', 'more', 'also', 'where', 'most', 'these'
        }

        filtered = [word for word in words if word not in stop_words]

        # Count frequencies
        from collections import Counter
        word_counts = Counter(filtered)

        # Get most common
        keywords = [word for word, _ in word_counts.most_common(top_n)]

        return keywords

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        if not text or len(text) < 100:
            return []

        if self.nlp:
            try:
                doc = self.nlp(text[:5000])  # Limit length for performance
                entities = []

                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PERSON", "PRODUCT", "GPE", "LOC", "EVENT", "WORK_OF_ART"]:
                        entities.append(ent.text)

                return list(set(entities))[:20]  # Limit to 20 unique entities
            except:
                pass

        # Fallback: extract capitalized phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))[:10]

    def summarize_text(self, text: str, research_question: Dict[str, Any]) -> str:
        """Create summary of text relevant to research question"""
        if not text or len(text) < 100:
            return "No text to summarize"

        # Extract sentences containing question keywords
        question_words = research_question.get("question", "").lower().split()
        sentences = re.split(r'[.!?]+', text)

        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            sentence_lower = sentence.lower()
            # Check if sentence contains any question keywords
            if any(keyword in sentence_lower for keyword in question_words):
                relevant_sentences.append(sentence)

        # If no relevant sentences, use first few sentences
        if not relevant_sentences:
            relevant_sentences = [s for s in sentences if len(s) > 20][:3]

        # Create summary
        summary = " ".join(relevant_sentences[:5])  # Use up to 5 sentences
        if len(summary) > 500:
            summary = summary[:497] + "..."

        return summary if summary else "Summary not available"

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe"""
        df_clean = df.copy()

        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_value = df_clean[col].mode()
                if not mode_value.empty:
                    df_clean[col] = df_clean[col].fillna(mode_value.iloc[0])
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")

        return df_clean

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types where appropriate"""
        df_clean = df.copy()

        for col in df_clean.columns:
            # Try to convert to numeric
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass

            # Try to convert to datetime
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
            except:
                pass

        return df_clean

    def _handle_outliers(self, df: pd.DataFrame, numerical_cols: List[str]) -> int:
        """Handle outliers in numerical columns"""
        outlier_count = 0

        for col in numerical_cols:
            try:
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Count outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count += len(outliers)

                # Cap outliers (optional - for now just count)
                # df[col] = df[col].clip(lower_bound, upper_bound)

            except:
                continue

        return outlier_count

    def _normalize_column_names(self, columns: List[str]) -> List[str]:
        """Normalize column names"""
        normalized = []

        for col in columns:
            # Convert to lowercase
            col_norm = str(col).lower()

            # Replace spaces and special characters
            col_norm = re.sub(r'[^\w]', '_', col_norm)

            # Remove multiple underscores
            col_norm = re.sub(r'_+', '_', col_norm)

            # Remove leading/trailing underscores
            col_norm = col_norm.strip('_')

            normalized.append(col_norm)

        return normalized

    def _identify_relevant_columns(self, df: pd.DataFrame, research_question: Dict[str, Any]) -> List[str]:
        """Identify columns relevant to research question"""
        question_text = research_question.get("question", "").lower()
        question_words = set(re.findall(r'\b\w{4,}\b', question_text))

        relevant_cols = []

        for col in df.columns:
            col_lower = str(col).lower()

            # Check for keyword matches
            if any(word in col_lower for word in question_words):
                relevant_cols.append(col)

        # If no matches found, return all columns
        return relevant_cols if relevant_cols else list(df.columns)

    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for dataframe"""
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "data_types": {},
            "missing_values": {},
            "numerical_stats": {}
        }

        # Data types
        stats["data_types"] = df.dtypes.astype(str).to_dict()

        # Missing values
        stats["missing_values"] = df.isnull().sum().to_dict()

        # Numerical statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            stats["numerical_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median())
            }

        return stats

    def _get_column_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get information about each column"""
        column_info = {}

        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
                "sample_values": []
            }

            # Get sample values
            if df[col].nunique() > 0:
                sample = df[col].dropna().unique()[:5]
                col_info["sample_values"] = [str(v) for v in sample]

            column_info[col] = col_info

        return column_info

    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Handle lists by converting to string or taking first element
                if v and isinstance(v[0], dict):
                    # List of dicts - take first dict
                    items.extend(self._flatten_dict(v[0], new_key, sep=sep).items())
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))

        return dict(items)

    def _extract_tables_from_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tabular data from dictionary"""
        tables = []

        def _find_tables(obj, path=""):
            if isinstance(obj, list):
                # Check if list contains dictionaries with same keys (tabular data)
                if len(obj) > 0 and all(isinstance(item, dict) for item in obj):
                    # Check if all dicts have same keys
                    if len(obj) > 1:
                        keys = set(obj[0].keys())
                        if all(set(item.keys()) == keys for item in obj[1:]):
                            tables.append({
                                "path": path,
                                "rows": len(obj),
                                "columns": len(keys),
                                "data": obj[:10]  # Limit to first 10 rows
                            })

            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    _find_tables(value, new_path)

        _find_tables(data)
        return tables

    def _extract_text_from_html(self, html_text: str) -> str:
        """Extract clean text from HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text(separator=' ', strip=True)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)

            return text

        except:
            # Fallback: simple regex to remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', html_text)
            text = re.sub(r'\s+', ' ', text)
            return text

    def _extract_links_from_html(self, html_text: str) -> List[Dict[str, str]]:
        """Extract links from HTML"""
        links = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')

            for a in soup.find_all('a', href=True):
                links.append({
                    "text": a.get_text(strip=True),
                    "url": a['href']
                })

        except:
            # Simple regex extraction
            import re
            link_pattern = r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>([^<]*)</a>'
            matches = re.findall(link_pattern, html_text, re.IGNORECASE)

            for url, text in matches:
                links.append({
                    "text": text.strip(),
                    "url": url
                })

        return links[:20]  # Limit to 20 links

    def _extract_headings_from_html(self, html_text: str) -> List[Dict[str, str]]:
        """Extract headings from HTML"""
        headings = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')

            for i in range(1, 7):
                for h in soup.find_all(f'h{i}'):
                    headings.append({
                        "level": i,
                        "text": h.get_text(strip=True)
                    })

        except:
            # Simple regex extraction
            import re
            for i in range(1, 7):
                pattern = f'<h{i}[^>]*>([^<]+)</h{i}>'
                matches = re.findall(pattern, html_text, re.IGNORECASE)

                for text in matches:
                    headings.append({
                        "level": i,
                        "text": text.strip()
                    })

        return headings[:10]  # Limit to 10 headings

    def _extract_html_tables(self, html_text: str) -> List[Dict[str, Any]]:
        """Extract tables from HTML"""
        tables = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')

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
                        "data": table_data,
                        "rows": len(table_data),
                        "columns": len(headers) if headers else len(table_data[0]) if table_data else 0
                    })

        except:
            pass

        return tables[:5]  # Limit to 5 tables

    def _extract_html_metadata(self, html_text: str) -> Dict[str, str]:
        """Extract metadata from HTML"""
        metadata = {}

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')

            # Title
            title = soup.title
            if title:
                metadata["title"] = title.get_text(strip=True)

            # Meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    metadata[name] = content

        except:
            pass

        return metadata

    def _extract_key_themes(self, text_list: List[str], research_question: Dict[str, Any]) -> List[str]:
        """Extract key themes from multiple texts"""
        # Combine all texts
        combined_text = " ".join(text_list)

        # Extract frequent bigrams
        words = re.findall(r'\b\w{3,}\b', combined_text.lower())

        # Create bigrams
        bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]

        # Count frequencies
        from collections import Counter
        bigram_counts = Counter(bigrams)

        # Get most common bigrams as themes
        themes = [bigram for bigram, _ in bigram_counts.most_common(5)]

        return themes

    def _identify_conflicts(self, text_list: List[str]) -> List[Dict[str, Any]]:
        """Identify conflicts between texts"""
        conflicts = []

        if len(text_list) < 2:
            return conflicts

        # Simple conflict detection based on contradictory phrases
        contradictory_phrases = [
            ("is", "is not"),
            ("does", "does not"),
            ("has", "has not"),
            ("shows", "does not show"),
            ("proves", "disproves"),
            ("supports", "contradicts"),
            ("increases", "decreases"),
            ("positive", "negative"),
            ("significant", "not significant"),
            ("effective", "ineffective")
        ]

        # Check each pair of texts
        for i in range(len(text_list)):
            for j in range(i + 1, len(text_list)):
                text1 = text_list[i].lower()
                text2 = text_list[j].lower()

                for positive, negative in contradictory_phrases:
                    if positive in text1 and negative in text2:
                        conflicts.append({
                            "source1_idx": i,
                            "source2_idx": j,
                            "conflict": f"'{positive}' vs '{negative}'"
                        })
                    elif positive in text2 and negative in text1:
                        conflicts.append({
                            "source1_idx": i,
                            "source2_idx": j,
                            "conflict": f"'{negative}' vs '{positive}'"
                        })

        return conflicts[:5]  # Limit to 5 conflicts

    def _identify_consensus(self, text_list: List[str], research_question: Dict[str, Any]) -> List[str]:
        """Identify consensus points across texts"""
        consensus = []

        if not text_list:
            return consensus

        # Extract key terms from research question
        question_terms = set(re.findall(r'\b\w{4,}\b', research_question.get("question", "").lower()))

        # Find terms mentioned in all texts
        all_terms = []
        for text in text_list:
            terms = set(re.findall(r'\b\w{4,}\b', text.lower()))
            # Filter by question terms if available
            if question_terms:
                terms = terms.intersection(question_terms)
            all_terms.append(terms)

        # Find intersection
        if all_terms:
            common_terms = set.intersection(*all_terms)
            consensus = list(common_terms)[:10]  # Limit to 10 terms

        return consensus

    def _find_common_items(self, items: List[str], threshold: int = 2) -> List[str]:
        """Find items that appear multiple times"""
        from collections import Counter
        item_counts = Counter(items)

        common = [item for item, count in item_counts.items() if count >= threshold]
        return common[:10]  # Limit to 10 items

    def _resolve_values(self, values: List[Any]) -> Any:
        """Resolve conflicting values"""
        if not values:
            return None

        # If all values are numeric, return average
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except:
                pass

        if len(numeric_values) == len(values):
            return sum(numeric_values) / len(numeric_values)

        # Otherwise, return most common value
        from collections import Counter
        str_values = [str(v) for v in values]
        most_common = Counter(str_values).most_common(1)

        if most_common:
            return most_common[0][0]

        return values[0]

    def extract_paper_sections(self, paper_text: str) -> Dict[str, str]:
        """Extract sections from research paper"""
        sections = {}

        # Common section headers
        section_patterns = {
            "abstract": r'\babstract\b',
            "introduction": r'\bintroduction\b',
            "methods": r'\bmethods?\b',
            "results": r'\bresults\b',
            "discussion": r'\bdiscussion\b',
            "conclusion": r'\bconclusion\b',
            "references": r'\breferences\b'
        }

        # Split by common section markers
        lines = paper_text.split('\n')

        current_section = None
        section_content = []

        for line in lines:
            line_lower = line.strip().lower()

            # Check for section headers
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line_lower) and len(line.strip()) < 100:
                    # Save previous section
                    if current_section and section_content:
                        sections[current_section] = '\n'.join(section_content)

                    # Start new section
                    current_section = section_name
                    section_content = []
                    break
            else:
                # Add to current section
                if current_section and line.strip():
                    section_content.append(line)

        # Save last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)

        return sections

    def extract_tables_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract table-like structures from text"""
        tables = []

        # Look for tab-separated or pipe-separated data
        lines = text.split('\n')

        current_table = None
        table_data = []

        for line in lines:
            # Check if line looks like table data
            if '|' in line or '\t' in line:
                if not current_table:
                    current_table = []

                # Parse line
                if '|' in line:
                    parts = [part.strip() for part in line.split('|')]
                else:
                    parts = [part.strip() for part in line.split('\t')]

                current_table.append(parts)
            elif current_table:
                # End of table
                if len(current_table) > 1:  # At least header and one data row
                    headers = current_table[0]
                    data = current_table[1:]

                    tables.append({
                        "headers": headers,
                        "data": data,
                        "rows": len(data),
                        "columns": len(headers)
                    })

                current_table = None

        return tables[:3]  # Limit to 3 tables

    def extract_references(self, text: str) -> List[str]:
        """Extract references from text"""
        references = []

        # Look for common reference patterns
        patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\([A-Z][a-z]+ et al\. \d{4}\)',  # (Author et al. 2024)
            r'\b\d{4}[a-z]?\b',  # Year references
            r'arXiv:\d+\.\d+',  # ArXiv references
            r'doi:\s*[^\s]+',  # DOI references
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)

        return list(set(references))[:10]  # Limit to 10 unique references

    def extract_numerical_data(self, text: str) -> Dict[str, List[float]]:
        """Extract numerical data from text"""
        numerical = {
            "percentages": [],
            "measurements": [],
            "statistics": [],
            "years": []
        }

        # Extract percentages
        percentages = re.findall(r'(\d+\.?\d*)%', text)
        numerical["percentages"] = [float(p) for p in percentages[:10]]

        # Extract measurements with units
        measurements = re.findall(r'(\d+\.?\d*)\s*(mm|cm|m|km|g|kg|ml|l|s|min|hr|day|week|month|year)s?\b', text,
                                  re.IGNORECASE)
        numerical["measurements"] = [f"{val}{unit}" for val, unit in measurements[:10]]

        # Extract statistics (p-values, t-values, etc.)
        stats = re.findall(r'(p|t|F|r|R²)\s*[=<>]\s*(\d+\.?\d*)', text, re.IGNORECASE)
        numerical["statistics"] = [f"{stat}={val}" for stat, val in stats[:10]]

        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        numerical["years"] = [int(year) for year in years[:10]]

        return numerical

    def extract_methodology(self, text: str) -> List[str]:
        """Extract methodology mentions from text"""
        methodology_terms = [
            "method", "experiment", "study", "trial", "survey", "analysis",
            "procedure", "protocol", "design", "approach", "technique",
            "randomized", "controlled", "blind", "double-blind", "cohort",
            "longitudinal", "cross-sectional", "qualitative", "quantitative",
            "statistical", "regression", "ANOVA", "t-test", "chi-square"
        ]

        methodology = []
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in methodology_terms):
                methodology.append(sentence.strip())

        return methodology[:5]  # Limit to 5 methodology mentions