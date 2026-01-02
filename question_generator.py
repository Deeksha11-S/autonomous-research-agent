from typing import List, Dict, Any
import asyncio
from datetime import datetime
from loguru import logger
import re

from backend.agents.base_agent import BaseAgent, AgentMessage
from backend.config import settings


class QuestionGeneratorAgent(BaseAgent):
    """Generates novel research questions from domain information"""

    def __init__(self, agent_id: str = "qgen_001"):
        capabilities = [
            "question_generation", "novelty_assessment",
            "synthesis", "creativity", "feasibility_analysis"
        ]
        super().__init__(agent_id, "question_generator", capabilities)
        self.setup_tools()

    def setup_tools(self):
        """Setup question generation tools"""
        self.add_tool(
            name="generate_questions",
            tool_func=self.generate_questions_tool,
            description="Generate novel research questions from domain information"
        )

        self.add_tool(
            name="assess_novelty",
            tool_func=self.assess_novelty_tool,
            description="Assess novelty score of research questions"
        )

        self.add_tool(
            name="check_feasibility",
            tool_func=self.check_feasibility_tool,
            description="Check feasibility of research questions"
        )

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process question generation request"""
        self.log(f"Received message: {message.message_type}")

        if message.message_type == "command" and message.content.get("action") == "generate_questions":
            try:
                domain_info = message.content.get("domain_info", {})
                if not domain_info:
                    raise ValueError("No domain information provided")

                questions = await self.generate_questions(domain_info)

                # Assess novelty and feasibility
                evaluated_questions = []
                for question in questions:
                    novelty_score = await self.assess_novelty(question, domain_info)
                    feasibility_score = await self.check_feasibility(question)

                    evaluated_questions.append({
                        **question,
                        "novelty_score": novelty_score,
                        "feasibility_score": feasibility_score,
                        "overall_score": (novelty_score * 0.6) + (feasibility_score * 0.4)
                    })

                # Sort by overall score
                evaluated_questions.sort(key=lambda x: x["overall_score"], reverse=True)

                return AgentMessage(
                    sender=self.agent_id,
                    recipient=message.sender,
                    content={
                        "questions": evaluated_questions[:5],  # Top 5 questions
                        "domain": domain_info.get("name", "Unknown"),
                        "generation_date": datetime.now().isoformat()
                    },
                    message_type="result",
                    confidence=self.calculate_questions_confidence(evaluated_questions)
                )

            except Exception as e:
                self.log(f"Question generation failed: {e}", "error")
                raise

        return await super().process(message)

    async def generate_questions_tool(self, domain_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate research questions from domain information"""
        prompt = f"""
        Based on the following emerging scientific domain information, generate 5-7 novel research questions.
        The questions should:
        1. Require synthesis of information from multiple sources
        2. Not be directly searchable/answerable with a simple Google search
        3. Be specific and testable
        4. Push the boundaries of the field

        Domain Information:
        Name: {domain_info.get('name', 'Unknown')}
        Description: {domain_info.get('description', 'No description')}
        Evidence: {domain_info.get('evidence', [])[:3]}

        Format each question as:
        - Question: [The research question]
        - Explanation: [Why this is novel and important]
        - Required Data: [What data would be needed to answer this]
        - Potential Impact: [Potential impact on the field]

        Generate questions now:
        """

        try:
            response = await self.llm.ainvoke(prompt)
            questions_text = response.content

            # Parse the generated questions
            questions = self._parse_questions_from_text(questions_text)

            # Add metadata
            for i, question in enumerate(questions):
                question["id"] = f"q_{i + 1}_{datetime.now().strftime('%Y%m%d')}"
                question["domain"] = domain_info.get("name")
                question["generated_at"] = datetime.now().isoformat()

            self.log(f"Generated {len(questions)} questions")
            return questions

        except Exception as e:
            self.log(f"LLM question generation failed: {e}", "error")
            return []

    async def generate_questions(self, domain_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Main method to generate questions"""
        questions = await self.generate_questions_tool(domain_info)

        # Store in memory
        for question in questions:
            self.memory.store(
                f"Research question: {question.get('question', '')}",
                metadata={
                    "type": "question",
                    "domain": domain_info.get("name"),
                    "novelty": question.get("novelty_score", 0),
                    "feasibility": question.get("feasibility_score", 0)
                }
            )

        return questions

    async def assess_novelty_tool(self, question: Dict[str, Any], domain_info: Dict[str, Any]) -> float:
        """Assess novelty of a research question (0-1 scale)"""
        prompt = f"""
        Assess the novelty of this research question on a scale of 0-1.

        Domain: {domain_info.get('name', 'Unknown')}
        Question: {question.get('question', '')}
        Explanation: {question.get('explanation', '')}

        Consider:
        1. How original is this question? (0.3 weight)
        2. Does it combine concepts in new ways? (0.3 weight)
        3. Is it addressing a genuine gap in knowledge? (0.2 weight)
        4. Could this question lead to paradigm shifts? (0.2 weight)

        Provide only a single number between 0 and 1 with one decimal place.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            novelty_text = response.content.strip()

            # Extract number
            import re
            match = re.search(r'(\d+\.\d+|\d+)', novelty_text)
            if match:
                novelty = float(match.group(1))
                return min(max(novelty, 0.0), 1.0)

            return 0.5  # Default

        except Exception as e:
            self.log(f"Novelty assessment failed: {e}", "warning")
            return 0.5

    async def assess_novelty(self, question: Dict[str, Any], domain_info: Dict[str, Any]) -> float:
        """Assess novelty with multiple methods"""
        # Method 1: LLM assessment
        llm_novelty = await self.assess_novelty_tool(question, domain_info)

        # Method 2: Check if question contains novel combinations
        question_text = question.get("question", "").lower()
        domain_name = domain_info.get("name", "").lower()

        # Look for interdisciplinary terms
        interdisciplinary_terms = [
            "cross-disciplinary", "interdisciplinary", "multidisciplinary",
            "integration of", "combined with", "fusion of", "bridge between"
        ]

        has_interdisciplinary = any(term in question_text for term in interdisciplinary_terms)

        # Method 3: Check question complexity
        words = question_text.split()
        complexity_score = min(len(words) / 100, 1.0)  # Longer questions might be more complex

        # Combine scores
        final_novelty = (llm_novelty * 0.5) + (has_interdisciplinary * 0.3) + (complexity_score * 0.2)

        return min(final_novelty, 1.0)

    async def check_feasibility_tool(self, question: Dict[str, Any]) -> float:
        """Check feasibility of answering the question (0-1 scale)"""
        prompt = f"""
        Assess the feasibility of answering this research question on a scale of 0-1.

        Question: {question.get('question', '')}
        Required Data: {question.get('required_data', 'Not specified')}

        Consider:
        1. Are the required data sources accessible? (0.3 weight)
        2. Are the required methods/tools available? (0.3 weight)
        3. Could this be answered within reasonable time/resources? (0.2 weight)
        4. Are ethical/regulatory constraints manageable? (0.2 weight)

        Provide only a single number between 0 and 1 with one decimal place.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            feasibility_text = response.content.strip()

            # Extract number
            match = re.search(r'(\d+\.\d+|\d+)', feasibility_text)
            if match:
                feasibility = float(match.group(1))
                return min(max(feasibility, 0.0), 1.0)

            return 0.5  # Default

        except Exception as e:
            self.log(f"Feasibility assessment failed: {e}", "warning")
            return 0.5

    async def check_feasibility(self, question: Dict[str, Any]) -> float:
        """Check feasibility with multiple methods"""
        # Method 1: LLM assessment
        llm_feasibility = await self.check_feasibility_tool(question)

        # Method 2: Check data requirements
        required_data = question.get("required_data", "").lower()

        data_availability_score = 0.5  # Default

        # Check for common data sources
        available_sources = ["public dataset", "api", "open data", "government data", "academic data"]
        unavailable_sources = ["proprietary", "confidential", "restricted", "classified"]

        available_count = sum(1 for source in available_sources if source in required_data)
        unavailable_count = sum(1 for source in unavailable_sources if source in required_data)

        if unavailable_count > 0:
            data_availability_score = 0.2
        elif available_count > 0:
            data_availability_score = 0.8

        # Method 3: Check time/resources
        question_text = question.get("question", "").lower()

        # Words indicating complexity
        complex_indicators = ["long-term", "decades", "years", "extensive", "comprehensive"]
        simple_indicators = ["quick", "rapid", "simple", "straightforward", "pilot"]

        complex_count = sum(1 for indicator in complex_indicators if indicator in question_text)
        simple_count = sum(1 for indicator in simple_indicators if indicator in question_text)

        if complex_count > simple_count:
            time_feasibility = 0.3
        elif simple_count > complex_count:
            time_feasibility = 0.8
        else:
            time_feasibility = 0.5

        # Combine scores
        final_feasibility = (llm_feasibility * 0.4) + (data_availability_score * 0.4) + (time_feasibility * 0.2)

        return min(final_feasibility, 1.0)

    def _parse_questions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse generated questions from LLM response"""
        questions = []

        # Split by question markers
        question_blocks = re.split(r'\n-+\s*\n|\n\d+[\.\)]\s+', text)

        for block in question_blocks:
            if not block.strip():
                continue

            # Extract components
            question_match = re.search(r'Question:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            explanation_match = re.search(r'Explanation:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            data_match = re.search(r'Required Data:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            impact_match = re.search(r'Potential Impact:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)

            if question_match:
                question = {
                    "question": question_match.group(1).strip(),
                    "explanation": explanation_match.group(1).strip() if explanation_match else "",
                    "required_data": data_match.group(1).strip() if data_match else "",
                    "potential_impact": impact_match.group(1).strip() if impact_match else ""
                }
                questions.append(question)

        # If parsing failed, try alternative approach
        if not questions:
            lines = text.strip().split('\n')
            current_question = {}

            for line in lines:
                line = line.strip()
                if line.startswith('Question:'):
                    if current_question:
                        questions.append(current_question)
                    current_question = {"question": line.replace('Question:', '').strip()}
                elif line.startswith('Explanation:'):
                    current_question["explanation"] = line.replace('Explanation:', '').strip()
                elif line.startswith('Required Data:'):
                    current_question["required_data"] = line.replace('Required Data:', '').strip()
                elif line.startswith('Potential Impact:'):
                    current_question["potential_impact"] = line.replace('Potential Impact:', '').strip()

            if current_question:
                questions.append(current_question)

        return questions

    def calculate_questions_confidence(self, questions: List[Dict[str, Any]]) -> float:
        """Calculate confidence in generated questions"""
        if not questions:
            return 0.0

        # Average overall score
        avg_score = sum(q.get("overall_score", 0) for q in questions) / len(questions)

        # Diversity of questions (based on uniqueness)
        unique_keywords = set()
        for question in questions:
            q_text = question.get("question", "").lower()
            keywords = set(re.findall(r'\b\w{5,}\b', q_text))
            unique_keywords.update(keywords)

        diversity_score = min(len(unique_keywords) / (len(questions) * 5), 1.0)

        # Return combined confidence
        return (avg_score * 0.7) + (diversity_score * 0.3)