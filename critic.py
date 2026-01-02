from typing import List, Dict, Any, Tuple
import asyncio
from datetime import datetime
from loguru import logger
import re
import numpy as np

from backend.agents.base_agent import BaseAgent, AgentMessage
from backend.config import settings


class CriticAgent(BaseAgent):
    """Critiques methodology, statistics, and assumptions ruthlessly"""

    def __init__(self, agent_id: str = "critic_001"):
        capabilities = [
            "methodology_critique", "statistical_validation",
            "assumption_analysis", "bias_detection", "logical_fallacy_detection"
        ]
        super().__init__(agent_id, "critic", capabilities)
        self.setup_tools()

    def setup_tools(self):
        """Setup critique tools"""
        self.add_tool(
            name="critique_methodology",
            tool_func=self.critique_methodology_tool,
            description="Critique research methodology and experimental design"
        )

        self.add_tool(
            name="validate_statistics",
            tool_func=self.validate_statistics_tool,
            description="Validate statistical methods and assumptions"
        )

        self.add_tool(
            name="analyze_assumptions",
            tool_func=self.analyze_assumptions_tool,
            description="Analyze and challenge underlying assumptions"
        )

        self.add_tool(
            name="detect_biases",
            tool_func=self.detect_biases_tool,
            description="Detect potential biases in research design"
        )

        self.add_tool(
            name="find_counterevidence",
            tool_func=self.find_counterevidence_tool,
            description="Find counterevidence to challenge findings"
        )

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process critique request"""
        self.log(f"Received message: {message.message_type}")

        if message.message_type == "command" and message.content.get("action") == "critique":
            try:
                research_context = message.content.get("research_context", {})
                target_component = message.content.get("target_component", {})
                component_type = message.content.get("component_type",
                                                     "experiment")  # experiment, hypothesis, data, etc.

                if not target_component:
                    raise ValueError("No target component provided for critique")

                # Perform comprehensive critique
                critique_results = await self.perform_critique(
                    target_component, component_type, research_context
                )

                # Determine if iteration is needed
                requires_iteration = self._requires_iteration(critique_results)

                return AgentMessage(
                    sender=self.agent_id,
                    recipient=message.sender,
                    content={
                        "critique": critique_results,
                        "requires_iteration": requires_iteration,
                        "iteration_reason": critique_results.get("major_issues", []),
                        "confidence": critique_results.get("overall_confidence", 0.5),
                        "critique_date": datetime.now().isoformat()
                    },
                    message_type="result",
                    confidence=self.calculate_critique_confidence(critique_results)
                )

            except Exception as e:
                self.log(f"Critique failed: {e}", "error")
                raise

        return await super().process(message)

    async def perform_critique(self, target: Dict[str, Any],
                               component_type: str,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive critique"""
        self.log(f"Critiquing {component_type}")

        critique = {
            "component_type": component_type,
            "target_id": target.get("id", ""),
            "critique_timestamp": datetime.now().isoformat(),
            "issues_found": [],
            "major_issues": [],
            "minor_issues": [],
            "suggestions": [],
            "counterevidence": [],
            "overall_assessment": "",
            "overall_confidence": 0.5
        }

        # Perform different critiques based on component type
        if component_type == "experiment":
            exp_critique = await self.critique_experiment(target, context)
            critique.update(exp_critique)

        elif component_type == "hypothesis":
            hyp_critique = await self.critique_hypothesis(target, context)
            critique.update(hyp_critique)

        elif component_type == "data":
            data_critique = await self.critique_data(target, context)
            critique.update(data_critique)

        elif component_type == "methodology":
            method_critique = await self.critique_methodology(target, context)
            critique.update(method_critique)

        elif component_type == "results":
            results_critique = await self.critique_results(target, context)
            critique.update(results_critique)

        else:
            # Generic critique
            generic_critique = await self.generic_critique(target, context)
            critique.update(generic_critique)

        # Calculate overall confidence
        critique["overall_confidence"] = self._calculate_overall_confidence(critique)

        # Generate overall assessment
        critique["overall_assessment"] = self._generate_overall_assessment(critique)

        return critique

    async def critique_experiment(self, experiment: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Critique experiment design"""
        critique = {
            "methodology_issues": [],
            "statistical_issues": [],
            "feasibility_issues": [],
            "ethical_issues": [],
            "validity_threats": []
        }

        # 1. Critique methodology
        methodology_critique = await self.critique_methodology_tool(experiment, context)
        critique["methodology_issues"] = methodology_critique.get("issues", [])

        # 2. Validate statistics
        statistical_critique = await self.validate_statistics_tool(experiment, context)
        critique["statistical_issues"] = statistical_critique.get("issues", [])

        # 3. Analyze assumptions
        assumption_critique = await self.analyze_assumptions_tool(experiment, context)
        critique["assumption_issues"] = assumption_critique.get("issues", [])

        # 4. Detect biases
        bias_critique = await self.detect_biases_tool(experiment, context)
        critique["bias_issues"] = bias_critique.get("issues", [])

        # 5. Check feasibility
        feasibility = experiment.get("feasibility_indicators", {})
        if feasibility.get("overall_feasibility", 0.5) < 0.4:
            critique["feasibility_issues"].append(
                f"Low feasibility score ({feasibility.get('overall_feasibility', 0.5):.2f}). May not be practical to implement."
            )

        # 6. Sample size critique
        sample_size = experiment.get("sample_size", {})
        if sample_size:
            min_size = sample_size.get("minimum_sample_size", 0)
            if min_size < 30:
                critique["statistical_issues"].append(
                    f"Small sample size ({min_size}) may lead to underpowered analysis."
                )

        # 7. Control critique
        controls = experiment.get("controls", {})
        if not controls or len(controls) < 2:
            critique["validity_threats"].append(
                "Insufficient controls for confounding variables."
            )

        # Combine all issues
        all_issues = []
        for category, issues in critique.items():
            if issues and "issues" in category:
                all_issues.extend(issues)

        # Categorize issues
        major_issues = [issue for issue in all_issues
                        if any(keyword in issue.lower() for keyword in
                               ["major", "critical", "fatal", "invalid", "unethical", "bias"])]

        minor_issues = [issue for issue in all_issues if issue not in major_issues]

        return {
            "issues_found": all_issues,
            "major_issues": major_issues,
            "minor_issues": minor_issues,
            "suggestions": self._generate_experiment_suggestions(critique, experiment),
            "counterevidence": await self.find_counterevidence_tool(experiment, context)
        }

    async def critique_hypothesis(self, hypothesis: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Critique hypothesis"""
        critique = {
            "clarity_issues": [],
            "testability_issues": [],
            "theoretical_issues": [],
            "logical_issues": []
        }

        statement = hypothesis.get("statement", "")
        variables = hypothesis.get("variables", {})
        testability = hypothesis.get("testability", "")

        # 1. Clarity critique
        if len(statement.split()) > 50:
            critique["clarity_issues"].append("Hypothesis is too long and may lack clarity.")
        if len(statement.split()) < 10:
            critique["clarity_issues"].append("Hypothesis is too vague.")

        # 2. Testability critique
        if not testability or len(testability) < 50:
            critique["testability_issues"].append(
                "Testability description is insufficient. How exactly will this be tested?"
            )

        # 3. Variable critique
        if not variables:
            critique["testability_issues"].append("No variables specified. Hypothesis cannot be tested.")
        else:
            if "independent" not in variables:
                critique["testability_issues"].append("Independent variable not specified.")
            if "dependent" not in variables:
                critique["testability_issues"].append("Dependent variable not specified.")

        # 4. Logical critique
        if "causes" in statement.lower() and "correlation" in context.get("data_summary", "").lower():
            critique["logical_issues"].append(
                "Correlation-causation fallacy: Data suggests correlation but hypothesis implies causation."
            )

        # 5. Check for tautologies
        if self._is_tautology(statement):
            critique["logical_issues"].append("Hypothesis may be tautological (true by definition).")

        # Combine issues
        all_issues = []
        for category, issues in critique.items():
            all_issues.extend(issues)

        return {
            "issues_found": all_issues,
            "major_issues": [issue for issue in all_issues if "cannot be tested" in issue],
            "minor_issues": [issue for issue in all_issues if "cannot be tested" not in issue],
            "suggestions": self._generate_hypothesis_suggestions(critique, hypothesis),
            "counterevidence": await self.find_counterevidence_tool(hypothesis, context)
        }

    async def critique_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Critique data quality and appropriateness"""
        critique = {
            "quality_issues": [],
            "relevance_issues": [],
            "sufficiency_issues": [],
            "bias_issues": []
        }

        quality_metrics = data.get("quality_metrics", {})
        sources = data.get("sources", [])
        data_summary = data.get("summary", "")

        # 1. Quality critique
        if quality_metrics:
            overall_quality = quality_metrics.get("overall_quality", 0.5)
            if overall_quality < 0.6:
                critique["quality_issues"].append(
                    f"Low data quality score ({overall_quality:.2f}). Results may not be reliable."
                )

        # 2. Source critique
        if len(sources) < settings.min_data_sources:
            critique["sufficiency_issues"].append(
                f"Only {len(sources)} data sources, minimum {settings.min_data_sources} required for triangulation."
            )

        # 3. Source credibility
        credible_count = sum(1 for source in sources
                             if any(domain in source.get("url", "").lower()
                                    for domain in [".gov", ".edu", ".ac.", "arxiv"]))

        if credible_count < len(sources) * 0.5:  # Less than 50% credible
            critique["quality_issues"].append(
                f"Low credibility sources: only {credible_count}/{len(sources)} from academic/gov sources."
            )

        # 4. Data relevance
        research_question = context.get("research_question", {}).get("question", "")
        if research_question and data_summary:
            # Simple keyword matching for relevance
            question_words = set(re.findall(r'\b\w{5,}\b', research_question.lower()))
            data_words = set(re.findall(r'\b\w{5,}\b', data_summary.lower()))

            overlap = len(question_words.intersection(data_words))
            relevance_ratio = overlap / max(len(question_words), 1)

            if relevance_ratio < 0.3:
                critique["relevance_issues"].append(
                    f"Low keyword overlap ({relevance_ratio:.2f}). Data may not be relevant to research question."
                )

        # 5. Sample size critique
        sample_size = data.get("integration_metrics", {}).get("successful_sources", 0)
        if sample_size < 3:
            critique["sufficiency_issues"].append(
                f"Small effective sample size ({sample_size}). Statistical power may be insufficient."
            )

        # Combine issues
        all_issues = []
        for category, issues in critique.items():
            all_issues.extend(issues)

        return {
            "issues_found": all_issues,
            "major_issues": [issue for issue in all_issues if
                             "insufficient" in issue.lower() or "unreliable" in issue.lower()],
            "minor_issues": [issue for issue in all_issues if
                             "insufficient" not in issue.lower() and "unreliable" not in issue.lower()],
            "suggestions": self._generate_data_suggestions(critique, data),
            "counterevidence": await self.find_counterevidence_tool(data, context)
        }

    async def critique_methodology(self, methodology: Dict[str, Any],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Critique research methodology"""
        return await self.critique_methodology_tool(methodology, context)

    async def critique_results(self, results: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Critique research results"""
        critique = {
            "statistical_issues": [],
            "interpretation_issues": [],
            "significance_issues": [],
            "generalizability_issues": []
        }

        # This would typically analyze p-values, effect sizes, etc.
        # For now, use LLM-based critique

        prompt = f"""
        Critically evaluate these research results:

        RESULTS: {results}

        CONTEXT: {context}

        Look for:
        1. Statistical issues (p-hacking, multiple comparisons, etc.)
        2. Interpretation errors (overstatement, causation claims from correlation, etc.)
        3. Significance issues (p-value > 0.05 but presented as significant)
        4. Generalizability issues (sample not representative, etc.)
        5. Effect size issues (trivial effect sizes presented as important)

        Provide specific, actionable critique.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            critique_text = response.content

            # Parse critique
            parsed_critique = self._parse_critique_text(critique_text)
            critique.update(parsed_critique)

        except Exception as e:
            self.log(f"Results critique failed: {e}", "warning")
            critique["interpretation_issues"].append("Could not perform detailed critique due to error.")

        # Combine issues
        all_issues = []
        for category, issues in critique.items():
            if issues:
                all_issues.extend(issues if isinstance(issues, list) else [issues])

        return {
            "issues_found": all_issues,
            "major_issues": [issue for issue in all_issues if
                             "significant" in issue.lower() or "invalid" in issue.lower()],
            "minor_issues": [issue for issue in all_issues if
                             "significant" not in issue.lower() and "invalid" not in issue.lower()],
            "suggestions": self._generate_results_suggestions(critique, results),
            "counterevidence": await self.find_counterevidence_tool(results, context)
        }

    async def generic_critique(self, target: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Generic critique for any component"""
        prompt = f"""
        Critically evaluate this research component:

        COMPONENT: {target}

        CONTEXT: {context}

        Provide a thorough critique including:
        1. Major flaws or weaknesses
        2. Minor issues or areas for improvement
        3. Logical inconsistencies
        4. Assumptions that should be challenged
        5. Suggestions for improvement

        Be rigorous and skeptical in your evaluation.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            critique_text = response.content

            # Parse critique
            critique = self._parse_comprehensive_critique(critique_text)

        except Exception as e:
            self.log(f"Generic critique failed: {e}", "warning")
            critique = {
                "issues_found": ["Critique generation failed"],
                "major_issues": [],
                "minor_issues": [],
                "suggestions": ["Revise based on expert feedback"]
            }

        return critique

    async def critique_methodology_tool(self, component: Dict[str, Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Critique methodology"""
        prompt = f"""
        Critically evaluate the methodology in this research component:

        COMPONENT: {component}

        CONTEXT: {context}

        Focus on:
        1. Methodological rigor
        2. Appropriate use of methods for research question
        3. Control of confounding variables
        4. Sampling methodology
        5. Data collection procedures
        6. Ethical considerations

        Identify specific methodological weaknesses.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            critique_text = response.content

            # Parse for issues
            issues = self._extract_issues_from_text(critique_text, "methodology")

            return {
                "issues": issues,
                "critique_text": critique_text
            }

        except Exception as e:
            self.log(f"Methodology critique failed: {e}", "warning")
            return {
                "issues": ["Methodology critique unavailable"],
                "critique_text": ""
            }

    async def validate_statistics_tool(self, component: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical methods"""
        prompt = f"""
        Validate the statistical methods in this research component:

        COMPONENT: {component}

        CONTEXT: {context}

        Check for:
        1. Appropriate statistical tests for data type
        2. Meeting test assumptions
        3. Sample size adequacy
        4. Multiple comparison corrections
        5. Effect size reporting
        6. p-value interpretation

        Flag any statistical issues or violations.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            critique_text = response.content

            # Parse for issues
            issues = self._extract_issues_from_text(critique_text, "statistical")

            return {
                "issues": issues,
                "critique_text": critique_text
            }

        except Exception as e:
            self.log(f"Statistical validation failed: {e}", "warning")
            return {
                "issues": ["Statistical validation unavailable"],
                "critique_text": ""
            }

    async def analyze_assumptions_tool(self, component: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and challenge assumptions"""
        prompt = f"""
        Analyze and challenge the assumptions in this research component:

        COMPONENT: {component}

        CONTEXT: {context}

        Identify:
        1. Explicit assumptions stated
        2. Implicit assumptions not stated
        3. Questionable or untested assumptions
        4. Assumptions that may not hold in practice
        5. Alternative assumptions that could be considered

        Challenge each assumption rigorously.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            critique_text = response.content

            # Parse for issues
            issues = self._extract_issues_from_text(critique_text, "assumption")

            return {
                "issues": issues,
                "critique_text": critique_text
            }

        except Exception as e:
            self.log(f"Assumption analysis failed: {e}", "warning")
            return {
                "issues": ["Assumption analysis unavailable"],
                "critique_text": ""
            }

    async def detect_biases_tool(self, component: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential biases"""
        prompt = f"""
        Detect potential biases in this research component:

        COMPONENT: {component}

        CONTEXT: {context}

        Look for:
        1. Selection bias
        2. Measurement bias
        3. Confirmation bias
        4. Publication bias
        5. Cognitive biases in design/interpretation
        6. Funding/source bias
        7. Cultural/western bias

        Be thorough in bias detection.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            critique_text = response.content

            # Parse for issues
            issues = self._extract_issues_from_text(critique_text, "bias")

            return {
                "issues": issues,
                "critique_text": critique_text
            }

        except Exception as e:
            self.log(f"Bias detection failed: {e}", "warning")
            return {
                "issues": ["Bias detection unavailable"],
                "critique_text": ""
            }

    async def find_counterevidence_tool(self, component: Dict[str, Any],
                                        context: Dict[str, Any]) -> List[str]:
        """Find counterevidence to challenge findings"""
        prompt = f"""
        Find potential counterevidence or alternative explanations for:

        COMPONENT: {component}

        CONTEXT: {context}

        Provide:
        1. Direct counterevidence that contradicts the findings
        2. Alternative explanations for the same results
        3. Conflicting studies in the literature
        4. Methodological limitations that could produce false results
        5. Boundary conditions where findings might not hold

        Be skeptical and thorough.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            counterevidence_text = response.content

            # Parse counterevidence
            counterevidence = self._extract_counterevidence(counterevidence_text)

            return counterevidence

        except Exception as e:
            self.log(f"Counterevidence search failed: {e}", "warning")
            return ["Counterevidence search unavailable"]

    def _requires_iteration(self, critique_results: Dict[str, Any]) -> bool:
        """Determine if iteration is required based on critique"""
        # Check for major issues
        major_issues = critique_results.get("major_issues", [])
        if major_issues:
            return True

        # Check overall confidence
        overall_confidence = critique_results.get("overall_confidence", 0.5)
        if overall_confidence < settings.min_confidence_threshold:
            return True

        # Check for statistical issues that require p-value < 0.05
        issues_found = critique_results.get("issues_found", [])
        statistical_issues = [issue for issue in issues_found
                              if "p-value" in issue.lower() or "statistical" in issue.lower()]

        if statistical_issues:
            # Check if p-value > 0.05 is mentioned
            for issue in statistical_issues:
                if "> 0.05" in issue or "greater than 0.05" in issue.lower():
                    return True

        # Check for trivial effect size
        for issue in issues_found:
            if "trivial" in issue.lower() and "effect" in issue.lower():
                return True

        return False

    def _calculate_overall_confidence(self, critique: Dict[str, Any]) -> float:
        """Calculate overall confidence based on critique"""
        base_confidence = 0.7  # Start with moderate confidence

        # Deductions for issues
        major_issues = len(critique.get("major_issues", []))
        minor_issues = len(critique.get("minor_issues", []))

        # Major issues have stronger impact
        confidence = base_confidence - (major_issues * 0.2) - (minor_issues * 0.05)

        # Bonus for thorough critique
        suggestions = len(critique.get("suggestions", []))
        if suggestions > 3:
            confidence += 0.1  # Bonus for constructive suggestions

        # Counterevidence reduces confidence
        counterevidence = len(critique.get("counterevidence", []))
        confidence -= counterevidence * 0.1

        return max(0.1, min(confidence, 1.0))

    def _generate_overall_assessment(self, critique: Dict[str, Any]) -> str:
        """Generate overall assessment text"""
        major_issues = len(critique.get("major_issues", []))
        minor_issues = len(critique.get("minor_issues", []))
        confidence = critique.get("overall_confidence", 0.5)

        if major_issues > 3:
            return "CRITICAL: Major flaws require complete revision."
        elif major_issues > 0:
            return "SERIOUS: Significant issues need addressing."
        elif minor_issues > 5:
            return "MODERATE: Multiple minor issues need attention."
        elif minor_issues > 0:
            return "MINOR: Some improvements suggested."
        elif confidence > 0.8:
            return "STRONG: Well-designed with high confidence."
        elif confidence > 0.6:
            return "GOOD: Generally sound with some minor concerns."
        else:
            return "NEEDS REVIEW: Confidence is low, review recommended."

    def _generate_experiment_suggestions(self, critique: Dict[str, Any],
                                         experiment: Dict[str, Any]) -> List[str]:
        """Generate suggestions for experiment improvement"""
        suggestions = []

        # Based on critique categories
        if critique.get("methodology_issues"):
            suggestions.append("Revise methodology to address identified weaknesses.")

        if critique.get("statistical_issues"):
            suggestions.append("Re-evaluate statistical methods and assumptions.")

        if critique.get("feasibility_issues"):
            suggestions.append("Simplify design or seek additional resources.")

        if critique.get("ethical_issues"):
            suggestions.append("Address ethical concerns before proceeding.")

        if critique.get("validity_threats"):
            suggestions.append("Add more controls to improve validity.")

        # Specific suggestions based on experiment
        sample_size = experiment.get("sample_size", {}).get("minimum_sample_size", 0)
        if sample_size < 30:
            suggestions.append(f"Increase sample size from {sample_size} to at least 30 for adequate power.")

        complexity = experiment.get("complexity", 0.5)
        if complexity > 0.7:
            suggestions.append("Simplify experimental design to reduce complexity.")

        return suggestions

    def _generate_hypothesis_suggestions(self, critique: Dict[str, Any],
                                         hypothesis: Dict[str, Any]) -> List[str]:
        """Generate suggestions for hypothesis improvement"""
        suggestions = []

        statement = hypothesis.get("statement", "")

        if critique.get("clarity_issues"):
            suggestions.append("Rewrite hypothesis for clarity and conciseness.")

        if critique.get("testability_issues"):
            suggestions.append("Make hypothesis more specific and testable.")

        if critique.get("theoretical_issues"):
            suggestions.append("Strengthen theoretical basis with more evidence.")

        if critique.get("logical_issues"):
            suggestions.append("Address logical fallacies in hypothesis formulation.")

        # Specific suggestions
        if len(statement.split()) > 50:
            suggestions.append("Shorten hypothesis to under 50 words for clarity.")

        variables = hypothesis.get("variables", {})
        if not variables.get("independent") or not variables.get("dependent"):
            suggestions.append("Clearly specify independent and dependent variables.")

        return suggestions

    def _generate_data_suggestions(self, critique: Dict[str, Any],
                                   data: Dict[str, Any]) -> List[str]:
        """Generate suggestions for data improvement"""
        suggestions = []

        if critique.get("quality_issues"):
            suggestions.append("Improve data quality through better collection/cleaning methods.")

        if critique.get("relevance_issues"):
            suggestions.append("Find more relevant data sources aligned with research question.")

        if critique.get("sufficiency_issues"):
            suggestions.append("Acquire additional data sources for triangulation.")

        if critique.get("bias_issues"):
            suggestions.append("Address potential biases in data collection.")

        # Specific suggestions
        sources = data.get("sources", [])
        if len(sources) < settings.min_data_sources:
            suggestions.append(f"Add {settings.min_data_sources - len(sources)} more data sources.")

        return suggestions

    def _generate_results_suggestions(self, critique: Dict[str, Any],
                                      results: Dict[str, Any]) -> List[str]:
        """Generate suggestions for results improvement"""
        suggestions = []

        if critique.get("statistical_issues"):
            suggestions.append("Re-analyze data with appropriate statistical methods.")

        if critique.get("interpretation_issues"):
            suggestions.append("Revise interpretation to avoid overstatement.")

        if critique.get("significance_issues"):
            suggestions.append("Report non-significant results appropriately.")

        if critique.get("generalizability_issues"):
            suggestions.append("Clearly state limitations on generalizability.")

        return suggestions

    def _is_tautology(self, statement: str) -> bool:
        """Check if statement is tautological"""
        statement_lower = statement.lower()

        # Common tautological patterns
        tautological_patterns = [
            r"is what it is",
            r"true because it's true",
            r"by definition",
            r"circular",
            r"self-evident"
        ]

        for pattern in tautological_patterns:
            if re.search(pattern, statement_lower):
                return True

        # Check for circular definitions
        words = statement_lower.split()
        if len(words) < 10:
            return False

        # Simple check for repetition
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:  # High repetition
            return True

        return False

    def _extract_issues_from_text(self, text: str, issue_type: str) -> List[str]:
        """Extract issues from critique text"""
        issues = []

        # Look for bullet points or numbered lists
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Skip empty lines and section headers
            if not line or line.upper() == line or len(line) < 10:
                continue

            # Check if line contains issue indicators
            issue_indicators = ["issue", "problem", "weakness", "limitation",
                                "concern", "flaw", "bias", "violation", "error"]

            if any(indicator in line.lower() for indicator in issue_indicators):
                # Clean up the line
                line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove numbers
                line = re.sub(r'^[•\-*]\s*', '', line)  # Remove bullets

                if len(line) > 20:  # Meaningful length
                    issues.append(line)

        # If no issues found, use sentences
        if not issues:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30 and any(indicator in sentence.lower()
                                              for indicator in issue_indicators):
                    issues.append(sentence)

        return issues[:10]  # Limit to 10 issues

    def _extract_counterevidence(self, text: str) -> List[str]:
        """Extract counterevidence from text"""
        counterevidence = []

        # Look for counterevidence indicators
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()

            if not line or len(line) < 20:
                continue

            counter_indicators = ["counterevidence", "contradict", "alternative",
                                  "conflict", "contrary", "however", "but", "although"]

            if any(indicator in line.lower() for indicator in counter_indicators):
                # Clean up
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[•\-*]\s*', '', line)

                counterevidence.append(line)

        # Fallback: use all meaningful lines
        if not counterevidence:
            for line in lines:
                line = line.strip()
                if len(line) > 30:
                    counterevidence.append(line)

        return counterevidence[:5]  # Limit to 5

    def _parse_critique_text(self, text: str) -> Dict[str, List[str]]:
        """Parse critique text into categories"""
        critique = {
            "statistical_issues": [],
            "interpretation_issues": [],
            "significance_issues": [],
            "generalizability_issues": []
        }

        text_lower = text.lower()

        # Categorize based on keywords
        lines = text.strip().split('\n')

        for line in lines:
            line_lower = line.lower()

            if any(term in line_lower for term in ["p-value", "statistical", "test", "analysis", "assumption"]):
                critique["statistical_issues"].append(line.strip())
            elif any(term in line_lower for term in ["interpret", "conclude", "claim", "overstate"]):
                critique["interpretation_issues"].append(line.strip())
            elif any(term in line_lower for term in ["significant", "p >", "p<", "effect size"]):
                critique["significance_issues"].append(line.strip())
            elif any(term in line_lower for term in ["generaliz", "sample", "represent", "population"]):
                critique["generalizability_issues"].append(line.strip())

        return critique

    def _parse_comprehensive_critique(self, text: str) -> Dict[str, Any]:
        """Parse comprehensive critique text"""
        critique = {
            "issues_found": [],
            "major_issues": [],
            "minor_issues": [],
            "suggestions": []
        }

        lines = text.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect sections
            if "major" in line.lower() and "issue" in line.lower():
                current_section = "major_issues"
                continue
            elif "minor" in line.lower() and "issue" in line.lower():
                current_section = "minor_issues"
                continue
            elif "suggestion" in line.lower() or "recommend" in line.lower():
                current_section = "suggestions"
                continue
            elif "issue" in line.lower() or "problem" in line.lower():
                current_section = "issues_found"
                continue

            # Add content to current section
            if current_section and len(line) > 10:
                # Clean line
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[•\-*]\s*', '', line)

                if line and line not in critique[current_section]:
                    critique[current_section].append(line)

        # Combine all issues
        all_issues = critique["major_issues"] + critique["minor_issues"] + critique["issues_found"]
        critique["issues_found"] = list(set(all_issues))  # Remove duplicates

        return critique

    def calculate_critique_confidence(self, critique_results: Dict[str, Any]) -> float:
        """Calculate confidence in critique"""
        # Confidence based on thoroughness
        major_issues = len(critique_results.get("major_issues", []))
        minor_issues = len(critique_results.get("minor_issues", []))
        suggestions = len(critique_results.get("suggestions", []))
        counterevidence = len(critique_results.get("counterevidence", []))

        # More issues found = more confident in critique
        if major_issues + minor_issues == 0:
            # No issues found - either perfect or superficial critique
            if suggestions > 2 or counterevidence > 0:
                return 0.8  # Found improvements but no flaws
            else:
                return 0.5  # Superficial critique

        # Calculate confidence based on critique depth
        total_items = major_issues + minor_issues + suggestions + counterevidence

        if total_items > 10:
            return 0.9  # Very thorough critique
        elif total_items > 5:
            return 0.8  # Thorough critique
        elif total_items > 2:
            return 0.7  # Adequate critique
        else:
            return 0.6  # Minimal critique