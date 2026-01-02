from typing import List, Dict, Any, Tuple
import asyncio
from datetime import datetime
from loguru import logger
import numpy as np
import re

from backend.agents.base_agent import BaseAgent, AgentMessage
from backend.config import settings


class ExperimentDesignerAgent(BaseAgent):
    """Designs experiments and formulates hypotheses based on data"""

    def __init__(self, agent_id: str = "exp_001"):
        capabilities = [
            "hypothesis_formulation", "experimental_design",
            "statistical_analysis", "methodology", "variable_selection"
        ]
        super().__init__(agent_id, "experiment_designer", capabilities)
        self.setup_tools()

    def setup_tools(self):
        """Setup experiment design tools"""
        self.add_tool(
            name="formulate_hypothesis",
            tool_func=self.formulate_hypothesis_tool,
            description="Formulate testable hypotheses from research question and data"
        )

        self.add_tool(
            name="design_experiment",
            tool_func=self.design_experiment_tool,
            description="Design experiment to test hypothesis"
        )

        self.add_tool(
            name="select_statistical_methods",
            tool_func=self.select_statistical_methods_tool,
            description="Select appropriate statistical methods for analysis"
        )

        self.add_tool(
            name="calculate_sample_size",
            tool_func=self.calculate_sample_size_tool,
            description="Calculate required sample size for experiment"
        )

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process experiment design request"""
        self.log(f"Received message: {message.message_type}")

        if message.message_type == "command" and message.content.get("action") == "design_experiment":
            try:
                research_question = message.content.get("research_question", {})
                data_summary = message.content.get("data_summary", {})

                if not research_question:
                    raise ValueError("No research question provided")

                # Formulate hypotheses
                hypotheses = await self.formulate_hypotheses(research_question, data_summary)

                # Design experiments for each hypothesis
                experiments = []
                for hypothesis in hypotheses[:3]:  # Design for top 3 hypotheses
                    experiment = await self.design_experiment(hypothesis, data_summary)
                    experiments.append(experiment)

                # Select primary experiment
                if experiments:
                    primary_experiment = self._select_primary_experiment(experiments)

                    # Calculate feasibility
                    feasibility = await self.assess_experiment_feasibility(primary_experiment, data_summary)

                    return AgentMessage(
                        sender=self.agent_id,
                        recipient=message.sender,
                        content={
                            "hypotheses": hypotheses,
                            "experiments": experiments,
                            "primary_experiment": primary_experiment,
                            "feasibility_assessment": feasibility,
                            "design_date": datetime.now().isoformat()
                        },
                        message_type="result",
                        confidence=self.calculate_experiment_confidence(primary_experiment, feasibility)
                    )
                else:
                    raise ValueError("No valid experiments designed")

            except Exception as e:
                self.log(f"Experiment design failed: {e}", "error")
                raise

        return await super().process(message)

    async def formulate_hypothesis_tool(self, research_question: Dict[str, Any],
                                        data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formulate testable hypotheses"""
        question_text = research_question.get("question", "")
        data_text = data_summary.get("summary", "No data summary available")

        prompt = f"""
        Based on the research question and available data, formulate 3-5 testable hypotheses.

        RESEARCH QUESTION: {question_text}

        AVAILABLE DATA SUMMARY: {data_text}

        For each hypothesis, provide:
        1. Hypothesis Statement: [Clear, testable statement]
        2. Type: [Null hypothesis (H0) or Alternative hypothesis (H1)]
        3. Variables: [Independent and dependent variables]
        4. Expected Direction: [Positive, Negative, or Non-directional]
        5. Theoretical Basis: [Why this hypothesis is plausible based on data]
        6. Testability: [How this can be tested with available data]

        Ensure hypotheses are:
        - Specific and measurable
        - Directly related to the research question
        - Grounded in the available data
        - Statistically testable
        """

        try:
            response = await self.llm.ainvoke(prompt)
            hypotheses_text = response.content

            # Parse hypotheses
            hypotheses = self._parse_hypotheses_from_text(hypotheses_text)

            # Add metadata
            for i, hypothesis in enumerate(hypotheses):
                hypothesis["id"] = f"h_{i + 1}_{datetime.now().strftime('%Y%m%d')}"
                hypothesis["question_id"] = research_question.get("id", "")
                hypothesis["formulated_at"] = datetime.now().isoformat()

            self.log(f"Formulated {len(hypotheses)} hypotheses")
            return hypotheses

        except Exception as e:
            self.log(f"Hypothesis formulation failed: {e}", "error")
            return []

    async def formulate_hypotheses(self, research_question: Dict[str, Any],
                                   data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formulate hypotheses with validation"""
        hypotheses = await self.formulate_hypothesis_tool(research_question, data_summary)

        # Score each hypothesis
        scored_hypotheses = []
        for hypothesis in hypotheses:
            score = await self._score_hypothesis(hypothesis, research_question, data_summary)
            scored_hypotheses.append({
                **hypothesis,
                "quality_score": score,
                "is_testable": self._check_testability(hypothesis, data_summary)
            })

        # Sort by quality score
        scored_hypotheses.sort(key=lambda x: x["quality_score"], reverse=True)

        # Store in memory
        for hypothesis in scored_hypotheses[:3]:
            self.memory.store(
                f"Hypothesis: {hypothesis.get('statement', '')[:100]}",
                metadata={
                    "type": "hypothesis",
                    "quality_score": hypothesis.get("quality_score", 0),
                    "is_testable": hypothesis.get("is_testable", False)
                }
            )

        return scored_hypotheses

    async def design_experiment_tool(self, hypothesis: Dict[str, Any],
                                     data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Design experiment to test hypothesis"""
        hypothesis_statement = hypothesis.get("statement", "")
        variables = hypothesis.get("variables", {})
        data_types = data_summary.get("data_types", [])

        prompt = f"""
        Design an experiment to test this hypothesis:

        HYPOTHESIS: {hypothesis_statement}

        VARIABLES: {variables}

        AVAILABLE DATA TYPES: {data_types}

        Provide a detailed experiment design including:

        1. EXPERIMENT TYPE: [Observational, Experimental, Quasi-experimental, Simulation, etc.]

        2. METHODOLOGY:
           - Study Design: [Cross-sectional, Longitudinal, Case-Control, Randomized Controlled Trial, etc.]
           - Sampling Strategy: [Random, Stratified, Convenience, etc.]
           - Data Collection Methods: [Surveys, Measurements, Observations, etc.]

        3. PROCEDURE:
           - Step-by-step procedure
           - Timeline
           - Ethical considerations

        4. VARIABLES OPERATIONALIZATION:
           - How each variable will be measured
           - Measurement instruments/tools
           - Units of measurement

        5. CONTROLS:
           - Control variables
           - Control groups
           - Confounding variable management

        6. DATA ANALYSIS PLAN:
           - Statistical tests to be used
           - Software/tools for analysis
           - Success criteria

        Make the design practical and feasible with typical research resources.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            experiment_text = response.content

            # Parse experiment design
            experiment = self._parse_experiment_from_text(experiment_text)

            # Add metadata
            experiment["hypothesis_id"] = hypothesis.get("id", "")
            experiment["designed_at"] = datetime.now().isoformat()
            experiment["complexity"] = self._calculate_experiment_complexity(experiment)

            # Add statistical methods
            experiment["statistical_methods"] = await self.select_statistical_methods(
                hypothesis, experiment, data_summary
            )

            # Calculate required sample size
            experiment["sample_size"] = await self.calculate_sample_size(
                hypothesis, experiment, data_summary
            )

            self.log(f"Designed experiment for hypothesis: {hypothesis.get('id')}")
            return experiment

        except Exception as e:
            self.log(f"Experiment design failed: {e}", "error")
            return {
                "hypothesis_id": hypothesis.get("id", ""),
                "error": str(e),
                "designed_at": datetime.now().isoformat()
            }

    async def design_experiment(self, hypothesis: Dict[str, Any],
                                data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Design experiment with feasibility check"""
        experiment = await self.design_experiment_tool(hypothesis, data_summary)

        if "error" in experiment:
            return experiment

        # Add feasibility indicators
        feasibility_indicators = self._calculate_feasibility_indicators(experiment, data_summary)
        experiment["feasibility_indicators"] = feasibility_indicators

        # Calculate resource requirements
        resource_requirements = self._estimate_resource_requirements(experiment)
        experiment["resource_requirements"] = resource_requirements

        # Store in memory
        self.memory.store(
            f"Experiment design for: {hypothesis.get('statement', '')[:100]}",
            metadata={
                "type": "experiment_design",
                "complexity": experiment.get("complexity", 0),
                "feasibility": feasibility_indicators.get("overall_feasibility", 0)
            }
        )

        return experiment

    async def select_statistical_methods_tool(self, hypothesis: Dict[str, Any],
                                              experiment: Dict[str, Any],
                                              data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select appropriate statistical methods"""
        hypothesis_statement = hypothesis.get("statement", "")
        experiment_type = experiment.get("type", "")
        variables = hypothesis.get("variables", {})
        data_types = data_summary.get("data_types", [])

        prompt = f"""
        Select appropriate statistical methods for testing this hypothesis:

        HYPOTHESIS: {hypothesis_statement}

        EXPERIMENT TYPE: {experiment_type}

        VARIABLES: {variables}

        DATA TYPES AVAILABLE: {data_types}

        For each statistical method, provide:
        1. Method Name: [e.g., t-test, ANOVA, regression, etc.]
        2. Purpose: [What it tests]
        3. Assumptions: [Statistical assumptions]
        4. When to Use: [Appropriate conditions]
        5. Interpretation: [How to interpret results]
        6. Software Implementation: [Python/R code example]

        Prioritize methods that are:
        - Appropriate for the data types
        - Robust to violations of assumptions
        - Commonly used in the field
        - Easy to interpret
        """

        try:
            response = await self.llm.ainvoke(prompt)
            methods_text = response.content

            # Parse statistical methods
            methods = self._parse_statistical_methods(methods_text)

            return methods

        except Exception as e:
            self.log(f"Statistical method selection failed: {e}", "warning")
            return [{"error": str(e)}]

    async def select_statistical_methods(self, hypothesis: Dict[str, Any],
                                         experiment: Dict[str, Any],
                                         data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select statistical methods with validation"""
        methods = await self.select_statistical_methods_tool(hypothesis, experiment, data_summary)

        # Score each method
        scored_methods = []
        for method in methods:
            if "error" in method:
                continue

            score = self._score_statistical_method(method, hypothesis, experiment, data_summary)
            scored_methods.append({
                **method,
                "appropriateness_score": score,
                "implementation_complexity": self._assess_implementation_complexity(method)
            })

        # Sort by appropriateness
        scored_methods.sort(key=lambda x: x["appropriateness_score"], reverse=True)

        return scored_methods[:3]  # Return top 3 methods

    async def calculate_sample_size_tool(self, hypothesis: Dict[str, Any],
                                         experiment: Dict[str, Any],
                                         data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate required sample size"""
        hypothesis_statement = hypothesis.get("statement", "")
        experiment_type = experiment.get("type", "")
        statistical_methods = experiment.get("statistical_methods", [])

        prompt = f"""
        Calculate the required sample size for this experiment:

        HYPOTHESIS: {hypothesis_statement}

        EXPERIMENT TYPE: {experiment_type}

        PRIMARY STATISTICAL METHOD: {statistical_methods[0].get('name', 'Unknown') if statistical_methods else 'Unknown'}

        Provide sample size calculation including:

        1. PARAMETERS:
           - Effect size (small, medium, large)
           - Alpha level (Type I error rate)
           - Power (1 - Beta)
           - Expected variability

        2. CALCULATION:
           - Formula/method used
           - Assumptions made
           - Justification for parameters

        3. RESULTS:
           - Minimum required sample size
           - Recommended sample size (with buffer)
           - Effect on power with different sample sizes

        4. PRACTICAL CONSIDERATIONS:
           - Feasibility of achieving sample size
           - Strategies for recruitment
           - Alternatives if sample size cannot be achieved
        """

        try:
            response = await self.llm.ainvoke(prompt)
            sample_size_text = response.content

            # Parse sample size calculation
            sample_size_info = self._parse_sample_size_calculation(sample_size_text)

            # Add practical assessment
            sample_size_info["feasibility"] = self._assess_sample_size_feasibility(sample_size_info)

            return sample_size_info

        except Exception as e:
            self.log(f"Sample size calculation failed: {e}", "warning")
            return {
                "error": str(e),
                "minimum_sample_size": 30,  # Default
                "recommended_sample_size": 50
            }

    async def calculate_sample_size(self, hypothesis: Dict[str, Any],
                                    experiment: Dict[str, Any],
                                    data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sample size with validation"""
        sample_size_info = await self.calculate_sample_size_tool(hypothesis, experiment, data_summary)

        # Add data-based validation if possible
        if data_summary.get("data_volume"):
            available_data_points = self._estimate_available_data_points(data_summary)
            sample_size_info["available_data_points"] = available_data_points

            # Check if available data is sufficient
            required = sample_size_info.get("minimum_sample_size", 0)
            sample_size_info["data_sufficiency"] = available_data_points >= required if required > 0 else True

        return sample_size_info

    async def assess_experiment_feasibility(self, experiment: Dict[str, Any],
                                            data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Assess feasibility of experiment"""
        feasibility = {
            "technical_feasibility": 0.7,
            "resource_feasibility": 0.5,
            "time_feasibility": 0.6,
            "ethical_feasibility": 0.8,
            "data_feasibility": 0.7,
            "overall_feasibility": 0.66
        }

        # Technical feasibility
        complexity = experiment.get("complexity", 0.5)
        feasibility["technical_feasibility"] = 1.0 - (complexity * 0.5)

        # Resource feasibility
        resources = experiment.get("resource_requirements", {})
        resource_score = self._calculate_resource_feasibility(resources)
        feasibility["resource_feasibility"] = resource_score

        # Time feasibility
        timeline = experiment.get("procedure", {}).get("timeline", "Not specified")
        time_score = self._assess_timeline_feasibility(timeline)
        feasibility["time_feasibility"] = time_score

        # Data feasibility
        sample_size = experiment.get("sample_size", {})
        data_score = self._assess_data_feasibility(sample_size, data_summary)
        feasibility["data_feasibility"] = data_score

        # Overall feasibility (weighted average)
        weights = {
            "technical_feasibility": 0.25,
            "resource_feasibility": 0.20,
            "time_feasibility": 0.15,
            "ethical_feasibility": 0.10,
            "data_feasibility": 0.30
        }

        overall = sum(
            feasibility[key] * weights.get(key, 0) for key in feasibility.keys() if key != "overall_feasibility")
        feasibility["overall_feasibility"] = overall

        # Add recommendations
        feasibility["recommendations"] = self._generate_feasibility_recommendations(feasibility, experiment)

        return feasibility

    def _score_hypothesis(self, hypothesis: Dict[str, Any],
                          research_question: Dict[str, Any],
                          data_summary: Dict[str, Any]) -> float:
        """Score hypothesis quality"""
        score = 0.5  # Base score

        # Clarity score
        statement = hypothesis.get("statement", "")
        clarity_score = min(len(statement.split()) / 50, 1.0)  # Optimal: 20-50 words
        score += clarity_score * 0.2

        # Testability score
        testability = hypothesis.get("testability", "")
        if "test" in testability.lower() and "data" in testability.lower():
            score += 0.2

        # Variable specificity
        variables = hypothesis.get("variables", {})
        if isinstance(variables, dict) and len(variables) >= 2:
            score += 0.2

        # Alignment with research question
        question_words = set(research_question.get("question", "").lower().split())
        statement_words = set(statement.lower().split())
        overlap = len(question_words.intersection(statement_words))
        alignment_score = overlap / max(len(question_words), 1)
        score += alignment_score * 0.2

        # Data support
        data_text = data_summary.get("summary", "").lower()
        statement_keywords = set(re.findall(r'\b\w{5,}\b', statement.lower()))
        data_keywords = set(re.findall(r'\b\w{5,}\b', data_text))

        data_support = len(statement_keywords.intersection(data_keywords)) / max(len(statement_keywords), 1)
        score += data_support * 0.2

        return min(score, 1.0)

    def _check_testability(self, hypothesis: Dict[str, Any], data_summary: Dict[str, Any]) -> bool:
        """Check if hypothesis is testable with available data"""
        statement = hypothesis.get("statement", "").lower()

        # Check for measurable concepts
        measurable_terms = ["increase", "decrease", "correlate", "affect", "influence",
                            "relationship", "difference", "effect", "impact"]

        has_measurable = any(term in statement for term in measurable_terms)

        # Check for variables
        variables = hypothesis.get("variables", {})
        has_variables = bool(variables) and len(variables) >= 2

        # Check data availability
        data_volume = data_summary.get("data_volume", 0)
        has_sufficient_data = data_volume > 100  # Arbitrary threshold

        return has_measurable and has_variables and has_sufficient_data

    def _calculate_experiment_complexity(self, experiment: Dict[str, Any]) -> float:
        """Calculate experiment complexity (0-1 scale)"""
        complexity = 0.5  # Base

        # Experiment type complexity
        exp_type = experiment.get("type", "").lower()
        type_complexity = {
            "observational": 0.3,
            "cross-sectional": 0.4,
            "longitudinal": 0.7,
            "experimental": 0.6,
            "randomized": 0.8,
            "simulation": 0.7,
            "quasi-experimental": 0.6
        }

        for key, value in type_complexity.items():
            if key in exp_type:
                complexity = value
                break

        # Procedure steps
        procedure = experiment.get("procedure", {})
        if isinstance(procedure, dict):
            steps = procedure.get("steps", [])
            if isinstance(steps, list):
                step_complexity = min(len(steps) / 10, 0.3)  # Max 10 steps
                complexity += step_complexity

        # Statistical methods complexity
        methods = experiment.get("statistical_methods", [])
        method_count = len(methods)
        method_complexity = min(method_count / 5, 0.2)  # Max 5 methods
        complexity += method_complexity

        return min(complexity, 1.0)

    def _calculate_feasibility_indicators(self, experiment: Dict[str, Any],
                                          data_summary: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feasibility indicators"""
        indicators = {}

        # Resource feasibility
        resources = experiment.get("resource_requirements", {})
        resource_score = 0.7  # Default

        if resources:
            # Simplified calculation
            required_items = sum(1 for key in ["equipment", "software", "personnel", "funding"]
                                 if key in resources and resources[key])
            resource_score = max(0.3, 1.0 - (required_items * 0.1))

        indicators["resource_feasibility"] = resource_score

        # Time feasibility
        timeline = experiment.get("procedure", {}).get("timeline", "")
        time_score = 0.6  # Default

        if timeline:
            # Check for long durations
            long_terms = ["months", "years", "long-term", "extended"]
            if any(term in timeline.lower() for term in long_terms):
                time_score = 0.3

        indicators["time_feasibility"] = time_score

        # Technical feasibility
        complexity = experiment.get("complexity", 0.5)
        technical_score = 1.0 - (complexity * 0.5)
        indicators["technical_feasibility"] = technical_score

        # Data feasibility
        sample_size = experiment.get("sample_size", {}).get("minimum_sample_size", 30)
        data_volume = data_summary.get("data_volume", 0)

        if data_volume > 0:
            data_score = min(data_volume / (sample_size * 10), 1.0)
        else:
            data_score = 0.5

        indicators["data_feasibility"] = data_score

        # Overall feasibility
        indicators["overall_feasibility"] = (
                resource_score * 0.3 +
                time_score * 0.2 +
                technical_score * 0.3 +
                data_score * 0.2
        )

        return indicators

    def _estimate_resource_requirements(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements"""
        requirements = {
            "equipment": [],
            "software": [],
            "personnel": [],
            "funding": "Minimal to moderate",
            "time": "Varies by experiment complexity"
        }

        exp_type = experiment.get("type", "").lower()

        # Basic equipment based on experiment type
        if any(t in exp_type for t in ["lab", "experimental", "measurement"]):
            requirements["equipment"].extend(["Basic lab equipment", "Measurement devices"])
        elif "simulation" in exp_type:
            requirements["equipment"].extend(["High-performance computer"])
        else:
            requirements["equipment"].append("Standard computer")

        # Software requirements
        statistical_methods = experiment.get("statistical_methods", [])
        if statistical_methods:
            requirements["software"].append("Statistical software (Python/R)")

        if "simulation" in exp_type:
            requirements["software"].append("Simulation software")

        # Personnel
        complexity = experiment.get("complexity", 0.5)
        if complexity > 0.7:
            requirements["personnel"].extend(["Principal researcher", "Research assistant", "Statistician"])
        elif complexity > 0.4:
            requirements["personnel"].extend(["Researcher", "Assistant"])
        else:
            requirements["personnel"].append("Single researcher")

        # Funding estimate
        if complexity > 0.7:
            requirements["funding"] = "Moderate to high"
        elif complexity > 0.4:
            requirements["funding"] = "Low to moderate"
        else:
            requirements["funding"] = "Minimal"

        # Time estimate
        if complexity > 0.7:
            requirements["time"] = "Several months"
        elif complexity > 0.4:
            requirements["time"] = "Weeks to months"
        else:
            requirements["time"] = "Days to weeks"

        return requirements

    def _score_statistical_method(self, method: Dict[str, Any],
                                  hypothesis: Dict[str, Any],
                                  experiment: Dict[str, Any],
                                  data_summary: Dict[str, Any]) -> float:
        """Score appropriateness of statistical method"""
        score = 0.5  # Base score

        method_name = method.get("name", "").lower()

        # Match method to experiment type
        exp_type = experiment.get("type", "").lower()

        # Common pairings
        appropriate_pairings = {
            "observational": ["regression", "correlation", "chi-square"],
            "experimental": ["t-test", "anova", "mann-whitney"],
            "longitudinal": ["repeated measures", "time series", "mixed models"],
            "survey": ["factor analysis", "reliability", "descriptive stats"]
        }

        for exp_key, methods in appropriate_pairings.items():
            if exp_key in exp_type:
                if any(m in method_name for m in methods):
                    score += 0.3
                break

        # Check assumptions match data
        assumptions = method.get("assumptions", "").lower()
        data_types = [t.lower() for t in data_summary.get("data_types", [])]

        # Simplified assumption checking
        if "normal" in assumptions and "normal" not in str(data_summary.get("distribution", "")):
            score -= 0.1
        if "continuous" in assumptions and "continuous" not in str(data_types):
            score -= 0.1

        return max(score, 0.1)

    def _assess_implementation_complexity(self, method: Dict[str, Any]) -> str:
        """Assess implementation complexity of statistical method"""
        method_name = method.get("name", "").lower()

        simple_methods = ["t-test", "chi-square", "correlation", "descriptive"]
        moderate_methods = ["anova", "regression", "mann-whitney", "wilcoxon"]
        complex_methods = ["mixed models", "factor analysis", "structural equation", "time series"]

        if any(m in method_name for m in simple_methods):
            return "low"
        elif any(m in method_name for m in moderate_methods):
            return "medium"
        elif any(m in method_name for m in complex_methods):
            return "high"
        else:
            return "medium"

    def _parse_sample_size_calculation(self, text: str) -> Dict[str, Any]:
        """Parse sample size calculation from text"""
        info = {
            "minimum_sample_size": 30,
            "recommended_sample_size": 50,
            "effect_size": "medium",
            "alpha": 0.05,
            "power": 0.8,
            "calculation_method": "Standard power analysis"
        }

        # Extract numbers
        import re

        # Look for sample sizes
        sample_matches = re.findall(r'sample.*?size.*?(\d+)', text.lower())
        if sample_matches:
            info["minimum_sample_size"] = int(sample_matches[0])
            if len(sample_matches) > 1:
                info["recommended_sample_size"] = int(sample_matches[1])

        # Look for effect size
        if "small" in text.lower():
            info["effect_size"] = "small"
        elif "large" in text.lower():
            info["effect_size"] = "large"

        return info

    def _assess_sample_size_feasibility(self, sample_size_info: Dict[str, Any]) -> str:
        """Assess feasibility of achieving sample size"""
        min_size = sample_size_info.get("minimum_sample_size", 30)

        if min_size <= 30:
            return "high"
        elif min_size <= 100:
            return "medium"
        elif min_size <= 500:
            return "low"
        else:
            return "very low"

    def _estimate_available_data_points(self, data_summary: Dict[str, Any]) -> int:
        """Estimate available data points"""
        data_volume = data_summary.get("data_volume", 0)

        if data_volume > 0:
            # Rough estimate: 1KB ≈ 100 data points
            return min(int(data_volume / 10), 10000)

        # Fallback based on data types
        data_types = data_summary.get("data_types", [])

        if "tabular" in data_types:
            return 1000
        elif "textual" in data_types:
            return 500
        else:
            return 100

    def _calculate_resource_feasibility(self, resources: Dict[str, Any]) -> float:
        """Calculate resource feasibility score"""
        # Simplified scoring
        equipment_count = len(resources.get("equipment", []))
        software_count = len(resources.get("software", []))
        personnel_count = len(resources.get("personnel", []))

        total_items = equipment_count + software_count + personnel_count

        if total_items == 0:
            return 1.0
        elif total_items == 1:
            return 0.8
        elif total_items == 2:
            return 0.6
        elif total_items == 3:
            return 0.4
        else:
            return 0.2

    def _assess_timeline_feasibility(self, timeline: str) -> float:
        """Assess timeline feasibility"""
        if not timeline:
            return 0.7

        timeline_lower = timeline.lower()

        if any(term in timeline_lower for term in ["days", "week", "short"]):
            return 0.9
        elif any(term in timeline_lower for term in ["weeks", "month"]):
            return 0.7
        elif any(term in timeline_lower for term in ["months", "quarter"]):
            return 0.5
        elif any(term in timeline_lower for term in ["year", "long", "extended"]):
            return 0.3
        else:
            return 0.6

    def _assess_data_feasibility(self, sample_size_info: Dict[str, Any],
                                 data_summary: Dict[str, Any]) -> float:
        """Assess data feasibility"""
        required = sample_size_info.get("minimum_sample_size", 30)
        available = self._estimate_available_data_points(data_summary)

        if available >= required * 1.5:  # 50% buffer
            return 0.9
        elif available >= required:
            return 0.7
        elif available >= required * 0.5:
            return 0.4
        else:
            return 0.2

    def _generate_feasibility_recommendations(self, feasibility: Dict[str, float],
                                              experiment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on feasibility assessment"""
        recommendations = []

        overall = feasibility.get("overall_feasibility", 0.5)

        if overall < 0.4:
            recommendations.append("Consider simplifying the experiment design")
            recommendations.append("Explore alternative hypotheses with lower resource requirements")

        if feasibility.get("resource_feasibility", 0.5) < 0.4:
            recommendations.append("Seek additional resources or funding")
            recommendations.append("Consider open-source alternatives for software/tools")

        if feasibility.get("time_feasibility", 0.5) < 0.4:
            recommendations.append("Break experiment into phases")
            recommendations.append("Consider parallel data collection methods")

        if feasibility.get("data_feasibility", 0.5) < 0.4:
            recommendations.append("Explore additional data sources")
            recommendations.append("Consider simulation or synthetic data generation")

        if not recommendations:
            recommendations.append("Experiment appears feasible with current design")

        return recommendations

    def _select_primary_experiment(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select primary experiment from list"""
        if not experiments:
            return {}

        # Score each experiment
        scored_experiments = []
        for exp in experiments:
            if "error" in exp:
                continue

            score = self._calculate_experiment_score(exp)
            scored_experiments.append({
                **exp,
                "selection_score": score
            })

        if not scored_experiments:
            return experiments[0] if experiments else {}

        # Select highest scoring
        scored_experiments.sort(key=lambda x: x["selection_score"], reverse=True)
        return scored_experiments[0]

    def _calculate_experiment_score(self, experiment: Dict[str, Any]) -> float:
        """Calculate overall experiment selection score"""
        score = 0.5  # Base

        # Feasibility contribution
        feasibility = experiment.get("feasibility_indicators", {}).get("overall_feasibility", 0.5)
        score += feasibility * 0.3

        # Complexity (inverse)
        complexity = experiment.get("complexity", 0.5)
        score += (1.0 - complexity) * 0.2

        # Statistical robustness
        methods = experiment.get("statistical_methods", [])
        if methods:
            method_scores = [m.get("appropriateness_score", 0.5) for m in methods if isinstance(m, dict)]
            if method_scores:
                avg_method_score = sum(method_scores) / len(method_scores)
                score += avg_method_score * 0.2

        # Sample size feasibility
        sample_size = experiment.get("sample_size", {})
        sample_feasibility = sample_size.get("feasibility", "medium")
        feasibility_map = {"high": 0.3, "medium": 0.2, "low": 0.1, "very low": 0.0}
        score += feasibility_map.get(sample_feasibility, 0.1)

        return min(score, 1.0)

    def calculate_experiment_confidence(self, experiment: Dict[str, Any],
                                        feasibility: Dict[str, Any]) -> float:
        """Calculate confidence in experiment design"""
        if not experiment or "error" in experiment:
            return 0.1

        # Base confidence from feasibility
        overall_feasibility = feasibility.get("overall_feasibility", 0.5)

        # Adjust based on experiment quality
        complexity = experiment.get("complexity", 0.5)

        # Ideal complexity is 0.4-0.6
        if 0.4 <= complexity <= 0.6:
            complexity_factor = 1.0
        elif complexity < 0.4:
            complexity_factor = 0.8  # Too simple
        else:
            complexity_factor = 0.6  # Too complex

        # Statistical methods quality
        methods = experiment.get("statistical_methods", [])
        method_quality = 0.5
        if methods:
            method_scores = [m.get("appropriateness_score", 0.5) for m in methods if isinstance(m, dict)]
            if method_scores:
                method_quality = sum(method_scores) / len(method_scores)

        # Combine factors
        confidence = (
                overall_feasibility * 0.4 +
                complexity_factor * 0.2 +
                method_quality * 0.3 +
                0.1  # Small bonus for having a design
        )

        return min(confidence, 1.0)

    def _parse_hypotheses_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse hypotheses from LLM response"""
        hypotheses = []

        # Split by hypothesis markers
        hypothesis_blocks = re.split(r'\n\d+[\.\)]\s+|\n-+\s*\n', text)

        for block in hypothesis_blocks:
            if not block.strip() or "hypothesis" not in block.lower():
                continue

            # Extract components
            statement_match = re.search(r'Hypothesis Statement:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            type_match = re.search(r'Type:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            variables_match = re.search(r'Variables:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            direction_match = re.search(r'Expected Direction:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            basis_match = re.search(r'Theoretical Basis:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            testability_match = re.search(r'Testability:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)

            if statement_match:
                hypothesis = {
                    "statement": statement_match.group(1).strip(),
                    "type": type_match.group(1).strip() if type_match else "H1",
                    "variables": self._parse_variables(variables_match.group(1).strip() if variables_match else ""),
                    "expected_direction": direction_match.group(1).strip() if direction_match else "Non-directional",
                    "theoretical_basis": basis_match.group(1).strip() if basis_match else "",
                    "testability": testability_match.group(1).strip() if testability_match else ""
                }
                hypotheses.append(hypothesis)

        # Fallback parsing
        if not hypotheses:
            lines = text.strip().split('\n')
            current_hypothesis = {}

            for line in lines:
                line = line.strip()
                if line.startswith('Hypothesis Statement:'):
                    if current_hypothesis:
                        hypotheses.append(current_hypothesis)
                    current_hypothesis = {"statement": line.replace('Hypothesis Statement:', '').strip()}
                elif line.startswith('Type:'):
                    current_hypothesis["type"] = line.replace('Type:', '').strip()
                elif line.startswith('Variables:'):
                    current_hypothesis["variables"] = self._parse_variables(line.replace('Variables:', '').strip())
                elif line.startswith('Expected Direction:'):
                    current_hypothesis["expected_direction"] = line.replace('Expected Direction:', '').strip()
                elif line.startswith('Theoretical Basis:'):
                    current_hypothesis["theoretical_basis"] = line.replace('Theoretical Basis:', '').strip()
                elif line.startswith('Testability:'):
                    current_hypothesis["testability"] = line.replace('Testability:', '').strip()

            if current_hypothesis:
                hypotheses.append(current_hypothesis)

        return hypotheses

    def _parse_variables(self, variables_text: str) -> Dict[str, Any]:
        """Parse variables from text"""
        variables = {}

        # Try to extract IV and DV
        iv_match = re.search(r'independent.*?:\s*(.+?)(?=\n|,|$)', variables_text, re.IGNORECASE)
        dv_match = re.search(r'dependent.*?:\s*(.+?)(?=\n|,|$)', variables_text, re.IGNORECASE)

        if iv_match:
            variables["independent"] = iv_match.group(1).strip()
        if dv_match:
            variables["dependent"] = dv_match.group(1).strip()

        # If not found, try other patterns
        if not variables:
            parts = variables_text.split(',')
            if len(parts) >= 2:
                variables["independent"] = parts[0].strip()
                variables["dependent"] = parts[1].strip()

        return variables

    def _parse_experiment_from_text(self, text: str) -> Dict[str, Any]:
        """Parse experiment design from text"""
        experiment = {
            "type": "Observational",
            "methodology": {},
            "procedure": {},
            "variables_operationalization": {},
            "controls": {},
            "data_analysis_plan": {}
        }

        sections = {
            "EXPERIMENT TYPE": "type",
            "METHODOLOGY": "methodology",
            "PROCEDURE": "procedure",
            "VARIABLES OPERATIONALIZATION": "variables_operationalization",
            "CONTROLS": "controls",
            "DATA ANALYSIS PLAN": "data_analysis_plan"
        }

        current_section = None
        section_content = []

        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Check for section headers
            section_found = False
            for header, section_key in sections.items():
                if line.upper().startswith(header):
                    # Save previous section
                    if current_section and section_content:
                        experiment[current_section] = self._parse_section_content(section_content, current_section)

                    # Start new section
                    current_section = section_key
                    section_content = []
                    section_found = True
                    break

            if not section_found and current_section and line:
                section_content.append(line)

        # Save last section
        if current_section and section_content:
            experiment[current_section] = self._parse_section_content(section_content, current_section)

        return experiment

    def _parse_section_content(self, content_lines: List[str], section: str) -> Dict[str, Any]:
        """Parse content for a specific section"""
        if section == "type":
            return " ".join(content_lines).strip()

        content_text = "\n".join(content_lines)

        # Simple parsing for now
        return {
            "description": content_text,
            "key_points": content_lines[:5]  # First 5 lines as key points
        }

    def _parse_statistical_methods(self, text: str) -> List[Dict[str, Any]]:
        """Parse statistical methods from text"""
        methods = []

        # Split by method
        method_blocks = re.split(r'\n\d+[\.\)]\s+|\nMethod Name:', text)

        for block in method_blocks:
            if not block.strip() or len(block.strip()) < 50:
                continue

            # Extract components
            name_match = re.search(r'Method Name:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            purpose_match = re.search(r'Purpose:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            assumptions_match = re.search(r'Assumptions:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            when_match = re.search(r'When to Use:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            interpretation_match = re.search(r'Interpretation:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
            implementation_match = re.search(r'Software Implementation:\s*(.+?)(?=\n|$)', block, re.IGNORECASE, re.DOTALL)

            if name_match or "test" in block.lower() or "analysis" in block.lower():
                method = {
                    "name": name_match.group(1).strip() if name_match else block.split('\n')[0].strip(),
                    "purpose": purpose_match.group(1).strip() if purpose_match else "",
                    "assumptions": assumptions_match.group(1).strip() if assumptions_match else "",
                    "when_to_use": when_match.group(1).strip() if when_match else "",
                    "interpretation": interpretation_match.group(1).strip() if interpretation_match else "",
                    "implementation": implementation_match.group(1).strip() if implementation_match else ""
                }
                methods.append(method)

        return methods