from typing import List, Dict, Any, Tuple
import asyncio
from datetime import datetime
from loguru import logger
import numpy as np
import re

from backend.agents.base_agent import BaseAgent, AgentMessage
from backend.config import settings


class UncertaintyAgent(BaseAgent):
    """Quantifies uncertainty and confidence for all agent outputs"""

    def __init__(self, agent_id: str = "uncertainty_001"):
        capabilities = [
            "uncertainty_quantification", "confidence_calibration",
            "risk_assessment", "reliability_analysis", "error_propagation"
        ]
        super().__init__(agent_id, "uncertainty", capabilities)
        self.setup_tools()

    def setup_tools(self):
        """Setup uncertainty quantification tools"""
        self.add_tool(
            name="quantify_uncertainty",
            tool_func=self.quantify_uncertainty_tool,
            description="Quantify uncertainty in agent outputs"
        )

        self.add_tool(
            name="calibrate_confidence",
            tool_func=self.calibrate_confidence_tool,
            description="Calibrate confidence scores to be well-calibrated"
        )

        self.add_tool(
            name="assess_reliability",
            tool_func=self.assess_reliability_tool,
            description="Assess reliability of agent outputs"
        )

        self.add_tool(
            name="propagate_errors",
            tool_func=self.propagate_errors_tool,
            description="Propagate errors through multi-agent pipeline"
        )

        self.add_tool(
            name="decide_abstention",
            tool_func=self.decide_abstention_tool,
            description="Decide whether to abstain based on confidence"
        )

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process uncertainty quantification request"""
        self.log(f"Received message: {message.message_type}")

        if message.message_type == "command" and message.content.get("action") == "quantify_uncertainty":
            try:
                agent_output = message.content.get("agent_output", {})
                agent_type = message.content.get("agent_type", "unknown")
                context = message.content.get("context", {})

                if not agent_output:
                    raise ValueError("No agent output provided")

                # Quantify uncertainty
                uncertainty_analysis = await self.quantify_uncertainty(agent_output, agent_type, context)

                # Decide on abstention
                should_abstain = await self.decide_abstention(uncertainty_analysis)

                return AgentMessage(
                    sender=self.agent_id,
                    recipient=message.sender,
                    content={
                        "uncertainty_analysis": uncertainty_analysis,
                        "should_abstain": should_abstain,
                        "abstention_reason": uncertainty_analysis.get("major_concerns", []),
                        "confidence_adjustment": uncertainty_analysis.get("confidence_adjustment", 0),
                        "quantification_date": datetime.now().isoformat()
                    },
                    message_type="result",
                    confidence=uncertainty_analysis.get("overall_confidence", 0.5)
                )

            except Exception as e:
                self.log(f"Uncertainty quantification failed: {e}", "error")
                raise

        return await super().process(message)

    async def quantify_uncertainty(self, agent_output: Dict[str, Any],
                                   agent_type: str,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty in agent output"""
        self.log(f"Quantifying uncertainty for {agent_type} output")

        analysis = {
            "agent_type": agent_type,
            "output_id": agent_output.get("id", ""),
            "quantification_timestamp": datetime.now().isoformat(),
            "confidence_scores": {},
            "uncertainty_sources": [],
            "error_estimates": {},
            "reliability_indicators": {},
            "calibration_status": "uncalibrated",
            "major_concerns": [],
            "minor_concerns": [],
            "overall_confidence": 0.5,
            "confidence_adjustment": 0.0
        }

        # Different quantification based on agent type
        if agent_type == "domain_scout":
            domain_analysis = await self.quantify_domain_uncertainty(agent_output, context)
            analysis.update(domain_analysis)

        elif agent_type == "question_generator":
            question_analysis = await self.quantify_question_uncertainty(agent_output, context)
            analysis.update(question_analysis)

        elif agent_type == "data_alchemist":
            data_analysis = await self.quantify_data_uncertainty(agent_output, context)
            analysis.update(data_analysis)

        elif agent_type == "experiment_designer":
            experiment_analysis = await self.quantify_experiment_uncertainty(agent_output, context)
            analysis.update(experiment_analysis)

        elif agent_type == "critic":
            critique_analysis = await self.quantify_critique_uncertainty(agent_output, context)
            analysis.update(critique_analysis)

        else:
            # Generic uncertainty quantification
            generic_analysis = await self.quantify_generic_uncertainty(agent_output, context)
            analysis.update(generic_analysis)

        # Calibrate confidence scores
        calibrated_analysis = await self.calibrate_confidence(analysis)
        analysis.update(calibrated_analysis)

        # Assess reliability
        reliability = await self.assess_reliability(analysis)
        analysis["reliability_assessment"] = reliability

        # Propagate errors if there's a pipeline context
        if context.get("pipeline_stage"):
            error_propagation = await self.propagate_errors(analysis, context)
            analysis["error_propagation"] = error_propagation

        # Calculate overall confidence
        analysis["overall_confidence"] = self._calculate_overall_confidence(analysis)

        # Determine calibration status
        analysis["calibration_status"] = self._determine_calibration_status(analysis)

        return analysis

    async def quantify_domain_uncertainty(self, domain_output: Dict[str, Any],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty in domain discovery"""
        analysis = {
            "confidence_scores": {},
            "uncertainty_sources": [],
            "error_estimates": {},
            "reliability_indicators": {}
        }

        domains = domain_output.get("domains", [])

        # Calculate confidence for each domain
        domain_confidences = {}
        for domain in domains:
            domain_name = domain.get("name", "unknown")
            confidence = domain.get("confidence", 0.5)

            # Adjust confidence based on evidence
            evidence_count = len(domain.get("evidence", []))
            source_count = len(domain.get("sources", []))
            momentum = domain.get("momentum", 1)

            # Calculate adjusted confidence
            adjusted_confidence = confidence * (
                    0.4 +  # Base confidence from scout
                    (min(evidence_count, 5) / 5) * 0.2 +  # Evidence completeness
                    (min(source_count, 3) / 3) * 0.2 +  # Source diversity
                    (min(momentum, 10) / 10) * 0.2  # Momentum factor
            )

            domain_confidences[domain_name] = min(adjusted_confidence, 1.0)

        analysis["confidence_scores"]["domain_confidences"] = domain_confidences

        # Identify uncertainty sources
        if len(domains) < 3:
            analysis["uncertainty_sources"].append("Limited number of domains discovered")

        for domain in domains:
            if len(domain.get("sources", [])) < 2:
                analysis["uncertainty_sources"].append(
                    f"Domain '{domain.get('name')}' has limited source diversity"
                )

        # Error estimates
        avg_confidence = np.mean(list(domain_confidences.values())) if domain_confidences else 0.5
        analysis["error_estimates"] = {
            "expected_error_rate": 1.0 - avg_confidence,
            "domain_count_error": max(0, 5 - len(domains)) / 5,  # Error for not finding 5 domains
            "source_diversity_error": self._calculate_source_diversity_error(domains)
        }

        # Reliability indicators
        analysis["reliability_indicators"] = {
            "domain_count": len(domains),
            "avg_evidence_per_domain": np.mean([len(d.get("evidence", [])) for d in domains]) if domains else 0,
            "source_variety": len(set(source for d in domains for source in d.get("sources", []))),
            "confidence_consistency": np.std(list(domain_confidences.values())) if domain_confidences else 0
        }

        # Major concerns
        if avg_confidence < 0.6:
            analysis["major_concerns"] = ["Low overall confidence in discovered domains"]
        if len(domains) == 0:
            analysis["major_concerns"] = ["No domains discovered"]

        return analysis

    async def quantify_question_uncertainty(self, question_output: Dict[str, Any],
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty in question generation"""
        analysis = {
            "confidence_scores": {},
            "uncertainty_sources": [],
            "error_estimates": {},
            "reliability_indicators": {}
        }

        questions = question_output.get("questions", [])

        # Calculate confidence for each question
        question_confidences = {}
        for question in questions:
            question_text = question.get("question", "unknown")
            novelty = question.get("novelty_score", 0.5)
            feasibility = question.get("feasibility_score", 0.5)
            overall = question.get("overall_score", 0.5)

            # Calculate uncertainty-adjusted confidence
            # Higher novelty but lower feasibility increases uncertainty
            uncertainty = abs(novelty - feasibility) * 0.5  # Divergence creates uncertainty

            adjusted_confidence = overall * (1.0 - uncertainty)

            question_confidences[question_text[:50] + "..."] = min(adjusted_confidence, 1.0)

        analysis["confidence_scores"]["question_confidences"] = question_confidences

        # Identify uncertainty sources
        for question in questions:
            novelty = question.get("novelty_score", 0.5)
            feasibility = question.get("feasibility_score", 0.5)

            if novelty > 0.8 and feasibility < 0.3:
                analysis["uncertainty_sources"].append(
                    f"Question is novel but not feasible: {question.get('question', '')[:50]}..."
                )
            elif novelty < 0.3 and feasibility > 0.8:
                analysis["uncertainty_sources"].append(
                    f"Question is feasible but not novel: {question.get('question', '')[:50]}..."
                )

        # Error estimates
        if questions:
            novelty_scores = [q.get("novelty_score", 0.5) for q in questions]
            feasibility_scores = [q.get("feasibility_score", 0.5) for q in questions]

            analysis["error_estimates"] = {
                "novelty_uncertainty": np.std(novelty_scores) if len(novelty_scores) > 1 else 0.2,
                "feasibility_uncertainty": np.std(feasibility_scores) if len(feasibility_scores) > 1 else 0.2,
                "consistency_error": self._calculate_consistency_error(questions)
            }

        # Reliability indicators
        analysis["reliability_indicators"] = {
            "question_count": len(questions),
            "avg_novelty": np.mean([q.get("novelty_score", 0.5) for q in questions]) if questions else 0,
            "avg_feasibility": np.mean([q.get("feasibility_score", 0.5) for q in questions]) if questions else 0,
            "novelty_feasibility_correlation": self._calculate_correlation(
                [q.get("novelty_score", 0.5) for q in questions],
                [q.get("feasibility_score", 0.5) for q in questions]
            ) if len(questions) > 1 else 0
        }

        # Major concerns
        if len(questions) == 0:
            analysis["major_concerns"] = ["No questions generated"]
        elif len(questions) < 3:
            analysis["major_concerns"] = ["Insufficient number of questions generated"]

        return analysis

    async def quantify_data_uncertainty(self, data_output: Dict[str, Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty in data acquisition"""
        analysis = {
            "confidence_scores": {},
            "uncertainty_sources": [],
            "error_estimates": {},
            "reliability_indicators": {}
        }

        quality_metrics = data_output.get("quality_metrics", {})
        sources = data_output.get("sources", [])
        data_summary = data_output.get("summary", "")

        # Confidence from quality metrics
        overall_quality = quality_metrics.get("overall_quality", 0.5)
        analysis["confidence_scores"]["data_quality_confidence"] = overall_quality

        # Source reliability confidence
        source_confidence = self._calculate_source_confidence(sources)
        analysis["confidence_scores"]["source_reliability_confidence"] = source_confidence

        # Data completeness confidence
        completeness = quality_metrics.get("data_completeness", 0.5)
        analysis["confidence_scores"]["completeness_confidence"] = completeness

        # Combined data confidence
        analysis["confidence_scores"]["overall_data_confidence"] = (
                overall_quality * 0.4 +
                source_confidence * 0.3 +
                completeness * 0.3
        )

        # Identify uncertainty sources
        if len(sources) < settings.min_data_sources:
            analysis["uncertainty_sources"].append(
                f"Insufficient data sources: {len(sources)} < {settings.min_data_sources}"
            )

        if overall_quality < 0.6:
            analysis["uncertainty_sources"].append("Low overall data quality")

        # Check for data source diversity
        source_types = [s.get("type", "").lower() for s in sources]
        unique_types = len(set(source_types))
        if unique_types < 2:
            analysis["uncertainty_sources"].append("Limited data source type diversity")

        # Error estimates
        analysis["error_estimates"] = {
            "data_quality_error": 1.0 - overall_quality,
            "source_reliability_error": 1.0 - source_confidence,
            "completeness_error": 1.0 - completeness,
            "expected_missing_data_rate": 0.3  # Conservative estimate
        }

        # Reliability indicators
        analysis["reliability_indicators"] = {
            "source_count": len(sources),
            "source_type_diversity": unique_types,
            "quality_score_consistency": quality_metrics.get("combined_quality", 0.5) if isinstance(quality_metrics,
                                                                                                    dict) else 0.5,
            "data_volume": len(data_summary) if data_summary else 0
        }

        # Major concerns
        if len(sources) < settings.min_data_sources:
            analysis["major_concerns"] = [f"Need at least {settings.min_data_sources} data sources"]
        if overall_quality < 0.4:
            analysis["major_concerns"] = ["Very low data quality"]

        return analysis

    async def quantify_experiment_uncertainty(self, experiment_output: Dict[str, Any],
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty in experiment design"""
        analysis = {
            "confidence_scores": {},
            "uncertainty_sources": [],
            "error_estimates": {},
            "reliability_indicators": {}
        }

        primary_experiment = experiment_output.get("primary_experiment", {})
        feasibility = experiment_output.get("feasibility_assessment", {})

        if not primary_experiment:
            analysis["major_concerns"] = ["No experiment designed"]
            return analysis

        # Confidence from feasibility
        overall_feasibility = feasibility.get("overall_feasibility", 0.5)
        analysis["confidence_scores"]["feasibility_confidence"] = overall_feasibility

        # Statistical methods confidence
        statistical_methods = primary_experiment.get("statistical_methods", [])
        method_confidence = self._calculate_method_confidence(statistical_methods)
        analysis["confidence_scores"]["statistical_confidence"] = method_confidence

        # Sample size confidence
        sample_size = primary_experiment.get("sample_size", {})
        sample_confidence = self._calculate_sample_confidence(sample_size)
        analysis["confidence_scores"]["sample_size_confidence"] = sample_confidence

        # Complexity-adjusted confidence
        complexity = primary_experiment.get("complexity", 0.5)
        complexity_factor = 1.0 - (complexity * 0.3)  # Higher complexity reduces confidence

        # Combined experiment confidence
        analysis["confidence_scores"]["overall_experiment_confidence"] = (
                overall_feasibility * 0.3 +
                method_confidence * 0.3 +
                sample_confidence * 0.2 +
                complexity_factor * 0.2
        )

        # Identify uncertainty sources
        if overall_feasibility < 0.6:
            analysis["uncertainty_sources"].append("Low feasibility score")

        if method_confidence < 0.6:
            analysis["uncertainty_sources"].append("Questionable statistical methods")

        if sample_confidence < 0.6:
            analysis["uncertainty_sources"].append("Inadequate sample size")

        if complexity > 0.7:
            analysis["uncertainty_sources"].append("High experiment complexity")

        # Error estimates
        analysis["error_estimates"] = {
            "feasibility_error": 1.0 - overall_feasibility,
            "statistical_error": 1.0 - method_confidence,
            "sample_error": 1.0 - sample_confidence,
            "type_I_error_rate": 0.05,  # Standard alpha
            "type_II_error_rate": 0.2  # Standard beta (80% power)
        }

        # Reliability indicators
        analysis["reliability_indicators"] = {
            "experiment_complexity": complexity,
            "statistical_method_count": len(statistical_methods),
            "feasibility_score_breakdown": {
                k: v for k, v in feasibility.items()
                if isinstance(v, (int, float)) and "feasibility" in k.lower()
            },
            "sample_size_adequacy": sample_size.get("feasibility", "unknown")
        }

        # Major concerns
        if overall_feasibility < 0.4:
            analysis["major_concerns"] = ["Experiment not feasible"]
        if sample_confidence < 0.4:
            analysis["major_concerns"] = ["Sample size severely inadequate"]

        return analysis

    async def quantify_critique_uncertainty(self, critique_output: Dict[str, Any],
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty in critique"""
        analysis = {
            "confidence_scores": {},
            "uncertainty_sources": [],
            "error_estimates": {},
            "reliability_indicators": {}
        }

        critique_confidence = critique_output.get("confidence", 0.5)
        requires_iteration = critique_output.get("requires_iteration", False)
        major_issues = critique_output.get("iteration_reason", [])

        # Critique confidence
        analysis["confidence_scores"]["critique_confidence"] = critique_confidence

        # Severity-adjusted confidence
        severity_factor = 1.0
        if requires_iteration and major_issues:
            severity_factor = 0.7  # More severe critique has higher confidence
        elif not requires_iteration and not major_issues:
            severity_factor = 0.5  # Mild critique has moderate confidence

        # Combined critique confidence
        analysis["confidence_scores"]["overall_critique_confidence"] = critique_confidence * severity_factor

        # Identify uncertainty sources
        if critique_confidence < 0.6:
            analysis["uncertainty_sources"].append("Low confidence in critique")

        if not major_issues and requires_iteration:
            analysis["uncertainty_sources"].append("Inconsistent critique: iteration required but no major issues")

        # Error estimates
        analysis["error_estimates"] = {
            "critique_error": 1.0 - critique_confidence,
            "false_positive_rate": 0.1,  # Critique finds issue when none exists
            "false_negative_rate": 0.2,  # Critique misses actual issue
            "severity_misclassification": 0.15  # Misclassifying issue severity
        }

        # Reliability indicators
        analysis["reliability_indicators"] = {
            "critique_severity": "high" if requires_iteration else "low",
            "major_issue_count": len(major_issues),
            "critique_consistency": 0.7,  # Estimated consistency
            "iteration_required": requires_iteration
        }

        # Major concerns
        if critique_confidence < 0.3:
            analysis["major_concerns"] = ["Very low confidence in critique validity"]

        return analysis

    async def quantify_generic_uncertainty(self, agent_output: Dict[str, Any],
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty for generic agent output"""
        analysis = {
            "confidence_scores": {},
            "uncertainty_sources": [],
            "error_estimates": {},
            "reliability_indicators": {}
        }

        # Extract confidence if present
        output_confidence = agent_output.get("confidence", 0.5)

        # Calculate basic uncertainty metrics
        analysis["confidence_scores"]["output_confidence"] = output_confidence

        # Identify uncertainty from output characteristics
        output_str = str(agent_output)
        output_length = len(output_str)

        # Longer outputs might be more certain (more detail)
        length_factor = min(output_length / 1000, 1.0)  # Cap at 1000 chars

        # Check for indicators of uncertainty in text
        uncertainty_indicators = ["uncertain", "maybe", "possibly", "perhaps",
                                  "likely", "probably", "estimate", "approximate"]

        indicator_count = sum(1 for indicator in uncertainty_indicators
                              if indicator in output_str.lower())
        indicator_factor = 1.0 - (indicator_count * 0.1)

        # Combined confidence
        analysis["confidence_scores"]["adjusted_confidence"] = (
                output_confidence * 0.5 +
                length_factor * 0.3 +
                indicator_factor * 0.2
        )

        # Uncertainty sources
        if indicator_count > 3:
            analysis["uncertainty_sources"].append(f"Many uncertainty indicators found ({indicator_count})")

        if output_length < 100:
            analysis["uncertainty_sources"].append("Output is very brief")

        # Error estimates
        analysis["error_estimates"] = {
            "base_error": 1.0 - output_confidence,
            "length_error": 1.0 - length_factor,
            "indicator_error": 1.0 - indicator_factor
        }

        # Reliability indicators
        analysis["reliability_indicators"] = {
            "output_length": output_length,
            "uncertainty_indicators": indicator_count,
            "has_confidence_score": "confidence" in agent_output
        }

        return analysis

    async def calibrate_confidence(self, uncertainty_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrate confidence scores"""
        calibrated = uncertainty_analysis.copy()

        # Get current confidence scores
        confidence_scores = uncertainty_analysis.get("confidence_scores", {})

        if not confidence_scores:
            calibrated["calibration_status"] = "no_scores"
            return calibrated

        # Apply calibration adjustments
        calibrated_scores = {}
        adjustments = {}

        for key, score in confidence_scores.items():
            if isinstance(score, (int, float)):
                # Apply Bayesian calibration
                calibrated_score = self._bayesian_calibration(score)
                calibrated_scores[key] = calibrated_score
                adjustments[key] = calibrated_score - score

        calibrated["confidence_scores"] = {**confidence_scores, **calibrated_scores}
        calibrated["confidence_adjustments"] = adjustments

        # Calculate overall calibration adjustment
        if adjustments:
            avg_adjustment = np.mean(list(adjustments.values()))
            calibrated["confidence_adjustment"] = avg_adjustment

        return calibrated

    async def assess_reliability(self, uncertainty_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess reliability of agent output"""
        reliability = {
            "overall_reliability": 0.5,
            "reliability_factors": {},
            "reliability_concerns": [],
            "reliability_strengths": []
        }

        # Extract relevant data
        confidence_scores = uncertainty_analysis.get("confidence_scores", {})
        reliability_indicators = uncertainty_analysis.get("reliability_indicators", {})
        uncertainty_sources = uncertainty_analysis.get("uncertainty_sources", [])

        # Calculate reliability factors
        factors = {}

        # 1. Confidence consistency
        if confidence_scores:
            scores = [v for v in confidence_scores.values() if isinstance(v, (int, float))]
            if scores:
                factors["confidence_consistency"] = 1.0 - np.std(scores)  # Lower std = more consistent

        # 2. Indicator completeness
        if reliability_indicators:
            indicator_count = len(reliability_indicators)
            factors["indicator_completeness"] = min(indicator_count / 5, 1.0)  # 5 indicators = complete

        # 3. Uncertainty source impact
        uncertainty_count = len(uncertainty_sources)
        factors["uncertainty_impact"] = max(0, 1.0 - (uncertainty_count * 0.1))

        # 4. Data volume (if available)
        if "data_volume" in reliability_indicators:
            volume = reliability_indicators["data_volume"]
            factors["data_sufficiency"] = min(volume / 1000, 1.0)  # 1000 chars = sufficient

        # Calculate overall reliability
        if factors:
            reliability["overall_reliability"] = np.mean(list(factors.values()))
            reliability["reliability_factors"] = factors

        # Identify concerns and strengths
        if uncertainty_count > 3:
            reliability["reliability_concerns"].append(f"Many uncertainty sources: {uncertainty_count}")

        if factors.get("confidence_consistency", 1.0) < 0.7:
            reliability["reliability_concerns"].append("Inconsistent confidence scores")

        if factors.get("indicator_completeness", 0) > 0.8:
            reliability["reliability_strengths"].append("Comprehensive reliability indicators")

        if reliability["overall_reliability"] > 0.7:
            reliability["reliability_strengths"].append("High overall reliability")

        return reliability

    async def propagate_errors(self, uncertainty_analysis: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate errors through pipeline"""
        propagation = {
            "pipeline_stage": context.get("pipeline_stage", "unknown"),
            "input_errors": {},
            "propagated_errors": {},
            "cumulative_error": 0.5,
            "error_amplification_factor": 1.0
        }

        # Get error estimates
        error_estimates = uncertainty_analysis.get("error_estimates", {})

        if not error_estimates:
            return propagation

        # Calculate base errors
        base_errors = []
        for key, error in error_estimates.items():
            if isinstance(error, (int, float)):
                base_errors.append(error)
                propagation["input_errors"][key] = error

        if base_errors:
            # Calculate average base error
            avg_base_error = np.mean(base_errors)

            # Apply error propagation (simplified)
            # Each stage amplifies error by 10%
            stage = context.get("pipeline_stage_number", 1)
            amplification = 1.0 + (stage * 0.1)

            propagated_error = avg_base_error * amplification

            propagation["cumulative_error"] = min(propagated_error, 1.0)
            propagation["error_amplification_factor"] = amplification

            # Calculate propagated errors for each type
            for key, error in error_estimates.items():
                if isinstance(error, (int, float)):
                    propagation["propagated_errors"][key] = min(error * amplification, 1.0)

        return propagation

    async def decide_abstention(self, uncertainty_analysis: Dict[str, Any]) -> bool:
        """Decide whether to abstain based on confidence"""
        overall_confidence = uncertainty_analysis.get("overall_confidence", 0.5)
        major_concerns = uncertainty_analysis.get("major_concerns", [])
        calibration_status = uncertainty_analysis.get("calibration_status", "uncalibrated")

        # Rule 1: Confidence below threshold
        if overall_confidence < settings.min_confidence_threshold:
            self.log(f"Abstaining: Confidence {overall_confidence:.2f} < threshold {settings.min_confidence_threshold}")
            return True

        # Rule 2: Major concerns present
        if major_concerns:
            self.log(f"Abstaining: {len(major_concerns)} major concerns")
            return True

        # Rule 3: Poor calibration
        if calibration_status == "poor":
            self.log("Abstaining: Poor calibration status")
            return True

        # Rule 4: Check reliability
        reliability = uncertainty_analysis.get("reliability_assessment", {}).get("overall_reliability", 0.5)
        if reliability < 0.4:
            self.log(f"Abstaining: Low reliability {reliability:.2f}")
            return True

        # Default: Don't abstain
        return False

    async def quantify_uncertainty_tool(self, agent_output: Dict[str, Any],
                                        agent_type: str,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for uncertainty quantification"""
        return await self.quantify_uncertainty(agent_output, agent_type, context)

    async def calibrate_confidence_tool(self, uncertainty_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for confidence calibration"""
        return await self.calibrate_confidence(uncertainty_analysis)

    async def assess_reliability_tool(self, uncertainty_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for reliability assessment"""
        return await self.assess_reliability(uncertainty_analysis)

    async def propagate_errors_tool(self, uncertainty_analysis: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for error propagation"""
        return await self.propagate_errors(uncertainty_analysis, context)

    async def decide_abstention_tool(self, uncertainty_analysis: Dict[str, Any]) -> bool:
        """Tool wrapper for abstention decision"""
        return await self.decide_abstention(uncertainty_analysis)

    def _calculate_source_diversity_error(self, domains: List[Dict[str, Any]]) -> float:
        """Calculate error due to lack of source diversity"""
        if not domains:
            return 1.0

        source_counts = []
        for domain in domains:
            sources = domain.get("sources", [])
            source_counts.append(len(sources))

        avg_sources = np.mean(source_counts) if source_counts else 0
        # Error is inverse of average sources (capped at 3 sources = no error)
        return max(0, 1.0 - (avg_sources / 3))

    def _calculate_consistency_error(self, questions: List[Dict[str, Any]]) -> float:
        """Calculate consistency error in questions"""
        if len(questions) < 2:
            return 0.5  # Can't assess consistency with single question

        novelty_scores = [q.get("novelty_score", 0.5) for q in questions]
        feasibility_scores = [q.get("feasibility_score", 0.5) for q in questions]

        novelty_std = np.std(novelty_scores)
        feasibility_std = np.std(feasibility_scores)

        # Average standard deviation as consistency error
        avg_std = (novelty_std + feasibility_std) / 2

        return min(avg_std, 1.0)

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation between two lists"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        try:
            correlation = np.corrcoef(x, y)[0, 1]
            # Handle NaN
            if np.isnan(correlation):
                return 0.0
            return correlation
        except:
            return 0.0

    def _calculate_source_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on source reliability"""
        if not sources:
            return 0.3  # Low confidence without sources

        credible_domains = [".gov", ".edu", ".ac.", "arxiv", "research", "university"]
        credibility_scores = []

        for source in sources:
            score = 0.5  # Base

            # URL credibility
            url = source.get("url", "").lower()
            for domain in credible_domains:
                if domain in url:
                    score = 0.8
                    break

            # Type credibility
            source_type = source.get("type", "").lower()
            if "academic" in source_type or "paper" in source_type:
                score = max(score, 0.9)
            elif "government" in source_type:
                score = max(score, 0.8)
            elif "dataset" in source_type and "kaggle" in url:
                score = max(score, 0.7)

            credibility_scores.append(score)

        return np.mean(credibility_scores) if credibility_scores else 0.5

    def _calculate_method_confidence(self, statistical_methods: List[Dict[str, Any]]) -> float:
        """Calculate confidence in statistical methods"""
        if not statistical_methods:
            return 0.3

        method_scores = []
        for method in statistical_methods:
            if isinstance(method, dict):
                appropriateness = method.get("appropriateness_score", 0.5)
                complexity = method.get("implementation_complexity", "medium")

                # Adjust for complexity
                complexity_map = {"low": 1.0, "medium": 0.8, "high": 0.6}
                complexity_factor = complexity_map.get(complexity, 0.7)

                method_scores.append(appropriateness * complexity_factor)

        return np.mean(method_scores) if method_scores else 0.5

    def _calculate_sample_confidence(self, sample_size_info: Dict[str, Any]) -> float:
        """Calculate confidence in sample size"""
        if not sample_size_info:
            return 0.3

        feasibility = sample_size_info.get("feasibility", "medium")
        feasibility_map = {"high": 0.9, "medium": 0.7, "low": 0.4, "very low": 0.2}

        return feasibility_map.get(feasibility, 0.5)

    def _bayesian_calibration(self, score: float) -> float:
        """Apply Bayesian calibration to confidence score"""
        # Simple calibration: adjust towards 0.5 based on distance
        # Scores near 0.5 are better calibrated

        distance = abs(score - 0.5)

        # If score is extreme (close to 0 or 1), pull it towards 0.5
        if distance > 0.4:  # Very confident or very unconfident
            adjustment = 0.15 * (1 if score < 0.5 else -1)
            return score + adjustment
        elif distance > 0.2:  # Moderately confident
            adjustment = 0.05 * (1 if score < 0.5 else -1)
            return score + adjustment
        else:
            return score  # Already well-calibrated

    def _calculate_overall_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence from analysis"""
        confidence_scores = analysis.get("confidence_scores", {})

        if not confidence_scores:
            return 0.5

        # Extract numeric confidence scores
        scores = []
        for key, value in confidence_scores.items():
            if isinstance(value, dict):
                # If it's a dict of scores, take average
                dict_scores = [v for v in value.values() if isinstance(v, (int, float))]
                if dict_scores:
                    scores.append(np.mean(dict_scores))
            elif isinstance(value, (int, float)):
                scores.append(value)

        if not scores:
            return 0.5

        # Calculate weighted overall confidence
        # Give more weight to overall_X_confidence scores
        weighted_scores = []
        for i, score in enumerate(scores):
            key = list(confidence_scores.keys())[i]
            if "overall" in key.lower():
                weight = 2.0
            else:
                weight = 1.0
            weighted_scores.append(score * weight)

        if weighted_scores:
            overall = np.mean(weighted_scores)
        else:
            overall = np.mean(scores)

        # Adjust based on major concerns
        major_concerns = len(analysis.get("major_concerns", []))
        if major_concerns > 0:
            overall *= 0.7  # Reduce confidence by 30% for major concerns

        return min(max(overall, 0.0), 1.0)

    def _determine_calibration_status(self, analysis: Dict[str, Any]) -> str:
        """Determine calibration status"""
        confidence_scores = analysis.get("confidence_scores", {})

        if not confidence_scores:
            return "unknown"

        # Extract all confidence scores
        scores = []
        for value in confidence_scores.values():
            if isinstance(value, dict):
                scores.extend([v for v in value.values() if isinstance(v, (int, float))])
            elif isinstance(value, (int, float)):
                scores.append(value)

        if not scores:
            return "unknown"

        # Check if scores are well-calibrated (not too extreme)
        extreme_count = sum(1 for score in scores if score < 0.2 or score > 0.8)
        extreme_ratio = extreme_count / len(scores)

        if extreme_ratio > 0.7:
            return "poor"  # Too many extreme scores
        elif extreme_ratio > 0.4:
            return "moderate"
        else:
            return "good"