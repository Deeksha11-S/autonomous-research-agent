from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from loguru import logger
from enum import Enum
import json

from backend.agents.base_agent import BaseAgent, AgentMessage
from backend.agents.domain_scout import DomainScoutAgent
from backend.agents.question_generator import QuestionGeneratorAgent
from backend.agents.data_alchemist import DataAlchemistAgent
from backend.agents.experiment_designer import ExperimentDesignerAgent
from backend.agents.critic import CriticAgent
from backend.agents.uncertainty import UncertaintyAgent
from backend.config import settings, AgentRole


class ResearchStage(Enum):
    """Research pipeline stages"""
    INITIALIZATION = "initialization"
    DOMAIN_DISCOVERY = "domain_discovery"
    QUESTION_GENERATION = "question_generation"
    DATA_ACQUISITION = "data_acquisition"
    EXPERIMENT_DESIGN = "experiment_design"
    CRITIQUE = "critique"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    PAPER_GENERATION = "paper_generation"
    COMPLETION = "completion"


class ResearchOrchestrator(BaseAgent):
    """Orchestrates the complete research pipeline"""

    def __init__(self, session_id: str):
        capabilities = [
            "workflow_orchestration", "agent_coordination",
            "conflict_resolution", "memory_management", "iteration_control"
        ]
        super().__init__("orchestrator_001", "orchestrator", capabilities)

        self.session_id = session_id
        self.research_stage = ResearchStage.INITIALIZATION
        self.current_iteration = 0
        self.max_iterations = settings.max_iterations
        self.research_context = {}
        self.agent_pool = {}
        self.research_log = []
        self.progress_callback = None

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents in the system"""
        self.log("Initializing agents...")

        if AgentRole.DOMAIN_SCOUT in settings.enable_agents:
            self.agent_pool[AgentRole.DOMAIN_SCOUT] = DomainScoutAgent()

        if AgentRole.QUESTION_GENERATOR in settings.enable_agents:
            self.agent_pool[AgentRole.QUESTION_GENERATOR] = QuestionGeneratorAgent()

        if AgentRole.DATA_ALCHEMIST in settings.enable_agents:
            self.agent_pool[AgentRole.DATA_ALCHEMIST] = DataAlchemistAgent()

        if AgentRole.EXPERIMENT_DESIGNER in settings.enable_agents:
            self.agent_pool[AgentRole.EXPERIMENT_DESIGNER] = ExperimentDesignerAgent()

        if AgentRole.CRITIC in settings.enable_agents:
            self.agent_pool[AgentRole.CRITIC] = CriticAgent()

        if AgentRole.UNCERTAINTY in settings.enable_agents:
            self.agent_pool[AgentRole.UNCERTAINTY] = UncertaintyAgent()

        self.log(f"Initialized {len(self.agent_pool)} agents")

    async def run_full_pipeline(self, progress_callback=None) -> Dict[str, Any]:
        """Run the complete research pipeline"""
        self.progress_callback = progress_callback
        self.research_stage = ResearchStage.INITIALIZATION

        try:
            self._log_progress("Starting autonomous research pipeline...", 0)

            # Main research loop with iterations
            while self.current_iteration < self.max_iterations:
                iteration_result = await self._run_iteration()

                # Check if we should continue
                if self._should_continue_research(iteration_result):
                    self.current_iteration += 1
                    self._log_progress(f"Starting iteration {self.current_iteration + 1}",
                                       (self.current_iteration / self.max_iterations) * 80)
                else:
                    break

            # Generate final paper
            self._log_progress("Generating research paper...", 90)
            final_paper = await self._generate_research_paper()

            self._log_progress("Research complete!", 100)

            return {
                "session_id": self.session_id,
                "final_paper": final_paper,
                "research_log": self.research_log,
                "total_iterations": self.current_iteration,
                "completion_time": datetime.now().isoformat(),
                "research_context": self.research_context
            }

        except Exception as e:
            self.log(f"Research pipeline failed: {e}", "error")
            raise

    async def _run_iteration(self) -> Dict[str, Any]:
        """Run a single research iteration"""
        iteration_log = {
            "iteration": self.current_iteration,
            "start_time": datetime.now().isoformat(),
            "stages": [],
            "results": {}
        }

        try:
            # Stage 1: Domain Discovery
            self.research_stage = ResearchStage.DOMAIN_DISCOVERY
            self._log_progress("Discovering emerging domains...", 10)

            domains = await self._run_domain_discovery()
            iteration_log["stages"].append({
                "stage": "domain_discovery",
                "result": domains
            })

            if not domains or "error" in domains:
                raise ValueError("Domain discovery failed")

            # Stage 2: Question Generation
            self.research_stage = ResearchStage.QUESTION_GENERATION
            self._log_progress("Generating research questions...", 30)

            questions = await self._run_question_generation(domains)
            iteration_log["stages"].append({
                "stage": "question_generation",
                "result": questions
            })

            if not questions or "error" in questions:
                raise ValueError("Question generation failed")

            # Stage 3: Data Acquisition
            self.research_stage = ResearchStage.DATA_ACQUISITION
            self._log_progress("Acquiring research data...", 50)

            data = await self._run_data_acquisition(questions, domains)
            iteration_log["stages"].append({
                "stage": "data_acquisition",
                "result": data
            })

            if not data or "error" in data:
                raise ValueError("Data acquisition failed")

            # Stage 4: Experiment Design
            self.research_stage = ResearchStage.EXPERIMENT_DESIGN
            self._log_progress("Designing experiments...", 70)

            experiments = await self._run_experiment_design(questions, data)
            iteration_log["stages"].append({
                "stage": "experiment_design",
                "result": experiments
            })

            if not experiments or "error" in experiments:
                raise ValueError("Experiment design failed")

            # Stage 5: Critique
            self.research_stage = ResearchStage.CRITIQUE
            self._log_progress("Critiquing research design...", 85)

            critique = await self._run_critique({
                "domains": domains,
                "questions": questions,
                "data": data,
                "experiments": experiments
            })
            iteration_log["stages"].append({
                "stage": "critique",
                "result": critique
            })

            # Stage 6: Uncertainty Quantification
            self.research_stage = ResearchStage.UNCERTAINTY_QUANTIFICATION
            self._log_progress("Quantifying uncertainty...", 95)

            uncertainty = await self._run_uncertainty_quantification({
                "domains": domains,
                "questions": questions,
                "data": data,
                "experiments": experiments,
                "critique": critique
            })
            iteration_log["stages"].append({
                "stage": "uncertainty_quantification",
                "result": uncertainty
            })

            # Update research context
            self.research_context.update({
                "iteration": self.current_iteration,
                "domains": domains,
                "questions": questions,
                "data": data,
                "experiments": experiments,
                "critique": critique,
                "uncertainty": uncertainty
            })

            iteration_log["end_time"] = datetime.now().isoformat()
            iteration_log["success"] = True

            self.research_log.append(iteration_log)

            return iteration_log

        except Exception as e:
            iteration_log["error"] = str(e)
            iteration_log["end_time"] = datetime.now().isoformat()
            iteration_log["success"] = False

            self.research_log.append(iteration_log)
            self.log(f"Iteration {self.current_iteration} failed: {e}", "error")

            return iteration_log

    async def _run_domain_discovery(self) -> Dict[str, Any]:
        """Run domain discovery stage"""
        if AgentRole.DOMAIN_SCOUT not in self.agent_pool:
            return {"error": "Domain scout agent not available"}

        scout = self.agent_pool[AgentRole.DOMAIN_SCOUT]

        try:
            message = AgentMessage(
                sender=self.agent_id,
                recipient=scout.agent_id,
                content={
                    "action": "discover_domains",
                    "context": self.research_context
                },
                message_type="command"
            )

            response = await scout.process(message)

            # Check uncertainty before accepting
            if AgentRole.UNCERTAINTY in self.agent_pool:
                uncertainty = self.agent_pool[AgentRole.UNCERTAINTY]

                uncertainty_msg = AgentMessage(
                    sender=self.agent_id,
                    recipient=uncertainty.agent_id,
                    content={
                        "action": "quantify_uncertainty",
                        "agent_output": response.content,
                        "agent_type": "domain_scout",
                        "context": self.research_context
                    },
                    message_type="command"
                )

                uncertainty_response = await uncertainty.process(uncertainty_msg)

                if uncertainty_response.content.get("should_abstain", False):
                    self.log("Domain scout should abstain - low confidence", "warning")
                    return {"error": "Low confidence in domain discovery"}

            return response.content

        except Exception as e:
            self.log(f"Domain discovery failed: {e}", "error")
            return {"error": str(e)}

    async def _run_question_generation(self, domains: Dict[str, Any]) -> Dict[str, Any]:
        """Run question generation stage"""
        if AgentRole.QUESTION_GENERATOR not in self.agent_pool:
            return {"error": "Question generator agent not available"}

        # Select the best domain
        selected_domain = self._select_best_domain(domains)
        if not selected_domain:
            return {"error": "No suitable domain selected"}

        generator = self.agent_pool[AgentRole.QUESTION_GENERATOR]

        try:
            message = AgentMessage(
                sender=self.agent_id,
                recipient=generator.agent_id,
                content={
                    "action": "generate_questions",
                    "domain_info": selected_domain,
                    "context": self.research_context
                },
                message_type="command"
            )

            response = await generator.process(message)

            # Uncertainty check
            if AgentRole.UNCERTAINTY in self.agent_pool:
                uncertainty = self.agent_pool[AgentRole.UNCERTAINTY]

                uncertainty_msg = AgentMessage(
                    sender=self.agent_id,
                    recipient=uncertainty.agent_id,
                    content={
                        "action": "quantify_uncertainty",
                        "agent_output": response.content,
                        "agent_type": "question_generator",
                        "context": {**self.research_context, "selected_domain": selected_domain}
                    },
                    message_type="command"
                )

                uncertainty_response = await uncertainty.process(uncertainty_msg)

                if uncertainty_response.content.get("should_abstain", False):
                    self.log("Question generator should abstain - low confidence", "warning")
                    return {"error": "Low confidence in question generation"}

            return response.content

        except Exception as e:
            self.log(f"Question generation failed: {e}", "error")
            return {"error": str(e)}

    async def _run_data_acquisition(self, questions: Dict[str, Any],
                                    domains: Dict[str, Any]) -> Dict[str, Any]:
        """Run data acquisition stage"""
        if AgentRole.DATA_ALCHEMIST not in self.agent_pool:
            return {"error": "Data alchemist agent not available"}

        # Select the best question
        selected_question = self._select_best_question(questions)
        if not selected_question:
            return {"error": "No suitable question selected"}

        alchemist = self.agent_pool[AgentRole.DATA_ALCHEMIST]

        try:
            message = AgentMessage(
                sender=self.agent_id,
                recipient=alchemist.agent_id,
                content={
                    "action": "acquire_data",
                    "research_question": selected_question,
                    "domain_info": domains,
                    "context": self.research_context
                },
                message_type="command"
            )

            response = await alchemist.process(message)

            # Uncertainty check
            if AgentRole.UNCERTAINTY in self.agent_pool:
                uncertainty = self.agent_pool[AgentRole.UNCERTAINTY]

                uncertainty_msg = AgentMessage(
                    sender=self.agent_id,
                    recipient=uncertainty.agent_id,
                    content={
                        "action": "quantify_uncertainty",
                        "agent_output": response.content,
                        "agent_type": "data_alchemist",
                        "context": {
                            **self.research_context,
                            "selected_question": selected_question
                        }
                    },
                    message_type="command"
                )

                uncertainty_response = await uncertainty.process(uncertainty_msg)

                if uncertainty_response.content.get("should_abstain", False):
                    self.log("Data alchemist should abstain - low confidence", "warning")
                    return {"error": "Low confidence in data acquisition"}

            return response.content

        except Exception as e:
            self.log(f"Data acquisition failed: {e}", "error")
            return {"error": str(e)}

    async def _run_experiment_design(self, questions: Dict[str, Any],
                                     data: Dict[str, Any]) -> Dict[str, Any]:
        """Run experiment design stage"""
        if AgentRole.EXPERIMENT_DESIGNER not in self.agent_pool:
            return {"error": "Experiment designer agent not available"}

        # Use the same question as data acquisition
        selected_question = questions.get("questions", [{}])[0] if questions.get("questions") else {}

        designer = self.agent_pool[AgentRole.EXPERIMENT_DESIGNER]

        try:
            message = AgentMessage(
                sender=self.agent_id,
                recipient=designer.agent_id,
                content={
                    "action": "design_experiment",
                    "research_question": selected_question,
                    "data_summary": data,
                    "context": self.research_context
                },
                message_type="command"
            )

            response = await designer.process(message)

            # Uncertainty check
            if AgentRole.UNCERTAINTY in self.agent_pool:
                uncertainty = self.agent_pool[AgentRole.UNCERTAINTY]

                uncertainty_msg = AgentMessage(
                    sender=self.agent_id,
                    recipient=uncertainty.agent_id,
                    content={
                        "action": "quantify_uncertainty",
                        "agent_output": response.content,
                        "agent_type": "experiment_designer",
                        "context": {
                            **self.research_context,
                            "selected_question": selected_question,
                            "data_summary": data
                        }
                    },
                    message_type="command"
                )

                uncertainty_response = await uncertainty.process(uncertainty_msg)

                if uncertainty_response.content.get("should_abstain", False):
                    self.log("Experiment designer should abstain - low confidence", "warning")
                    return {"error": "Low confidence in experiment design"}

            return response.content

        except Exception as e:
            self.log(f"Experiment design failed: {e}", "error")
            return {"error": str(e)}

    async def _run_critique(self, research_components: Dict[str, Any]) -> Dict[str, Any]:
        """Run critique stage"""
        if AgentRole.CRITIC not in self.agent_pool:
            return {"error": "Critic agent not available"}

        critic = self.agent_pool[AgentRole.CRITIC]

        try:
            # Critique the experiment design
            experiments = research_components.get("experiments", {})
            primary_experiment = experiments.get("primary_experiment", {})

            if not primary_experiment:
                return {"error": "No experiment to critique"}

            message = AgentMessage(
                sender=self.agent_id,
                recipient=critic.agent_id,
                content={
                    "action": "critique",
                    "target_component": primary_experiment,
                    "component_type": "experiment",
                    "research_context": {
                        **self.research_context,
                        **research_components
                    }
                },
                message_type="command"
            )

            response = await critic.process(message)

            # Uncertainty check
            if AgentRole.UNCERTAINTY in self.agent_pool:
                uncertainty = self.agent_pool[AgentRole.UNCERTAINTY]

                uncertainty_msg = AgentMessage(
                    sender=self.agent_id,
                    recipient=uncertainty.agent_id,
                    content={
                        "action": "quantify_uncertainty",
                        "agent_output": response.content,
                        "agent_type": "critic",
                        "context": {
                            **self.research_context,
                            **research_components
                        }
                    },
                    message_type="command"
                )

                uncertainty_response = await uncertainty.process(uncertainty_msg)

                if uncertainty_response.content.get("should_abstain", False):
                    self.log("Critic should abstain - low confidence", "warning")
                    return {"error": "Low confidence in critique"}

            return response.content

        except Exception as e:
            self.log(f"Critique failed: {e}", "error")
            return {"error": str(e)}

    async def _run_uncertainty_quantification(self, all_components: Dict[str, Any]) -> Dict[str, Any]:
        """Run uncertainty quantification for all components"""
        if AgentRole.UNCERTAINTY not in self.agent_pool:
            return {"error": "Uncertainty agent not available"}

        uncertainty = self.agent_pool[AgentRole.UNCERTAINTY]

        try:
            # Quantify uncertainty for the overall research
            overall_assessment = {
                "domains": all_components.get("domains", {}),
                "questions": all_components.get("questions", {}),
                "data": all_components.get("data", {}),
                "experiments": all_components.get("experiments", {}),
                "critique": all_components.get("critique", {})
            }

            message = AgentMessage(
                sender=self.agent_id,
                recipient=uncertainty.agent_id,
                content={
                    "action": "quantify_uncertainty",
                    "agent_output": overall_assessment,
                    "agent_type": "orchestrator",
                    "context": {
                        **self.research_context,
                        "pipeline_stage": "final",
                        "pipeline_stage_number": self.current_iteration
                    }
                },
                message_type="command"
            )

            response = await uncertainty.process(message)

            return response.content

        except Exception as e:
            self.log(f"Uncertainty quantification failed: {e}", "error")
            return {"error": str(e)}

    async def _generate_research_paper(self) -> Dict[str, Any]:
        """Generate the final research paper"""
        self.log("Generating research paper...")

        # Use the research context to generate paper
        paper = {
            "title": self._generate_paper_title(),
            "abstract": self._generate_abstract(),
            "sections": self._generate_paper_sections(),
            "references": self._generate_references(),
            "confidence_scores": self._calculate_paper_confidence(),
            "generated_at": datetime.now().isoformat(),
            "session_id": self.session_id
        }

        # Convert to markdown
        paper["markdown"] = self._convert_to_markdown(paper)

        return paper

    def _select_best_domain(self, domains: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the best domain from discovery results"""
        domain_list = domains.get("domains", [])
        if not domain_list:
            return None

        # Select domain with highest confidence
        return max(domain_list, key=lambda x: x.get("confidence", 0))

    def _select_best_question(self, questions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the best question from generation results"""
        question_list = questions.get("questions", [])
        if not question_list:
            return None

        # Select question with highest overall score
        return max(question_list, key=lambda x: x.get("overall_score", 0))

    def _should_continue_research(self, iteration_result: Dict[str, Any]) -> bool:
        """Determine if research should continue to next iteration"""
        if not iteration_result.get("success", False):
            return True  # Try again if iteration failed

        # Check if critique requires iteration
        stages = iteration_result.get("stages", [])
        for stage in stages:
            if stage.get("stage") == "critique":
                critique_result = stage.get("result", {})
                if critique_result.get("requires_iteration", False):
                    self.log("Critique requires another iteration", "info")
                    return True

        # Check if we have enough iterations
        if self.current_iteration < 2:  # Minimum 3 iterations
            return True

        # Check overall confidence
        for stage in stages:
            if stage.get("stage") == "uncertainty_quantification":
                uncertainty_result = stage.get("result", {})
                overall_confidence = uncertainty_result.get("uncertainty_analysis", {}).get("overall_confidence", 0.5)

                if overall_confidence < settings.min_confidence_threshold:
                    self.log(f"Low overall confidence ({overall_confidence}), continuing", "info")
                    return True

        return False

    def _generate_paper_title(self) -> str:
        """Generate paper title from research context"""
        if not self.research_context.get("domains"):
            return "Autonomous Research Paper"

        domains = self.research_context.get("domains", {}).get("domains", [])
        if domains:
            best_domain = domains[0].get("name", "Emerging Field")
            return f"Autonomous Research in {best_domain}: AI-Generated Insights"

        return "AI-Generated Research Paper"

    def _generate_abstract(self) -> str:
        """Generate paper abstract"""
        abstract_parts = []

        # Add domain info
        if self.research_context.get("domains"):
            domains = self.research_context["domains"].get("domains", [])
            if domains:
                abstract_parts.append(f"This paper explores the emerging field of {domains[0].get('name', 'unknown')}.")

        # Add question info
        if self.research_context.get("questions"):
            questions = self.research_context["questions"].get("questions", [])
            if questions:
                abstract_parts.append(f"We investigate the question: '{questions[0].get('question', '')}'.")

        # Add method info
        if self.research_context.get("experiments"):
            experiments = self.research_context["experiments"]
            abstract_parts.append("Using autonomous AI agents, we design experiments and analyze data.")

        # Add conclusion
        abstract_parts.append(
            "This research demonstrates the potential of fully autonomous AI systems for scientific discovery.")

        return " ".join(abstract_parts)

    def _generate_paper_sections(self) -> Dict[str, str]:
        """Generate paper sections"""
        sections = {
            "introduction": self._generate_introduction(),
            "methods": self._generate_methods(),
            "results": self._generate_results(),
            "discussion": self._generate_discussion(),
            "limitations": self._generate_limitations()
        }

        return sections

    def _generate_introduction(self) -> str:
        """Generate introduction section"""
        intro = []

        intro.append("# Introduction\n")

        # Domain context
        if self.research_context.get("domains"):
            domains = self.research_context["domains"].get("domains", [])
            if domains:
                intro.append(
                    f"The field of {domains[0].get('name', 'this domain')} has emerged as a promising area of research.")
                intro.append(f"Recent developments suggest significant potential for innovation and discovery.")

        # Research question
        if self.research_context.get("questions"):
            questions = self.research_context["questions"].get("questions", [])
            if questions:
                intro.append(f"\nThis research addresses the question: {questions[0].get('question', '')}")
                intro.append(f"{questions[0].get('explanation', '')}")

        # Paper overview
        intro.append("\nThis paper presents findings from a fully autonomous AI research system.")
        intro.append(
            "All aspects of the research—from domain discovery to paper writing—were conducted by AI agents without human intervention.")

        return "\n".join(intro)

    def _generate_methods(self) -> str:
        """Generate methods section"""
        methods = ["# Methods\n"]

        methods.append("## Autonomous Research System\n")
        methods.append("This research was conducted using a multi-agent AI system with the following components:\n")

        methods.append("- **Domain Scout Agent**: Discovers emerging scientific fields")
        methods.append("- **Question Generator Agent**: Formulates novel research questions")
        methods.append("- **Data Alchemist Agent**: Acquires and cleans data from multiple sources")
        methods.append("- **Experiment Designer Agent**: Designs experiments and statistical methods")
        methods.append("- **Critic Agent**: Critiques methodology and identifies weaknesses")
        methods.append("- **Uncertainty Quantification Agent**: Assesses confidence in all outputs\n")

        # Specific methods from experiments
        if self.research_context.get("experiments"):
            experiments = self.research_context["experiments"]
            primary_experiment = experiments.get("primary_experiment", {})

            if primary_experiment:
                methods.append("## Experimental Design\n")

                exp_type = primary_experiment.get("type", "Observational")
                methods.append(f"**Experiment Type**: {exp_type}")

                methodology = primary_experiment.get("methodology", {})
                if isinstance(methodology, dict) and methodology.get("description"):
                    methods.append(f"**Methodology**: {methodology['description']}")

                # Statistical methods
                stat_methods = primary_experiment.get("statistical_methods", [])
                if stat_methods:
                    methods.append("\n**Statistical Methods**:")
                    for method in stat_methods[:3]:
                        if isinstance(method, dict):
                            methods.append(f"- {method.get('name', 'Unknown method')}")

        return "\n".join(methods)

    def _generate_results(self) -> str:
        """Generate results section"""
        results = ["# Results\n"]

        results.append("## Key Findings\n")

        # Domain discovery results
        if self.research_context.get("domains"):
            domains = self.research_context["domains"].get("domains", [])
            if domains:
                results.append(f"The system identified {len(domains)} emerging domains:")
                for domain in domains[:3]:
                    results.append(
                        f"- **{domain.get('name', 'Unknown')}** (confidence: {domain.get('confidence', 0) * 100:.1f}%)")

        # Data acquisition results
        if self.research_context.get("data"):
            data = self.research_context["data"]
            sources = data.get("sources", [])
            quality = data.get("quality_metrics", {}).get("overall_quality", 0.5)

            results.append(f"\n## Data Collection\n")
            results.append(
                f"Data was acquired from {len(sources)} sources with overall quality score: {quality * 100:.1f}%")

        # Experiment results (simulated since we don't run actual experiments)
        results.append("\n## Experimental Analysis\n")
        results.append("The autonomous system designed and critiqued experimental approaches.")
        results.append("Based on the critique and uncertainty quantification, the system achieved:")

        if self.research_context.get("uncertainty"):
            uncertainty = self.research_context["uncertainty"]
            overall_conf = uncertainty.get("uncertainty_analysis", {}).get("overall_confidence", 0.5)
            results.append(f"- Overall confidence: {overall_conf * 100:.1f}%")

        return "\n".join(results)

    def _generate_discussion(self) -> str:
        """Generate discussion section"""
        discussion = ["# Discussion\n"]

        discussion.append("## Implications of Autonomous Research\n")
        discussion.append("This research demonstrates that AI systems can autonomously:")
        discussion.append("1. Identify promising research directions")
        discussion.append("2. Formulate meaningful research questions")
        discussion.append("3. Acquire and process relevant data")
        discussion.append("4. Design and critique experimental approaches")
        discussion.append("5. Generate coherent research papers\n")

        discussion.append("## Significance of Findings\n")
        discussion.append("The ability of AI to conduct end-to-end research has several implications:")
        discussion.append("- **Accelerated Discovery**: AI can explore vast research spaces quickly")
        discussion.append("- **Novel Perspectives**: AI may identify connections humans might miss")
        discussion.append("- **Resource Efficiency**: Autonomous research reduces human effort requirements")
        discussion.append("- **Reproducibility**: AI systems can precisely document their methodology\n")

        # Critique insights
        if self.research_context.get("critique"):
            critique = self.research_context["critique"]
            major_issues = critique.get("major_issues", [])

            if major_issues:
                discussion.append("## Methodological Considerations\n")
                discussion.append("The critic agent identified several areas for improvement:")
                for issue in major_issues[:3]:
                    discussion.append(f"- {issue}")

        return "\n".join(discussion)

    def _generate_limitations(self) -> str:
        """Generate limitations section"""
        limitations = ["# Limitations and Future Work\n"]

        limitations.append("## Current Limitations\n")
        limitations.append("1. **Data Quality**: Reliance on publicly available data may limit depth")
        limitations.append("2. **Experimental Validation**: This system designs but does not execute experiments")
        limitations.append("3. **Domain Expertise**: AI lacks deep domain-specific knowledge of human experts")
        limitations.append(
            "4. **Ethical Review**: Autonomous systems require human oversight for ethical considerations")
        limitations.append("5. **Interpretation Depth**: AI analysis may miss nuanced interpretations\n")

        limitations.append("## Future Directions\n")
        limitations.append("1. **Integration with Lab Systems**: Connect to physical experimental setups")
        limitations.append("2. **Multi-modal Data**: Incorporate images, videos, and sensor data")
        limitations.append("3. **Collaborative AI-Human Research**: Hybrid systems combining AI and human expertise")
        limitations.append("4. **Real-time Adaptation**: Systems that learn from experimental results")
        limitations.append("5. **Ethical Framework Development**: Standards for autonomous research ethics")

        return "\n".join(limitations)

    def _generate_references(self) -> List[str]:
        """Generate references section"""
        references = ["# References\n"]

        # Add data sources as references
        if self.research_context.get("data"):
            data = self.research_context["data"]
            sources = data.get("sources", [])

            for i, source in enumerate(sources[:10], 1):
                url = source.get("url", "")
                desc = source.get("description", "")

                if url:
                    references.append(f"{i}. {desc} [URL: {url}]")
                elif desc:
                    references.append(f"{i}. {desc}")

        # Add AI system references
        references.append("\n## System References\n")
        references.append("- Groq API for LLM inference")
        references.append("- LangGraph for agent orchestration")
        references.append("- Streamlit for user interface")
        references.append("- FastAPI for backend services")

        return references

    def _calculate_paper_confidence(self) -> Dict[str, float]:
        """Calculate confidence scores for paper"""
        confidences = {}

        # Extract confidence from different stages
        if self.research_context.get("domains"):
            domains = self.research_context["domains"]
            if isinstance(domains, dict):
                confidences["domain_selection"] = domains.get("confidence", 0.5)

        if self.research_context.get("questions"):
            questions = self.research_context["questions"]
            if isinstance(questions, dict):
                q_list = questions.get("questions", [])
                if q_list:
                    confidences["question_quality"] = np.mean([q.get("overall_score", 0.5) for q in q_list])

        if self.research_context.get("data"):
            data = self.research_context["data"]
            quality = data.get("quality_metrics", {}).get("overall_quality", 0.5)
            confidences["data_quality"] = quality

        if self.research_context.get("experiments"):
            experiments = self.research_context["experiments"]
            primary_exp = experiments.get("primary_experiment", {})
            feasibility = experiments.get("feasibility_assessment", {}).get("overall_feasibility", 0.5)
            confidences["experiment_design"] = feasibility

        if self.research_context.get("uncertainty"):
            uncertainty = self.research_context["uncertainty"]
            overall_conf = uncertainty.get("uncertainty_analysis", {}).get("overall_confidence", 0.5)
            confidences["overall_confidence"] = overall_conf

        # Calculate paper-specific confidence
        if confidences:
            confidences["paper_coherence"] = 0.7  # Estimated
            confidences["methodological_soundness"] = np.mean(list(confidences.values())) if confidences else 0.5

        return confidences

    def _convert_to_markdown(self, paper: Dict[str, Any]) -> str:
        """Convert paper dict to markdown format"""
        markdown_parts = []

        # Title
        markdown_parts.append(f"# {paper.get('title', 'Research Paper')}\n")

        # Abstract
        markdown_parts.append("## Abstract")
        markdown_parts.append(paper.get('abstract', '') + "\n")

        # Sections
        sections = paper.get('sections', {})
        for section_name, section_content in sections.items():
            if section_content:
                markdown_parts.append(section_content + "\n")

        # References
        references = paper.get('references', [])
        markdown_parts.extend(references)

        # Confidence scores
        markdown_parts.append("\n## Confidence Metrics\n")
        confidence_scores = paper.get('confidence_scores', {})
        for metric, score in confidence_scores.items():
            markdown_parts.append(f"- **{metric.replace('_', ' ').title()}**: {score * 100:.1f}%")

        # Metadata
        markdown_parts.append(f"\n---\n")
        markdown_parts.append(f"*Generated autonomously by AI Research Assistant*  \n")
        markdown_parts.append(f"*Session ID: {paper.get('session_id', '')}*  \n")
        markdown_parts.append(f"*Generated: {paper.get('generated_at', '')}*")

        return "\n".join(markdown_parts)

    def _log_progress(self, message: str, progress: float):
        """Log progress and call progress callback"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": self.research_stage.value,
            "message": message,
            "progress": progress
        }

        self.research_log.append(log_entry)
        self.log(f"[{self.research_stage.value}] {message}")

        if self.progress_callback:
            try:
                self.progress_callback(message, progress / 100)
            except Exception as e:
                self.log(f"Progress callback failed: {e}", "warning")

    def cancel(self):
        """Cancel the research pipeline"""
        self.log("Research cancelled by user", "warning")
        # This would typically set a cancellation flag and stop async operations

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process orchestrator messages"""
        # The orchestrator primarily manages the pipeline, not individual messages
        # This method handles coordination messages between agents

        if message.message_type == "coordination":
            # Handle inter-agent coordination
            return await self._handle_coordination(message)

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            content={"status": "orchestrator_busy", "stage": self.research_stage.value},
            message_type="information"
        )

    async def _handle_coordination(self, message: AgentMessage) -> AgentMessage:
        """Handle coordination between agents"""
        # This would implement conflict resolution, resource allocation, etc.
        # For now, it's a placeholder

        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            content={"coordinated": True, "action": "continue"},
            message_type="command"
        )