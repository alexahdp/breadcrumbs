# LLM

## Terms

### Voice
 - **VAD** - Voice Activity Detection
 - **Barge-in** - The ability for a user to interrupt a system (like a voice assistant) while it’s speaking.
Problem solved: In natural conversations, people interrupt each other — barge-in allows more human-like, fluid interactions instead of waiting for long responses to finish. Use cases: Voice assistants (Alexa, Siri), customer support bots, or real-time voice UIs.
 - **Turn-taking** - The logic that manages when each party (human or AI) can talk in a conversation.
Problem solved: Prevents overlapping speech and ensures conversational flow.
Use cases: Any dialogue system — especially those using speech, such as call-center bots or multi-agent conversations.
 - **SIP** - Session Initiation Protocol. A protocol used to establish, modify, and terminate real-time sessions like voice/video calls over IP networks. Problem solved: It connects VoIP systems (like softphones, PBXs, or telephony gateways).
Use cases: Integrating AI voice agents with phone systems or call routing platforms.

### Cognitive Architectures & Reasoning Patterns

**Planner–Executor–Critic Pattern**  
 - Meaning: A design pattern for autonomous AI systems:
 - Planner decides what to do (strategy),
 - Executor performs the actions,
 - Critic evaluates results and provides feedback.
 - Problem solved: Makes reasoning systems more robust and self-correcting.
 - Use cases: AI agents that perform multi-step reasoning or real-world actions — e.g., robotic control, autonomous
 decision-making, or LLM agents.

**Role Routing**  
 - Meaning: A system that dynamically assigns subtasks to agents with specific “roles” or capabilities.
 - Problem solved: Enables coordination between specialized agents (planner, coder, verifier, etc.).
 - Use cases: Multi-agent LLM architectures, AI orchestration systems (like AutoGen or CrewAI).


**Tool Router**  
 - Meaning: A mechanism that decides which external tool or API to call based on the AI’s current intent or task.
 - Problem solved: Connects reasoning models (like LLMs) with the correct external functions (e.g., web search, database query, calculator).
 - Use cases: AI orchestration frameworks, LLM tool-use systems, dynamic function calling.


**Coordination Protocol and Evaluation**  
 - Meaning: Rules that govern how multiple agents (or reasoning modules) cooperate, share results, and evaluate progress.
 - Problem solved: Prevents chaos in distributed AI systems; ensures consistent decision-making.
 - Use cases: Multi-agent collaboration, autonomous system coordination, distributed cognition.

### Reasoning Paradigms (Emergent AI Architectures)

**ReAct (Reason + Act)**  
 - Meaning: A reasoning framework where an AI alternates between thinking (“reason”) and performing an action (“act”), based on feedback.
 - Problem solved: Keeps reasoning interpretable and grounded in real outcomes.
 - Use cases: LLM agents that query APIs, take notes, or search data dynamically.

**Chain of Thought (CoT)**  
 - Meaning: An internal reasoning trace where the AI breaks down complex problems step by step.
 - Problem solved: Improves reasoning accuracy by making decisions explicit.
 - Use cases: Math reasoning, code generation, complex decision-making.

**Tree of Thought (ToT)**  
 - Meaning: An extension of CoT — explores multiple reasoning branches and selects the best path.
 - Problem solved: Enables deliberate exploration of multiple hypotheses instead of linear reasoning.
 - Use cases: Planning, game playing, multi-solution search problems.

**Graph of Thought (GoT)**  
 - Meaning: Generalizes ToT — models reasoning as a graph where nodes represent partial thoughts and edges show relationships.
 - Problem solved: Allows parallel reasoning and reuse of sub-thoughts.
 - Use cases: Research agents, scientific reasoning, complex multi-agent tasks.

### Modern AI System Frameworks

**DSPy (Dynamic System for Python)**  
 - Meaning: A declarative framework for programming reasoning and decision-making in LLMs (developed by Stanford).
 - Problem solved: Replaces manual prompt engineering with structured “policies” that train or compose LLM reasoning.
 - Use cases: Building reasoning pipelines, optimizing LLM performance, or training multi-step agents.

**MiPRO (Minimal Prompt Optimization)**  
 - Meaning: Likely a framework or method for optimizing prompts efficiently (used with DSPy).
 - Problem solved: Reduces cost and complexity of tuning large prompts for multiple tasks.
 - Use cases: Prompt optimization, adaptive LLM configurations.

**GEPA (Generalized Execution and Planning Architecture)**  
 - Meaning: A higher-level architecture for integrating planning, execution, and reasoning (similar to the planner-executor-critic model).
 - Problem solved: Unifies different agent roles and reasoning processes.
 - Use cases: Building complex AI agents, task planning in robotics or reasoning systems.


### Frameworks

**LangChain**  
**LlamaIndex** - for connecting LLMs with external (private) data sources. Fits for retrieval-augmented generation (RAG) use cases.
**Haystack (deepset)** - an open-source framework for building search systems powered by LLMs and embeddings. Stable, production-ready.
** Microsoft Semantic Kernel** - an open-source framework for integrating LLMs with traditional programming languages and external data sources. Focuses on building AI applications that combine LLM capabilities with structured logic.
**LLM LogProbs** - a tool for analyzing hallucinations
**LangFuse** - Observability platform for LLM applications, providing monitoring, logging, and analytics to track performance and detect issues in real-time. 






## Links
 - 