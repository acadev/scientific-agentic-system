# Scientific Agentic System with Ollama Integration
# A comprehensive example for hackathon participants

"""
Core Agent Framework
====================
This module demonstrates the fundamental building blocks of an agentic system:
1. Agent architecture patterns
2. Tool integration strategies  
3. Memory and state management
4. Multi-agent coordination
5. Evaluation frameworks

Usage:
    python main.py --model llama3.1 --task research --query "protein folding mechanisms"
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import requests
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CORE AGENT ARCHITECTURE
# =============================================================================

class AgentRole(Enum):
    """Defines different agent roles in the system"""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"

@dataclass
class Message:
    """Standard message format for agent communication"""
    sender: str
    recipient: str
    content: str
    message_type: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    """Represents the current state of an agent"""
    agent_id: str
    role: AgentRole
    status: str = "idle"
    current_task: Optional[str] = None
    memory: Dict[str, Any] = field(default_factory=dict)
    tools_available: List[str] = field(default_factory=list)

class OllamaClient:
    """Client for interacting with Ollama local models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = self._get_available_models()
        
    def _get_available_models(self) -> List[str]:
        """Fetch list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json()['models']]
                logger.info(f"Available models: {models}")
                return models
            else:
                logger.warning("Could not fetch available models, using defaults")
                return ["llama3.1", "mistral", "codellama"]
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return []
    
    async def generate(self, model: str, prompt: str, system_prompt: str = None) -> str:
        """Generate response from Ollama model"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            if response.status_code == 200:
                return response.json()['response']
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Could not generate response - {str(e)}"

# =============================================================================
# 2. TOOL INTEGRATION FRAMEWORK
# =============================================================================

class Tool(ABC):
    """Abstract base class for agent tools"""
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class WebSearchTool(Tool):
    """Tool for web searching (mock implementation for demo)"""
    
    def name(self) -> str:
        return "web_search"
    
    def description(self) -> str:
        return "Search the web for scientific papers and information"
    
    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Mock web search - in production, integrate with actual search API"""
        await asyncio.sleep(1)  # Simulate API call
        
        # Mock results based on query
        mock_results = [
            {
                "title": f"Research paper on {query} - Study #{i+1}",
                "url": f"https://pubmed.example.com/paper{i+1}",
                "abstract": f"This study examines {query} and provides insights...",
                "authors": ["Dr. Smith", "Dr. Johnson"],
                "year": 2023 - i
            }
            for i in range(num_results)
        ]
        
        return {
            "success": True,
            "query": query,
            "results": mock_results,
            "count": len(mock_results)
        }

class CalculatorTool(Tool):
    """Tool for mathematical calculations"""
    
    def name(self) -> str:
        return "calculator"
    
    def description(self) -> str:
        return "Perform mathematical calculations and statistical analysis"
    
    async def execute(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate mathematical expressions"""
        try:
            # Basic safety - only allow certain operations
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Invalid characters in expression")
            
            result = eval(expression)
            return {
                "success": True,
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "expression": expression,
                "error": str(e)
            }

class FileAnalysisTool(Tool):
    """Tool for analyzing scientific data files"""
    
    def name(self) -> str:
        return "file_analysis"
    
    def description(self) -> str:
        return "Analyze scientific data files (CSV, JSON, etc.)"
    
    async def execute(self, file_path: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Analyze a file and return insights"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": "File not found"}
            
            # Mock analysis - in production, implement actual file analysis
            return {
                "success": True,
                "file_path": file_path,
                "analysis_type": analysis_type,
                "insights": {
                    "file_size": path.stat().st_size,
                    "file_type": path.suffix,
                    "summary": f"Analyzed {path.name} using {analysis_type} method"
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# =============================================================================
# 3. MEMORY AND STATE MANAGEMENT
# =============================================================================

class MemoryManager:
    """Manages agent memory using SQLite for persistence"""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the memory database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for different types of memory
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestamp REAL,
                event_type TEXT,
                content TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                concept TEXT,
                knowledge TEXT,
                confidence REAL,
                last_updated REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS working_memory (
                agent_id TEXT PRIMARY KEY,
                current_context TEXT,
                active_goals TEXT,
                recent_actions TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_episode(self, agent_id: str, event_type: str, content: str, metadata: Dict = None):
        """Store an episodic memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO episodic_memory (agent_id, timestamp, event_type, content, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (agent_id, time.time(), event_type, content, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
    
    def store_knowledge(self, agent_id: str, concept: str, knowledge: str, confidence: float = 1.0):
        """Store semantic knowledge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update or insert knowledge
        cursor.execute('''
            INSERT OR REPLACE INTO semantic_memory 
            (agent_id, concept, knowledge, confidence, last_updated)
            VALUES (?, ?, ?, ?, ?)
        ''', (agent_id, concept, knowledge, confidence, time.time()))
        
        conn.commit()
        conn.close()
    
    def get_relevant_memories(self, agent_id: str, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant memories (simplified similarity search)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple keyword-based retrieval (in production, use vector search)
        keywords = query.lower().split()
        
        memories = []
        for keyword in keywords:
            cursor.execute('''
                SELECT content, metadata, timestamp FROM episodic_memory 
                WHERE agent_id = ? AND (content LIKE ? OR metadata LIKE ?)
                ORDER BY timestamp DESC LIMIT ?
            ''', (agent_id, f'%{keyword}%', f'%{keyword}%', limit))
            
            memories.extend([
                {"content": row[0], "metadata": json.loads(row[1]), "timestamp": row[2]}
                for row in cursor.fetchall()
            ])
        
        conn.close()
        return memories[:limit]

# =============================================================================
# 4. BASE AGENT CLASS
# =============================================================================

class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, role: AgentRole, model: str = "llama3.1"):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.state = AgentState(agent_id=agent_id, role=role)
        self.ollama = OllamaClient()
        self.memory = MemoryManager()
        self.tools: Dict[str, Tool] = {}
        self.message_queue: List[Message] = []
        
        # Register default tools
        self._register_default_tools()
        
        # Set role-specific system prompt
        self.system_prompt = self._get_system_prompt()
    
    def _register_default_tools(self):
        """Register default tools available to all agents"""
        tools = [WebSearchTool(), CalculatorTool(), FileAnalysisTool()]
        for tool in tools:
            self.tools[tool.name()] = tool
            self.state.tools_available.append(tool.name())
    
    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt"""
        prompts = {
            AgentRole.RESEARCHER: """You are a scientific researcher agent. Your role is to:
- Search for and analyze scientific literature
- Identify key research questions and hypotheses
- Gather evidence from multiple sources
- Maintain objectivity and scientific rigor
Always cite your sources and acknowledge limitations in your analysis.""",
            
            AgentRole.ANALYST: """You are a data analyst agent. Your role is to:
- Analyze scientific data and identify patterns
- Perform statistical calculations and interpretations
- Create summaries of quantitative findings
- Identify potential correlations and causations
Always validate your analytical methods and report confidence levels.""",
            
            AgentRole.SYNTHESIZER: """You are a synthesis agent. Your role is to:
- Combine information from multiple sources
- Identify common themes and contradictions
- Create coherent summaries and conclusions
- Bridge different domains of knowledge
Always acknowledge conflicting evidence and areas of uncertainty.""",
            
            AgentRole.VALIDATOR: """You are a validation agent. Your role is to:
- Check the accuracy and reliability of information
- Identify potential biases and limitations
- Verify methodological soundness
- Assess the strength of evidence
Always provide constructive criticism and suggest improvements.""",
            
            AgentRole.COORDINATOR: """You are a coordinator agent. Your role is to:
- Manage workflow between different agents
- Prioritize tasks and allocate resources
- Ensure quality standards are met
- Facilitate communication and collaboration
Always consider the bigger picture and project goals."""
        }
        
        return prompts.get(self.role, "You are a helpful scientific assistant.")
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process an incoming message and generate a response"""
        self.message_queue.append(message)
        self.memory.store_episode(
            self.agent_id, 
            "message_received", 
            message.content,
            {"sender": message.sender, "type": message.message_type}
        )
        
        # Generate response using Ollama
        context = self._build_context(message)
        response_content = await self.ollama.generate(
            model=self.model,
            prompt=f"Message: {message.content}\n\nContext: {context}\n\nResponse:",
            system_prompt=self.system_prompt
        )
        
        # Store response in memory
        self.memory.store_episode(
            self.agent_id,
            "response_generated",
            response_content,
            {"original_message": message.content}
        )
        
        # Create response message
        response = Message(
            sender=self.agent_id,
            recipient=message.sender,
            content=response_content,
            message_type="response"
        )
        
        return response
    
    def _build_context(self, message: Message) -> str:
        """Build context for the LLM from memory and current state"""
        relevant_memories = self.memory.get_relevant_memories(
            self.agent_id, 
            message.content,
            limit=3
        )
        
        context_parts = [
            f"Agent Role: {self.role.value}",
            f"Available Tools: {', '.join(self.state.tools_available)}",
            f"Current Status: {self.state.status}"
        ]
        
        if relevant_memories:
            context_parts.append("Relevant Past Experiences:")
            for memory in relevant_memories:
                context_parts.append(f"- {memory['content']}")
        
        return "\n".join(context_parts)
    
    async def use_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Use a specific tool"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool {tool_name} not available"}
        
        self.state.status = f"using_{tool_name}"
        result = await self.tools[tool_name].execute(**kwargs)
        self.state.status = "idle"
        
        # Store tool usage in memory
        self.memory.store_episode(
            self.agent_id,
            "tool_used",
            f"Used {tool_name} with result: {result}",
            {"tool": tool_name, "params": kwargs}
        )
        
        return result

# =============================================================================
# 5. MULTI-AGENT COORDINATION
# =============================================================================

class AgentCoordinator:
    """Coordinates multiple agents working on a scientific task"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, List[str]] = {}
        self.results: Dict[str, Any] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} with role {agent.role.value}")
    
    async def execute_workflow(self, workflow_name: str, initial_query: str) -> Dict[str, Any]:
        """Execute a predefined workflow"""
        if workflow_name not in self.workflows:
            return {"success": False, "error": f"Workflow {workflow_name} not found"}
        
        workflow_agents = self.workflows[workflow_name]
        results = {"workflow": workflow_name, "query": initial_query, "steps": []}
        
        current_input = initial_query
        
        for agent_id in workflow_agents:
            if agent_id not in self.agents:
                logger.error(f"Agent {agent_id} not found")
                continue
            
            agent = self.agents[agent_id]
            logger.info(f"Executing step with agent {agent_id} ({agent.role.value})")
            
            # Create message for agent
            message = Message(
                sender="coordinator",
                recipient=agent_id,
                content=current_input,
                message_type="task"
            )
            
            # Process message and get response
            response = await agent.process_message(message)
            
            if response:
                step_result = {
                    "agent_id": agent_id,
                    "agent_role": agent.role.value,
                    "input": current_input,
                    "output": response.content,
                    "timestamp": response.timestamp
                }
                
                results["steps"].append(step_result)
                current_input = response.content  # Use output as input for next agent
            
            # Small delay between steps
            await asyncio.sleep(0.5)
        
        results["final_output"] = current_input
        results["success"] = True
        
        self.results[workflow_name] = results
        return results
    
    def define_workflow(self, name: str, agent_sequence: List[str]):
        """Define a new workflow"""
        self.workflows[name] = agent_sequence
        logger.info(f"Defined workflow '{name}' with agents: {agent_sequence}")

# =============================================================================
# 6. EVALUATION FRAMEWORK
# =============================================================================

class EvaluationMetrics:
    """Framework for evaluating agent performance"""
    
    @staticmethod
    def response_quality(response: str, criteria: List[str]) -> float:
        """Evaluate response quality based on criteria"""
        # Simple keyword-based evaluation (in production, use more sophisticated methods)
        score = 0.0
        for criterion in criteria:
            if criterion.lower() in response.lower():
                score += 1.0
        
        return min(score / len(criteria), 1.0) if criteria else 0.0
    
    @staticmethod
    def workflow_efficiency(workflow_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate workflow efficiency"""
        steps = workflow_results.get("steps", [])
        if not steps:
            return {"efficiency": 0.0, "completion_rate": 0.0}
        
        total_time = steps[-1]["timestamp"] - steps[0]["timestamp"]
        avg_step_time = total_time / len(steps)
        
        # Simple efficiency metrics
        efficiency = max(0.0, 1.0 - (avg_step_time / 10.0))  # Assume 10s is baseline
        completion_rate = 1.0  # If we got here, workflow completed
        
        return {
            "efficiency": efficiency,
            "completion_rate": completion_rate,
            "total_time": total_time,
            "steps_completed": len(steps)
        }

# =============================================================================
# 7. SCIENTIFIC WORKFLOW EXAMPLES
# =============================================================================

async def create_research_workflow() -> AgentCoordinator:
    """Create a scientific research workflow"""
    coordinator = AgentCoordinator()
    
    # Create specialized agents
    researcher = BaseAgent("researcher_01", AgentRole.RESEARCHER)
    analyst = BaseAgent("analyst_01", AgentRole.ANALYST)
    synthesizer = BaseAgent("synthesizer_01", AgentRole.SYNTHESIZER)
    validator = BaseAgent("validator_01", AgentRole.VALIDATOR)
    
    # Register agents
    for agent in [researcher, analyst, synthesizer, validator]:
        coordinator.register_agent(agent)
    
    # Define workflows
    coordinator.define_workflow(
        "literature_review",
        ["researcher_01", "analyst_01", "synthesizer_01", "validator_01"]
    )
    
    coordinator.define_workflow(
        "hypothesis_testing",
        ["researcher_01", "analyst_01", "validator_01"]
    )
    
    return coordinator

# =============================================================================
# 8. MAIN DEMONSTRATION SCRIPT
# =============================================================================

async def main():
    """Main demonstration of the agentic system"""
    print("🧬 Scientific Agentic System Demo")
    print("=" * 50)
    
    # Check Ollama connection
    ollama = OllamaClient()
    if not ollama.available_models:
        print("⚠️  Warning: Could not connect to Ollama. Make sure it's running.")
        print("   Install: https://ollama.ai")
        print("   Run: ollama serve")
        return
    
    print(f"✅ Connected to Ollama with models: {ollama.available_models}")
    
    # Create workflow coordinator
    print("\n🔧 Setting up multi-agent workflow...")
    coordinator = await create_research_workflow()
    
    # Demo query
    research_query = """
    I need to understand the current state of research on CRISPR gene editing 
    applications in treating genetic disorders. Please provide a comprehensive 
    analysis including recent developments, limitations, and future prospects.
    """
    
    print(f"\n🔍 Executing research workflow...")
    print(f"Query: {research_query}")
    
    # Execute workflow
    results = await coordinator.execute_workflow("literature_review", research_query)
    
    # Display results
    print(f"\n📊 Workflow Results:")
    print(f"Success: {results['success']}")
    print(f"Steps completed: {len(results.get('steps', []))}")
    
    for i, step in enumerate(results.get('steps', []), 1):
        print(f"\n--- Step {i}: {step['agent_role'].title()} ---")
        print(f"Input: {step['input'][:100]}...")
        print(f"Output: {step['output'][:200]}...")
    
    # Evaluate performance
    print(f"\n📈 Performance Evaluation:")
    efficiency = EvaluationMetrics.workflow_efficiency(results)
    for metric, value in efficiency.items():
        print(f"{metric}: {value}")
    
    # Demo individual agent interaction
    print(f"\n🤖 Individual Agent Demo:")
    researcher = coordinator.agents["researcher_01"]
    
    message = Message(
        sender="user",
        recipient="researcher_01",
        content="What are the main challenges in protein folding prediction?",
        message_type="question"
    )
    
    response = await researcher.process_message(message)
    print(f"Question: {message.content}")
    print(f"Response: {response.content}")
    
    # Demo tool usage
    print(f"\n🔧 Tool Usage Demo:")
    search_result = await researcher.use_tool(
        "web_search", 
        query="protein folding alphafold", 
        num_results=3
    )
    print(f"Search results: {search_result}")
    
    print(f"\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
