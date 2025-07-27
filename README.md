# Scientific Agentic System with Ollama Integration

A comprehensive educational framework for building intelligent agent systems for scientific applications, designed for hackathon participants learning AI and collaborative development.

## 🎯 What You'll Learn

This codebase demonstrates all key concepts from **Session 2A: Agentic Systems Deep Dive**:

- **Agent Architecture Patterns** - Modular, role-based agent design
- **Tool Integration Strategies** - Extending agent capabilities with external tools
- **Memory and State Management** - Persistent context and learning systems
- **Multi-Agent Coordination** - Collaborative workflows and task decomposition
- **Evaluation Frameworks** - Metrics for assessing agent performance

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download/windows
```

### 2. Start Ollama Service

```bash
ollama serve
```

### 3. Download Models

```bash
# Recommended models for the hackathon
ollama pull llama3.1:8b     # Best balance of speed/capability
ollama pull mistral:7b      # Fast and efficient
ollama pull codellama:7b    # Good for code-related tasks
ollama pull phi3:mini       # Lightweight option
```

### 4. Set Up Python Environment

```bash
# Clone or download the hackathon code
git clone <repository-url>
cd scientific-agentic-system

# Install requirements (minimal dependencies)
pip install requests

# Verify environment
python environment_check.py

# Run automated setup
python setup_ollama.py
```

### 5. Run the Demo

```bash
python main.py
```

## 📚 Learning Path

### For Beginners (Track 1)

Start with the tutorial to learn concepts step by step:

```bash
# Interactive tutorial - all exercises
python tutorial.py

# Or run specific exercises
python tutorial.py --exercise 1  # Basic LLM interaction
python tutorial.py --exercise 2  # Agent creation
python tutorial.py --exercise 3  # Tool integration
```

**Learning Progression:**
1. **Exercise 1**: Basic LLM interaction with Ollama
2. **Exercise 2**: Creating agents with different roles
3. **Exercise 3**: Adding tools to extend capabilities
4. **Exercise 4**: Memory systems for context
5. **Exercise 5**: Multi-agent workflows
6. **Exercise 6**: Performance evaluation
7. **Exercise 7**: Complete scientific application

### For Advanced Users (Track 2)

Jump directly to specific concepts:

```bash
# Study the main system
python main.py

# Explore individual components
python -c "from main import BaseAgent, AgentRole; agent = BaseAgent('test', AgentRole.RESEARCHER); print(agent.system_prompt)"

# Run workflow examples
python -c "import asyncio; from main import create_research_workflow; asyncio.run(create_research_workflow())"
```

## 🏗️ System Architecture

### Core Components

```
scientific-agentic-system/
├── main.py              # Complete agentic system implementation
├── tutorial.py          # Step-by-step learning exercises
├── setup_ollama.py      # Automated setup and verification
├── environment_check.py # Python environment validation
└── README.md           # This documentation
```

### Agent Architecture

```python
# Basic agent structure
agent = BaseAgent(
    agent_id="researcher_01",
    role=AgentRole.RESEARCHER,
    model="llama3.1:8b"
)

# Agents have:
# - Role-specific system prompts
# - Built-in tool integration
# - Persistent memory systems
# - Message-based communication
```

### Available Agent Roles

- **RESEARCHER** - Literature search and analysis
- **ANALYST** - Data processing and statistical analysis  
- **SYNTHESIZER** - Information integration and summarization
- **VALIDATOR** - Quality control and fact-checking
- **COORDINATOR** - Workflow management and task allocation

### Built-in Tools

- **WebSearchTool** - Scientific literature and information retrieval
- **CalculatorTool** - Mathematical and statistical calculations
- **FileAnalysisTool** - Scientific data file processing

### Memory System

- **Episodic Memory** - Specific interactions and events
- **Semantic Memory** - Learned concepts and knowledge
- **Working Memory** - Current context and active goals

## 🔬 Scientific Applications

### Example Workflows

1. **Literature Review**
   ```
   Researcher → Analyst → Synthesizer → Validator
   ```

2. **Hypothesis Testing**
   ```
   Researcher → Analyst → Validator
   ```

3. **Data Analysis Pipeline**
   ```
   Analyst → Synthesizer → Validator
   ```

### Sample Research Queries

Try these with the system:

- "What are the current limitations of CRISPR gene editing technology?"
- "Analyze the relationship between climate change and coral reef biodiversity"
- "Compare machine learning approaches for protein structure prediction"
- "Investigate microplastic pollution impacts on marine food chains"

## 🛠️ Extending the System

### Adding New Tools

```python
class CustomTool(Tool):
    def name(self) -> str:
        return "custom_tool"
    
    def description(self) -> str:
        return "Description of what this tool does"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        # Your tool implementation
        return {"success": True, "result": "..."}

# Register with agent
agent.tools["custom_tool"] = CustomTool()
```

### Creating Custom Agent Roles

```python
# Add to AgentRole enum
class AgentRole(Enum):
    # ... existing roles ...
    DOMAIN_EXPERT = "domain_expert"

# Add corresponding system prompt in BaseAgent._get_system_prompt()
```

### Defining New Workflows

```python
coordinator = AgentCoordinator()

# Register your agents
coordinator.register_agent(agent1)
coordinator.register_agent(agent2)

# Define workflow
coordinator.define_workflow(
    "custom_workflow",
    ["agent1_id", "agent2_id"]
)

# Execute
results = await coordinator.execute_workflow("custom_workflow", "Your query")
```

## 🏆 Hackathon Project Ideas

### Beginner Projects

1. **Research Assistant** - Build a literature review bot for your scientific field
2. **Data Analyzer** - Create an agent that processes and explains experimental data
3. **Study Planner** - Design a system that helps plan scientific experiments
4. **Citation Finder** - Build a tool that finds and formats scientific references

### Advanced Projects

1. **Multi-Modal Research System** - Integrate image, text, and data analysis
2. **Collaborative Lab Assistant** - Coordinate multiple scientific workflows
3. **Hypothesis Generator** - AI system that proposes testable hypotheses
4. **Research Quality Evaluator** - Assess and improve scientific writing
5. **Cross-Domain Knowledge Bridge** - Connect insights across scientific fields

### Domain-Specific Applications

- **Biology**: Gene analysis, protein folding, ecological modeling
- **Chemistry**: Molecular design, reaction optimization, materials science
- **Physics**: Data analysis, theoretical modeling, experimental design
- **Environmental Science**: Climate modeling, pollution analysis, sustainability
- **Medicine**: Drug discovery, diagnostic assistance, treatment optimization

## 🤝 Collaboration Features

### GitHub Integration

The system is designed for collaborative development:

```bash
# Fork the repository
git fork <original-repo>

# Create feature branch
git checkout -b feature/my-scientific-tool

# Make changes and commit
git add .
git commit -m "Add custom scientific analysis tool"

# Push and create pull request
git push origin feature/my-scientific-tool
```

### Team Development

- **Modular Design** - Easy to divide work among team members
- **Clear Interfaces** - Well-defined APIs for tools and agents
- **Documentation** - Comprehensive examples and tutorials
- **Testing Framework** - Built-in evaluation and validation

## 📊 Performance and Evaluation

### Built-in Metrics

- **Response Quality** - Content relevance and completeness
- **Workflow Efficiency** - Speed and resource usage
- **Scientific Rigor** - Accuracy and methodology

### Custom Evaluation

```python
# Create custom metrics for your domain
def domain_specific_metric(text: str) -> float:
    # Your evaluation logic
    return score

# Apply to agent responses
score = domain_specific_metric(agent_response)
```

## 🔧 Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

**No Models Available**
```bash
# Pull a basic model
ollama pull llama3.1:8b

# List available models
ollama list
```

**Python Environment Issues**
```bash
# Run environment check
python environment_check.py

# Install missing dependencies
pip install requests
```

**Memory Database Issues**
```bash
# Delete and recreate database
rm agent_memory.db
python -c "from main import MemoryManager; MemoryManager()"
```

### Performance Optimization

- **Model Selection** - Choose appropriate model size for your hardware
- **Batch Processing** - Process multiple queries together
- **Memory Management** - Regularly clean old memories
- **Tool Caching** - Cache expensive tool operations

## 🌟 Post-Hackathon Development

### Sustaining Your Project

1. **Documentation** - Maintain clear README and API docs
2. **Testing** - Add unit tests for reliability
3. **Community** - Engage with other developers
4. **Publishing** - Share your work on GitHub

### Scaling Up

- **Production Deployment** - Move from local to cloud infrastructure
- **API Integration** - Connect to real scientific databases
- **User Interface** - Build web or desktop interfaces
- **Model Training** - Fine-tune models for your domain

### Funding and Support

- **Academic Grants** - Apply for research funding
- **Industry Partnerships** - Collaborate with scientific organizations
- **Open Source** - Contribute to the scientific software community

## 📖 Additional Resources

### Learning Materials

- [Ollama Documentation](https://ollama.ai/docs)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [Multi-Agent Systems Research](https://arxiv.org/search/?query=multi-agent+systems)

### Scientific APIs and Datasets

- [PubMed API](https://www.ncbi.nlm.nih.gov/pmc/tools/developers/) - Medical literature
- [arXiv API](https://arxiv.org/help/api) - Scientific preprints
- [NASA APIs](https://api.nasa.gov/) - Space and Earth science data
- [NCBI Datasets](https://www.ncbi.nlm.nih.gov/datasets/) - Biological data

### Community and Support

- **GitHub Discussions** - Ask questions and share ideas
- **Scientific Python Community** - Connect with domain experts
- **AI/ML Communities** - Learn about latest developments

## 🤖 Technical Details

### Model Requirements

- **Minimum**: 8GB RAM for 7B parameter models
- **Recommended**: 16GB RAM for 13B parameter models
- **Optimal**: 32GB+ RAM for multiple concurrent models

### API Compatibility

The system is designed to be model-agnostic:

```python
# Easy to swap between local and cloud models
# Just change the OllamaClient to OpenAIClient, etc.
```

### Security Considerations

- **Local Execution** - All models run locally for data privacy
- **Safe Evaluation** - Mathematical expressions are sanitized
- **Memory Isolation** - Each agent maintains separate memory space

## 📞 Support and Contributing

### Getting Help

1. Check this README for common solutions
2. Run the diagnostic scripts (`environment_check.py`, `setup_ollama.py`)
3. Review tutorial exercises for learning concepts
4. Ask questions in GitHub Issues

### Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### License

This project is designed for educational use in scientific research. Please respect any licensing terms of dependencies and cite appropriately in academic work.

---

## 🎉 Ready to Start Building?

1. **Complete the setup** - Follow the Quick Start guide
2. **Run the tutorial** - Learn concepts step by step
3. **Try the examples** - Experiment with scientific queries
4. **Build your project** - Apply concepts to your research domain
5. **Collaborate and share** - Work with others and contribute back

**Happy hacking! 🚀**