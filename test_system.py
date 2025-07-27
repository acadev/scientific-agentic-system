#!/usr/bin/env python3
"""
Test Suite for Scientific Agentic System
=========================================

This file contains tests to verify that the system works correctly.

Usage:
    python test_system.py
"""

import asyncio
import unittest
from unittest.mock import Mock, patch
from agentic_system_codebase import (
    BaseAgent, AgentRole, Message, OllamaClient, 
    CalculatorTool, WebSearchTool, FileAnalysisTool,
    MemoryManager, AgentCoordinator, EvaluationMetrics
)


class TestAgenticSystem(unittest.TestCase):
    """Test cases for the Scientific Agentic System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_agent = BaseAgent("test_agent", AgentRole.RESEARCHER, "llama3:latest")
        self.coordinator = AgentCoordinator()
    
    def test_agent_creation(self):
        """Test that agents are created correctly"""
        self.assertEqual(self.test_agent.agent_id, "test_agent")
        self.assertEqual(self.test_agent.role, AgentRole.RESEARCHER)
        self.assertIsNotNone(self.test_agent.system_prompt)
        self.assertEqual(len(self.test_agent.tools), 3)  # Default tools
    
    def test_memory_manager(self):
        """Test the memory management system"""
        memory = MemoryManager(":memory:")  # Use in-memory database for testing
        
        # Test storing episode
        memory.store_episode("test_agent", "test_event", "test content")
        
        # Test storing knowledge
        memory.store_knowledge("test_agent", "test_concept", "test knowledge")
        
        # Test retrieving memories
        memories = memory.get_relevant_memories("test_agent", "test")
        self.assertIsInstance(memories, list)
    
    def test_calculator_tool(self):
        """Test the calculator tool"""
        async def run_test():
            calc = CalculatorTool()
            result = await calc.execute(expression="2+2")
            self.assertTrue(result["success"])
            self.assertEqual(result["result"], 4)
            
            # Test invalid expression
            result = await calc.execute(expression="import os")
            self.assertFalse(result["success"])
        
        asyncio.run(run_test())
    
    def test_web_search_tool(self):
        """Test the web search tool (mock)"""
        async def run_test():
            search = WebSearchTool()
            result = await search.execute(query="protein folding", num_results=3)
            self.assertTrue(result["success"])
            self.assertEqual(len(result["results"]), 3)
            self.assertEqual(result["query"], "protein folding")
        
        asyncio.run(run_test())
    
    def test_file_analysis_tool(self):
        """Test the file analysis tool"""
        async def run_test():
            file_tool = FileAnalysisTool()
            
            # Test with non-existent file
            result = await file_tool.execute(file_path="/nonexistent/file.txt")
            self.assertFalse(result["success"])
            
            # Test with existing file (this test file)
            result = await file_tool.execute(file_path=__file__)
            self.assertTrue(result["success"])
        
        asyncio.run(run_test())
    
    def test_agent_coordinator(self):
        """Test the agent coordination system"""
        # Register test agent
        self.coordinator.register_agent(self.test_agent)
        self.assertIn("test_agent", self.coordinator.agents)
        
        # Define a workflow
        self.coordinator.define_workflow("test_workflow", ["test_agent"])
        self.assertIn("test_workflow", self.coordinator.workflows)
    
    def test_evaluation_metrics(self):
        """Test the evaluation framework"""
        # Test response quality metric
        response = "This response contains scientific evidence and research findings"
        criteria = ["scientific", "research", "evidence"]
        quality = EvaluationMetrics.response_quality(response, criteria)
        self.assertGreater(quality, 0.8)  # Should find all criteria
        
        # Test workflow efficiency
        mock_workflow = {
            "steps": [
                {"timestamp": 1000.0},
                {"timestamp": 1005.0}
            ]
        }
        efficiency = EvaluationMetrics.workflow_efficiency(mock_workflow)
        self.assertIn("efficiency", efficiency)
        self.assertIn("completion_rate", efficiency)
    
    @patch('requests.get')
    def test_ollama_client_connection(self, mock_get):
        """Test Ollama client connection (mocked)"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "test_model"}]
        }
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        self.assertIn("test_model", client.available_models)
    
    def test_message_creation(self):
        """Test message creation and structure"""
        message = Message(
            sender="test_sender",
            recipient="test_recipient", 
            content="test content",
            message_type="test"
        )
        
        self.assertEqual(message.sender, "test_sender")
        self.assertEqual(message.recipient, "test_recipient")
        self.assertEqual(message.content, "test content")
        self.assertEqual(message.message_type, "test")
        self.assertIsNotNone(message.timestamp)


def run_integration_test():
    """Run a more comprehensive integration test"""
    async def integration_test():
        print("🧪 Running Integration Test...")
        
        try:
            # Test full workflow creation and basic functionality
            from agentic_system_codebase import create_research_workflow
            
            coordinator = await create_research_workflow()
            print("✅ Research workflow created successfully")
            
            # Test that all agents are registered
            expected_agents = ["researcher_01", "analyst_01", "synthesizer_01", "validator_01"]
            for agent_id in expected_agents:
                assert agent_id in coordinator.agents, f"Agent {agent_id} not found"
            print("✅ All agents registered correctly")
            
            # Test that workflows are defined
            expected_workflows = ["literature_review", "hypothesis_testing"]
            for workflow in expected_workflows:
                assert workflow in coordinator.workflows, f"Workflow {workflow} not found"
            print("✅ All workflows defined correctly")
            
            print("🎉 Integration test passed!")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            raise
    
    asyncio.run(integration_test())


if __name__ == "__main__":
    print("🔬 Running Scientific Agentic System Tests")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    
    # Run integration test
    run_integration_test()
    
    print("\n✅ All tests completed!")
