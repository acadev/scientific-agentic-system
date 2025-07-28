#!/usr/bin/env python3
"""
Diagnostic script to test the Scientific Agentic System
"""

import asyncio
import requests
from agentic_system_codebase import OllamaClient, BaseAgent, AgentRole

async def test_ollama_connection():
    """Test Ollama connection and model availability"""
    print("🔍 Testing Ollama Connection...")
    
    try:
        # Test basic connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running with {len(models)} models:")
            for model in models:
                print(f"   - {model['name']}")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running? Try: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

async def test_ollama_client():
    """Test OllamaClient class"""
    print("\n🤖 Testing OllamaClient...")
    
    client = OllamaClient()
    print(f"Available models: {client.available_models}")
    
    if not client.available_models:
        print("❌ No models available. Try: ollama pull llama3.1:8b")
        return False
    
    # Test generation with first available model
    test_model = client.available_models[0]
    print(f"Testing generation with model: {test_model}")
    
    response = await client.generate(
        model=test_model,
        prompt="Say hello in one sentence.",
        system_prompt="You are a helpful assistant."
    )
    
    if response.startswith("Error:"):
        print(f"❌ Generation failed: {response}")
        return False
    else:
        print(f"✅ Generation successful: {response[:100]}...")
        return True

async def test_agent_creation():
    """Test basic agent creation"""
    print("\n👤 Testing Agent Creation...")
    
    try:
        agent = BaseAgent("test_agent", AgentRole.RESEARCHER, "llama3.1:8b")
        print(f"✅ Agent created: {agent.agent_id} ({agent.role.value})")
        print(f"   Tools available: {len(agent.tools)}")
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

async def test_tools():
    """Test tool functionality"""
    print("\n🔧 Testing Tools...")
    
    try:
        from agentic_system_codebase import CalculatorTool, WebSearchTool, FileAnalysisTool
        
        # Test calculator
        calc = CalculatorTool()
        result = await calc.execute(expression="2+2")
        if result["success"] and result["result"] == 4:
            print("✅ Calculator tool working")
        else:
            print(f"❌ Calculator tool failed: {result}")
        
        # Test web search (mock)
        search = WebSearchTool()
        result = await search.execute(query="test", num_results=2)
        if result["success"] and len(result["results"]) == 2:
            print("✅ Web search tool working")
        else:
            print(f"❌ Web search tool failed: {result}")
        
        # Test file analysis
        file_tool = FileAnalysisTool()
        result = await file_tool.execute(file_path=__file__)
        if result["success"]:
            print("✅ File analysis tool working")
        else:
            print(f"❌ File analysis tool failed: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Tool testing failed: {e}")
        return False

async def main():
    """Run all diagnostic tests"""
    print("🧪 Scientific Agentic System Diagnostic")
    print("=" * 50)
    
    tests = [
        test_ollama_connection(),
        test_ollama_client(),
        test_agent_creation(),
        test_tools()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    passed = sum(1 for r in results if r is True)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        print("\nCommon fixes:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull a model: ollama pull llama3.1:8b")
        print("3. Check Python dependencies: pip install requests")

if __name__ == "__main__":
    asyncio.run(main())