#!/usr/bin/env python3
"""
Integration test for the complete workflow
"""

import asyncio
from agentic_system_codebase import create_research_workflow

async def test_full_workflow():
    """Test the complete research workflow"""
    print("🔬 Testing Full Research Workflow...")
    
    try:
        # Create coordinator
        coordinator = await create_research_workflow()
        print(f"✅ Coordinator created with {len(coordinator.agents)} agents")
        
        # Test simple query
        test_query = "What is machine learning?"
        print(f"Testing query: {test_query}")
        
        # Execute workflow (with shorter query to avoid long processing)
        results = await coordinator.execute_workflow("literature_review", test_query)
        
        if results["success"]:
            print(f"✅ Workflow completed with {len(results.get('steps', []))} steps")
            return True
        else:
            print(f"❌ Workflow failed: {results}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_full_workflow())