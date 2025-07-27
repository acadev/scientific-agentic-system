#!/usr/bin/env python3
"""
Scientific Agentic System - Main Entry Point
============================================

This is the main entry point for the Scientific Agentic System.
Run this file to start the interactive demo.

Usage:
    python main.py

Make sure Ollama is running before executing:
    ollama serve
"""

from agentic_system_codebase import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
