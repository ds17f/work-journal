#!/usr/bin/env python3
"""Quick test script to verify local LLM integration works."""

import sys
sys.path.insert(0, 'src')

from work_journal.llm import LLMClient

def test_local_llms():
    print("🤖 Testing local LLM integration...")
    
    try:
        # Create LLM client with default configuration
        client = LLMClient()
        
        print(f"Current configuration:")
        print(f"  Provider: {client.config.workflows['processing'].provider}")  
        print(f"  Model: {client.config.workflows['processing'].model}")
        print(f"  Current preset: {client.config.current_preset}")
        
        # Test current provider
        current_provider = client.config.workflows['processing'].provider
        current_model = client.config.workflows['processing'].model
        
        print(f"\n🔌 Testing {current_provider} connection...")
        success = client.test_provider(current_provider, current_model)
        
        if success:
            print(f"✅ {current_provider.title()} connection successful!")
            
            # Try a simple LLM call
            print("\n🧠 Testing LLM call...")
            response = client.call_llm(
                "processing",
                [{"role": "user", "content": "Say 'Hello from local AI!' and nothing else."}],
                max_tokens=20
            )
            print(f"Response: {response}")
            
        else:
            print(f"❌ {current_provider.title()} connection failed!")
            
            if current_provider == "lmstudio":
                print("Make sure:")
                print("  1. LM Studio is running")
                print("  2. Go to 'Local Server' tab and click 'Start Server'")
                print("  3. Model is loaded and server shows port 1234")
            elif current_provider == "ollama":
                print("Make sure:")
                print("  1. Ollama is running: ollama serve")
                print("  2. Model is installed: ollama pull llama3.2:3b")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\nTroubleshooting for {current_provider}:")
        
        if current_provider == "lmstudio":
            print("  1. Install LM Studio from lmstudio.ai")
            print("  2. Download a model (search 'llama-3.2-3b-instruct')")
            print("  3. Start the local server in LM Studio")
        else:
            print("  1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
            print("  2. Start Ollama: ollama serve") 
            print("  3. Install model: ollama pull llama3.2:3b")

if __name__ == "__main__":
    test_local_llms()