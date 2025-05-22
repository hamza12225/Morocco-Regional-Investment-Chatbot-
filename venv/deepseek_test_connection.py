# deepseek_test.py

import os
import asyncio
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class DeepSeekTester:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"
        
        # Check if API key exists
        if not self.api_key:
            print("❌ ERROR: DEEPSEEK_API_KEY not found in environment variables")
            print("Please create a .env file with your DeepSeek API key")
            return None
        
        print(f"✅ API Key found: {self.api_key[:10]}...")
        
        # Initialize both sync and async clients
        self.sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def test_sync_connection(self):
        """Test synchronous connection to DeepSeek API"""
        print("\n🔄 Testing Synchronous Connection...")
        
        try:
            response = self.sync_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello! Can you confirm that the DeepSeek API is working? Please respond with a brief confirmation."}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            print("✅ Sync Connection Successful!")
            print(f"📝 Response: {response.choices[0].message.content}")
            print(f"🔢 Tokens Used: {response.usage.total_tokens}")
            print(f"💰 Prompt Tokens: {response.usage.prompt_tokens}")
            print(f"💰 Completion Tokens: {response.usage.completion_tokens}")
            
            return True
            
        except Exception as e:
            print(f"❌ Sync Connection Failed: {str(e)}")
            return False
    
    async def test_async_connection(self):
        """Test asynchronous connection to DeepSeek API"""
        print("\n🔄 Testing Asynchronous Connection...")
        
        try:
            response = await self.async_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a Morocco investment expert."},
                    {"role": "user", "content": "What are the top 3 regions in Morocco for foreign investment? Give a brief answer."}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            print("✅ Async Connection Successful!")
            print(f"📝 Response: {response.choices[0].message.content}")
            print(f"🔢 Tokens Used: {response.usage.total_tokens}")
            
            return True
            
        except Exception as e:
            print(f"❌ Async Connection Failed: {str(e)}")
            return False
    
    def test_model_variants(self):
        """Test different DeepSeek model variants"""
        print("\n🔄 Testing Different Model Variants...")
        
        models_to_test = [
            "deepseek-chat",
            "deepseek-coder",
        ]
        
        for model in models_to_test:
            try:
                print(f"\n🧪 Testing model: {model}")
                
                response = self.sync_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": "Respond with just 'Working' if you can understand this message."}
                    ],
                    max_tokens=50,
                    temperature=0.3
                )
                
                print(f"✅ {model}: {response.choices[0].message.content.strip()}")
                
            except Exception as e:
                print(f"❌ {model}: Failed - {str(e)}")
    
    def test_morocco_investment_query(self):
        """Test a Morocco-specific investment query"""
        print("\n🔄 Testing Morocco Investment Query...")
        
        try:
            query = """
            I'm interested in investing 25 million MAD in Morocco. 
            I'm considering the automotive sector. 
            Can you recommend which region would be best and why?
            """
            
            response = self.sync_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert on Morocco investment opportunities. Provide specific, actionable advice about regional advantages for different sectors."
                    },
                    {"role": "user", "content": query}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            print("✅ Morocco Investment Query Successful!")
            print("📝 Response:")
            print("-" * 50)
            print(response.choices[0].message.content)
            print("-" * 50)
            print(f"🔢 Total Tokens: {response.usage.total_tokens}")
            
            return True
            
        except Exception as e:
            print(f"❌ Morocco Investment Query Failed: {str(e)}")
            return False
    
    def test_rate_limits(self):
        """Test API rate limits with multiple quick requests"""
        print("\n🔄 Testing Rate Limits (5 quick requests)...")
        
        success_count = 0
        
        for i in range(5):
            try:
                response = self.sync_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": f"Test message #{i+1}. Respond with just the number {i+1}."}
                    ],
                    max_tokens=10,
                    temperature=0.3
                )
                
                print(f"✅ Request {i+1}: {response.choices[0].message.content.strip()}")
                success_count += 1
                
            except Exception as e:
                print(f"❌ Request {i+1}: Failed - {str(e)}")
        
        print(f"📊 Rate Limit Test: {success_count}/5 requests successful")
        return success_count
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting DeepSeek API Tests...")
        print("=" * 60)
        
        # Test 1: Basic sync connection
        sync_success = self.test_sync_connection()
        
        # Test 2: Async connection  
        async_success = await self.test_async_connection()
        
        # Test 3: Different models
        self.test_model_variants()
        
        # Test 4: Morocco-specific query
        morocco_success = self.test_morocco_investment_query()
        
        # Test 5: Rate limits
        rate_limit_success = self.test_rate_limits()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Sync Connection: {'PASSED' if sync_success else 'FAILED'}")
        print(f"✅ Async Connection: {'PASSED' if async_success else 'FAILED'}")
        print(f"✅ Morocco Query: {'PASSED' if morocco_success else 'FAILED'}")
        print(f"✅ Rate Limit Test: {rate_limit_success}/5 requests successful")
        
        overall_success = sync_success and async_success and morocco_success and rate_limit_success >= 3
        
        if overall_success:
            print("\n🎉 ALL TESTS PASSED! DeepSeek API is ready for your Morocco Investment Chatbot!")
        else:
            print("\n⚠️ Some tests failed. Please check your API key and connection.")
        
        return overall_success

# Simple synchronous test function
def quick_test():
    """Quick test function for immediate feedback"""
    print("🚀 Quick DeepSeek API Test")
    print("-" * 30)
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("❌ No API key found!")
        print("Please set DEEPSEEK_API_KEY in your environment or .env file")
        return False
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Hello! Just say 'API Working' to confirm connection."}
            ],
            max_tokens=10
        )
        
        print(f"✅ SUCCESS: {response.choices[0].message.content}")
        print(f"🔢 Tokens used: {response.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

# Main execution
if __name__ == "__main__":
    # First, create the .env file template if it doesn't exist
    if not os.path.exists('.env'):
        print("📝 Creating .env template file...")
        with open('.env', 'w') as f:
            f.write("# DeepSeek API Configuration\n")
            f.write("DEEPSEEK_API_KEY=your_deepseek_api_key_here\n")
            f.write("DEEPSEEK_BASE_URL=https://api.deepseek.com\n")
            f.write("ENVIRONMENT=development\n")
        print("✅ .env file created! Please add your DeepSeek API key.")
    
    # Quick test first
    print("Starting quick test...\n")
    quick_success = quick_test()
    
    if quick_success:
        print("\n" + "="*50)
        print("Quick test passed! Running comprehensive tests...")
        print("="*50)
        
        # Run comprehensive tests
        tester = DeepSeekTester()
        if tester.api_key:  # Only run if API key is available
            asyncio.run(tester.run_all_tests())
    else:
        print("\n❌ Quick test failed. Please check your API key before running full tests.")
        print("\n📋 Setup Instructions:")
        print("1. Get your API key from https://platform.deepseek.com")
        print("2. Add it to the .env file: DEEPSEEK_API_KEY=your_key_here")
        print("3. Run this test again")

# Requirements for this test
print("\n📦 Required packages (install with pip):")
print("pip install openai python-dotenv asyncio")