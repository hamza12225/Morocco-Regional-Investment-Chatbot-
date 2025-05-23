import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dotenv import load_dotenv
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Imports
# Updated LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

class DeepSeekConfig:
    """Enhanced configuration for DeepSeek API with LangChain compatibility"""
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = "deepseek-chat"
        self.max_tokens = 2000
        self.temperature = 0.7
        self.timeout = 30
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")

class TokenUsageCallback(AsyncCallbackHandler):
    """Custom callback to track token usage across LangChain operations"""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    async def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            self.total_tokens += usage.get('total_tokens', 0)
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)

class MoroccoKnowledgeBase:
    """Enhanced knowledge base with vector search capabilities"""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",  # Lightest
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize vector store with Morocco investment data"""
        
        # Comprehensive Morocco investment knowledge
        knowledge_documents = [
            # Regional Information
            Document(page_content="""
            Casablanca-Settat Region Economic Profile:
            - GDP Share: 32% of Morocco's total GDP
            - Population: 6.8 million inhabitants
            - Key Sectors: Automotive (Renault, PSA), Aerospace (Boeing, Safran), Finance (BMCE, Attijariwafa), Logistics, Textiles
            - Infrastructure: Port of Casablanca (40M tons capacity), Mohammed V International Airport, Industrial zones in Nouaceur, Berrechid, Mohammedia
            - Workforce: 2.5M active population, multilingual (Arabic, French, English), strong technical education
            - Investment Advantages: Established financial hub, excellent infrastructure, skilled workforce, proximity to European markets
            """, metadata={"type": "regional", "region": "casablanca_settat"}),
            
            Document(page_content="""
            Tangier-Tetouan-Al Hoceima Region Economic Profile:
            - GDP Share: 8% of Morocco's total GDP
            - Population: 3.5 million inhabitants
            - Key Sectors: Manufacturing, Automotive (Renault factory), Textiles, Logistics
            - Infrastructure: Tanger Med Port (largest in Africa, 9M TEU capacity), Ibn Battouta Airport, Tanger Free Zone, Tanger Automotive City
            - Workforce: 800K active population, proximity to Europe advantage, competitive labor costs
            - Investment Advantages: Gateway to Europe (14km from Spain), modern port facilities, established automotive cluster, free zone benefits
            """, metadata={"type": "regional", "region": "tangier_tetouan"}),
            
            # Regulatory Information
            Document(page_content="""
            Morocco Investment Process and Timeline:
            Step 1: Business plan validation (1-2 weeks)
            Step 2: Legal structure selection - SARL (min 10,000 MAD), SA (min 300,000 MAD), SAS (min 10,000 MAD) (1 week)
            Step 3: Company name reservation at OMPIC (2-3 days)
            Step 4: Capital deposit in Moroccan bank (1 day)
            Step 5: Company registration at Commercial Court (5-10 days)
            Step 6: Tax registration (Patente) (3-5 days)
            Step 7: Social security registration (CNSS) (3-5 days)
            Step 8: Sector-specific permits and licenses (15-45 days)
            Total timeframe: 6-12 weeks depending on sector complexity
            """, metadata={"type": "regulatory", "category": "process"}),
            
            Document(page_content="""
            Required Documents for Morocco Investment:
            - Valid passport and residence permit
            - Comprehensive business plan and feasibility study
            - Bank certificate of capital deposit
            - Company statutes (Arabic translation required by certified translator)
            - Lease agreement for business premises
            - Sector-specific technical approvals and permits
            - Environmental impact assessment (for industrial projects)
            - Professional qualifications certificates (for regulated sectors)
            """, metadata={"type": "regulatory", "category": "documents"}),
            
            # Financial Information
            Document(page_content="""
            Morocco Investment Incentives 2024:
            Manufacturing Sector:
            - Corporate tax: 0% for first 5 years, then 17.5%
            - Minimum investment: 10 million MAD
            - Employment requirement: 25+ permanent jobs
            - VAT exemption on imported equipment
            
            Export Activities:
            - Corporate tax: 0% for first 10 years, then 17.5%
            - Minimum export requirement: 70% of production
            - Full VAT exemption on imported raw materials
            
            Regional Incentives:
            - Southern Provinces: Additional 5-year tax exemption
            - Rural areas: 50% reduction in land acquisition costs
            - Industrial zones: Ready infrastructure and utilities
            """, metadata={"type": "financial", "category": "incentives"}),
            
            Document(page_content="""
            Morocco Investment Costs and Financing:
            Land and Construction:
            - Industrial land: 200-500 MAD/m² (varies by region)
            - Construction costs: 3,000-5,000 MAD/m² for industrial buildings
            
            Labor Costs (2024):
            - Minimum wage: 3,000 MAD/month
            - Skilled technician: 5,000-8,000 MAD/month
            - Engineer: 8,000-15,000 MAD/month
            - Manager: 15,000-30,000 MAD/month
            
            Utilities:
            - Electricity: 1.2-1.8 MAD/kWh (industrial rates)
            - Water: 8-12 MAD/m³ (industrial rates)
            - High-speed internet: 500-2,000 MAD/month
            
            Financing Options:
            - Local banks: Investment loans up to 80% of project cost
            - Hassan II Fund: Strategic project financing
            - IFC World Bank: Equity and debt financing
            """, metadata={"type": "financial", "category": "costs"})
        ]
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=knowledge_documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create BM25 retriever for keyword-based search
        self.bm25_retriever = BM25Retriever.from_documents(knowledge_documents)
        
        # Create ensemble retriever combining vector and keyword search
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]  # Favor vector search slightly
        )
    
    def retrieve_relevant_info(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant information using ensemble retrieval"""
        return self.ensemble_retriever.get_relevant_documents(query)

# LangChain Tools for Morocco Investment
@tool
def get_regional_info(region: str, aspect: str = "overview") -> str:
    """Get detailed information about Morocco regions for investment analysis.
    
    Args:
        region: Region name (casablanca_settat, tangier_tetouan, rabat_sale, etc.)
        aspect: Specific aspect (sectors, infrastructure, workforce, advantages)
    """
    regional_data = {
        "casablanca_settat": {
            "overview": "Morocco's economic powerhouse with 32% of GDP, strong in automotive, aerospace, finance",
            "sectors": "Automotive (Renault, PSA), Aerospace (Boeing, Safran), Finance, Logistics, Textiles",
            "infrastructure": "Port of Casablanca (40M tons), Mohammed V Airport, Industrial zones",
            "workforce": "2.5M active, multilingual, highly skilled technical workforce"
        },
        "tangier_tetouan": {
            "overview": "Gateway to Europe with 8% of GDP, manufacturing and logistics hub",
            "sectors": "Manufacturing, Automotive, Textiles, Logistics",
            "infrastructure": "Tanger Med Port (largest in Africa), Free zones, Automotive city",
            "workforce": "800K active, competitive costs, proximity to Europe"
        }
    }
    
    return regional_data.get(region, {}).get(aspect, f"No specific data available for {region} - {aspect}")

@tool
def calculate_investment_costs(sector: str, region: str, investment_size: str) -> str:
    """Calculate estimated investment costs for Morocco projects.
    
    Args:
        sector: Business sector (manufacturing, services, logistics, etc.)
        region: Target region for investment
        investment_size: Size category (small: <5M MAD, medium: 5-50M MAD, large: >50M MAD)
    """
    base_costs = {
        "manufacturing": {
            "land_cost_per_m2": 350,
            "construction_cost_per_m2": 4000,
            "equipment_multiplier": 2.5
        },
        "services": {
            "land_cost_per_m2": 600,
            "construction_cost_per_m2": 3500,
            "equipment_multiplier": 1.2
        }
    }
    
    regional_multipliers = {
        "casablanca_settat": 1.2,
        "tangier_tetouan": 1.0,
        "rabat_sale": 1.1
    }
    
    size_factors = {
        "small": 1000,  # m2
        "medium": 5000,
        "large": 15000
    }
    
    if sector not in base_costs or region not in regional_multipliers:
        return "Unable to calculate costs - invalid sector or region"
    
    area = size_factors.get(investment_size, 5000)
    base = base_costs[sector]
    multiplier = regional_multipliers[region]
    
    land_cost = area * base["land_cost_per_m2"] * multiplier
    construction_cost = area * base["construction_cost_per_m2"] * multiplier
    equipment_cost = construction_cost * base["equipment_multiplier"]
    
    total_cost = land_cost + construction_cost + equipment_cost
    
    return f"""Investment Cost Estimate for {sector} in {region}:
    - Land cost: {land_cost:,.0f} MAD
    - Construction: {construction_cost:,.0f} MAD  
    - Equipment: {equipment_cost:,.0f} MAD
    - Total estimated cost: {total_cost:,.0f} MAD
    - USD equivalent: ${total_cost/10:,.0f} (approx.)
    """

@tool
def get_regulatory_timeline(business_type: str, sector: str) -> str:
    """Get regulatory timeline and requirements for Morocco business setup.
    
    Args:
        business_type: Type of business structure (SARL, SA, SAS, Branch)
        sector: Business sector for specific requirements
    """
    base_timeline = {
        "SARL": "6-8 weeks total timeline, minimum 10,000 MAD capital",
        "SA": "8-10 weeks total timeline, minimum 300,000 MAD capital", 
        "SAS": "6-8 weeks total timeline, minimum 10,000 MAD capital",
        "Branch": "4-6 weeks total timeline, no minimum capital requirement"
    }
    
    sector_additions = {
        "manufacturing": "+2-4 weeks for industrial permits and environmental clearance",
        "financial": "+4-6 weeks for financial sector licensing",
        "healthcare": "+6-8 weeks for health ministry approvals",
        "education": "+3-5 weeks for education ministry permits"
    }
    
    timeline = base_timeline.get(business_type, "Standard 6-8 weeks")
    additional = sector_additions.get(sector, "No additional sector-specific requirements")
    
    return f"""Regulatory Timeline for {business_type} in {sector}:
    Base timeline: {timeline}
    Sector-specific additions: {additional}
    
    Key steps:
    1. Business plan validation (1-2 weeks)
    2. Legal structure setup (1 week)
    3. OMPIC registration (2-3 days)
    4. Bank account and capital deposit (1 day)
    5. Commercial court registration (5-10 days)
    6. Tax and social security registration (1 week)
    7. Sector permits and licenses (varies by sector)
    """

class EnhancedMoroccoAgent:
    """Enhanced Morocco investment agent using LangChain"""
    
    def __init__(self, agent_type: str, config: DeepSeekConfig, knowledge_base: MoroccoKnowledgeBase):
        self.agent_type = agent_type
        self.config = config
        self.knowledge_base = knowledge_base
        
        # Initialize LangChain ChatOpenAI with DeepSeek
        self.llm = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        # Memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Keep last 5 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        # Redis for caching
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        
        # Initialize agent with tools
        self.tools = [get_regional_info, calculate_investment_costs, get_regulatory_timeline]
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create LangChain agent with Morocco-specific tools"""
        
        system_prompt = f"""
You are a specialized Morocco Investment Expert with focus on {self.agent_type.replace('_', ' ').title()}.

Your expertise includes:
- Comprehensive knowledge of Morocco's investment landscape
- Regional economic profiles and sector analysis
- Regulatory processes and legal requirements  
- Financial incentives and cost analysis
- Practical implementation guidance

Guidelines:
1. Use the available tools to get accurate, up-to-date information
2. Always provide specific numbers, timeframes, and requirements
3. Reference official sources and government agencies
4. Explain complex processes step-by-step
5. Suggest practical next steps for investors
6. Use MAD (Moroccan Dirham) for all monetary amounts
7. Be professional but approachable in tone

Available tools help you access:
- Regional economic data and comparisons
- Investment cost calculations
- Regulatory timelines and requirements

Current date: {datetime.now().strftime('%Y-%m-%d')}
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
    
    async def process_query(self, user_query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process query using LangChain agent with enhanced context"""
        
        # Check cache
        cache_key = f"enhanced_query:{self.agent_type}:{hash(user_query)}"
        cached_response = self.redis_client.get(cache_key)
        if cached_response:
            return json.loads(cached_response)
        
        try:
            # Get relevant context from knowledge base
            relevant_docs = self.knowledge_base.retrieve_relevant_info(user_query)
            context_info = "\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            # Enhanced query with context
            enhanced_query = f"""
Context from knowledge base:
{context_info}

User Query: {user_query}

Please provide a comprehensive response using the available tools and context information.
"""
            
            # Track token usage
            callback = TokenUsageCallback()
            
            # Process with agent
            response = await self.agent.ainvoke(
                {"input": enhanced_query},
                callbacks=[callback]
            )
            
            result = {
                "agent_type": self.agent_type,
                "response": response["output"],
                "tools_used": [tool.name for tool in self.tools],
                "context_documents": len(relevant_docs),
                "usage": {
                    "total_tokens": callback.total_tokens,
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 1 hour
            self.redis_client.setex(cache_key, 3600, json.dumps(result))
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

class EnhancedAgentOrchestrator:
    """Enhanced orchestrator using LangChain for better agent coordination"""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.knowledge_base = MoroccoKnowledgeBase(config)
        
        # Initialize enhanced agents
        self.agents = {
            "regional_expert": EnhancedMoroccoAgent("regional_expert", config, self.knowledge_base),
            "regulatory": EnhancedMoroccoAgent("regulatory", config, self.knowledge_base), 
            "financial": EnhancedMoroccoAgent("financial", config, self.knowledge_base)
        }
        
        # Router LLM
        self.router_llm = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            temperature=0.3
        )
    
    async def route_query(self, user_query: str) -> Dict[str, Any]:
        """Enhanced query routing using LangChain"""
        
        routing_prompt = ChatPromptTemplate.from_template("""
Analyze this Morocco investment query and determine which expert agents should respond.

Query: "{query}"

Available agents:
- regional_expert: Regional analysis, sector strengths, infrastructure, workforce, comparative analysis
- regulatory: Legal processes, permits, compliance, government procedures, business setup
- financial: Investment costs, incentives, financing options, ROI analysis, tax benefits

Instructions:
1. Select ONE primary agent (most relevant)
2. Select secondary agents if query needs multiple expertise areas (max 2)
3. Provide brief reasoning

Return ONLY valid JSON:
{{"primary": "agent_name", "secondary": ["agent_name"], "reasoning": "explanation"}}
""")
        
        try:
            messages = routing_prompt.format_messages(query=user_query)
            response = await self.router_llm.ainvoke(messages)
            
            # Clean and parse JSON response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            routing_result = json.loads(content)
            return routing_result
            
        except Exception as e:
            # Enhanced fallback routing
            query_lower = user_query.lower()
            
            financial_keywords = ["cost", "price", "financing", "incentive", "tax", "budget", "investment", "roi", "funding"]
            regulatory_keywords = ["permit", "legal", "process", "registration", "compliance", "license", "setup", "procedure"]
            regional_keywords = ["region", "city", "location", "sector", "industry", "infrastructure", "workforce"]
            
            financial_score = sum(1 for word in financial_keywords if word in query_lower)
            regulatory_score = sum(1 for word in regulatory_keywords if word in query_lower)
            regional_score = sum(1 for word in regional_keywords if word in query_lower)
            
            scores = {
                "financial": financial_score,
                "regulatory": regulatory_score, 
                "regional_expert": regional_score
            }
            
            primary = max(scores, key=scores.get)
            
            return {
                "primary": primary,
                "secondary": [],
                "reasoning": f"Fallback routing based on keyword analysis. Error: {str(e)}"
            }
    
    async def process_investment_query(self, user_query: str, conversation_history: Optional[List] = None) -> Dict[str, Any]:
        """Enhanced query processing with LangChain orchestration"""
        
        # Route query
        routing = await self.route_query(user_query)
        
        # Process with primary agent
        primary_result = await self.agents[routing["primary"]].process_query(user_query)
        
        # Process with secondary agents if needed
        secondary_results = []
        for agent_type in routing.get("secondary", []):
            if agent_type in self.agents:
                result = await self.agents[agent_type].process_query(user_query)
                secondary_results.append(result)
        
        # Synthesize if multiple agents involved
        if secondary_results:
            final_response = await self._synthesize_responses(
                primary_result, secondary_results, user_query, routing
            )
        else:
            final_response = {
                "response": primary_result["response"],
                "agents_used": [primary_result["agent_type"]],
                "routing_reasoning": routing.get("reasoning", ""),
                "total_tokens": primary_result["usage"]["total_tokens"],
                "tools_used": primary_result.get("tools_used", []),
                "context_documents": primary_result.get("context_documents", 0),
                "timestamp": datetime.now().isoformat()
            }
        
        return final_response
    
    async def _synthesize_responses(self, primary_result: Dict, secondary_results: List[Dict], 
                                  user_query: str, routing: Dict) -> Dict[str, Any]:
        """Synthesize multiple agent responses using LangChain"""
        
        synthesis_prompt = ChatPromptTemplate.from_template("""
User asked: "{query}"

PRIMARY EXPERT RESPONSE ({primary_agent}):
{primary_response}

SECONDARY EXPERT RESPONSES:
{secondary_responses}

Synthesize these expert responses into a single, comprehensive answer that:
1. Directly addresses the user's question
2. Integrates insights from all experts seamlessly
3. Maintains specific details, numbers, and sources
4. Provides clear, actionable next steps
5. Is well-organized and easy to read
6. Eliminates redundancy between responses

Provide a cohesive, professional response:
""")
        
        secondary_text = ""
        for result in secondary_results:
            secondary_text += f"\n{result['agent_type'].replace('_', ' ').title()} Expert:\n{result['response']}\n"
        
        messages = synthesis_prompt.format_messages(
            query=user_query,
            primary_agent=primary_result['agent_type'],
            primary_response=primary_result['response'],
            secondary_responses=secondary_text
        )
        
        try:
            synthesis_response = await self.router_llm.ainvoke(messages)
            
            total_tokens = primary_result["usage"]["total_tokens"]
            agents_used = [primary_result["agent_type"]]
            all_tools = primary_result.get("tools_used", [])
            total_context_docs = primary_result.get("context_documents", 0)
            
            for result in secondary_results:
                total_tokens += result["usage"]["total_tokens"]
                agents_used.append(result["agent_type"])
                all_tools.extend(result.get("tools_used", []))
                total_context_docs += result.get("context_documents", 0)
            
            return {
                "response": synthesis_response.content,
                "agents_used": agents_used,
                "routing_reasoning": routing.get("reasoning", ""),
                "total_tokens": total_tokens,
                "tools_used": list(set(all_tools)),
                "context_documents": total_context_docs,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Fallback to primary response
            return {
                "response": f"Based on expert analysis:\n\n{primary_result['response']}",
                "agents_used": [primary_result["agent_type"]],
                "routing_reasoning": f"Synthesis failed: {str(e)}",
                "total_tokens": primary_result["usage"]["total_tokens"],
                "tools_used": primary_result.get("tools_used", []),
                "context_documents": primary_result.get("context_documents", 0),
                "timestamp": datetime.now().isoformat(),
                "error": f"Synthesis error: {str(e)}"
            }

# FastAPI Application with Enhanced Features
app = FastAPI(title="Enhanced Morocco Investment Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize enhanced orchestrator
config = DeepSeekConfig()
orchestrator = EnhancedAgentOrchestrator(config)

class EnhancedQueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    preferred_agent: Optional[str] = None

class EnhancedQueryResponse(BaseModel):
    response: str
    agents_used: List[str]
    routing_reasoning: str
    total_tokens: int
    tools_used: List[str]
    context_documents: int
    timestamp: str

@app.post("/chat", response_model=EnhancedQueryResponse)
async def enhanced_chat_endpoint(request: EnhancedQueryRequest):
    """Enhanced chat endpoint with LangChain integration"""
    try:
        result = await orchestrator.process_investment_query(
            request.query,
            request.conversation_history
        )
        return EnhancedQueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available agents and their capabilities"""
    return {
        "agents": {
            "regional_expert": {
                "description": "Regional analysis and sector expertise",
                "capabilities": ["Regional comparisons", "Sector analysis", "Infrastructure assessment", "Workforce analysis"]
            },
            "regulatory": {
                "description": "Legal processes and compliance",
                "capabilities": ["Business setup procedures", "Permit requirements", "Legal structures", "Compliance guidelines"]
            },
            "financial": {
                "description": "Investment costs and financing",
                "capabilities": ["Cost analysis", "Investment incentives", "Financing options", "ROI calculations"]
            }
        },
        "tools": [
            "get_regional_info: Detailed regional information",
            "calculate_investment_costs: Investment cost estimation",
            "get_regulatory_timeline: Regulatory process timeline"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "langchain_version": "0.0.350",
        "vector_store": "initialized",
        "agents": len(orchestrator.agents)
    }
@app.on_event("startup")
async def startup_event():
    # Preload models and dependencies
    MoroccoKnowledgeBase(config).retrieve_relevant_info("test") 
    print("➔ Preloaded all ML models and dependencies")

@app.get("/ready")
async def readiness_check():
    return {"status": "ready"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable auto-reload
        timeout_keep_alive=100  # Add this line
    )