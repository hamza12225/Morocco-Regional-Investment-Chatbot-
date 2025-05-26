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
        
        knowledge_documents.extend([
        # Strategic Objectives
        Document(
            page_content="""
            Morocco Investment Charter Objectives:
            - Create stable employment opportunities to support economic growth.
            - Reduce territorial disparities through equitable regional development.
            - Promote investment in priority sectors and Next Generation Industries (e.g., digital technologies, renewable energy, pharmaceuticals).
            - Strengthen Morocco's position as a continental and international hub for foreign direct investment (FDI).
            - Encourage export growth and expansion of Moroccan companies abroad.
            - Substitute imports with local production.
            - Achieve sustainable development through eco-friendly practices.
            - Improve the business climate and streamline investment processes.
            - Increase private investment to 65% of total investment by 2035.
            - Target: 550 billion MAD in private investment and 500,000 jobs by 2026.
            """,
            metadata={"type": "strategic", "category": "objectives"}
        ),

        # Main Support Mechanism
        Document(
            page_content="""
            Main Investment Support Mechanism:
            - Eligibility: Investment projects ≥ 50 million MAD or creating ≥ 50 permanent jobs.
            - Incentives:
            - Stable Jobs: 5% (ratio >1 to ≤1.5), 7% (ratio >1.5 to ≤3), 10% (ratio >3) of investment amount.
            - Gender Ratio: 3% for projects with balanced women’s salaries.
            - Future Professions & Industry Upgrading: 3% for projects in Next Generation Industries (e.g., AI, biotechnology, renewable energy).
            - Sustainable Projects: 3% for projects using non-conventional water and meeting sustainability criteria.
            - Local Inclusion: 3% for projects with ≥20% local inclusion (agri-food, pharmaceuticals) or ≥40% (other manufacturing).
            - Published: Official Bulletin, March 13, 2023.
            """,
            metadata={"type": "financial", "category": "incentives_main"}
        ),

        # Strategic Projects Support
        Document(
            page_content="""
            Specific Support Mechanism for Strategic Projects:
            - Eligibility: Projects ≥ 2 billion MAD, contributing to water, energy, food, or health security, or meeting criteria like significant job creation, economic influence, or technology development (e.g., defense industry).
            - Incentives: Tailored support measures aligned with investor needs.
            - Contact: Ministry of Investment, Convergence and Evaluation of Public Policies.
            - Published: Decree adopted January 26, 2023.
            """,
            metadata={"type": "financial", "category": "incentives_strategic"}
        ),

        # Moroccan Businesses Abroad
        Document(
            page_content="""
            Specific Support Mechanism for Moroccan Businesses Abroad:
            - Objective: Enhance Morocco’s economic influence, especially in Africa.
            - Conditions: Support provided without causing job losses in Morocco.
            - Status: Application text to be published post-Framework Law.
            """,
            metadata={"type": "financial", "category": "incentives_international"}
        ),

        # Small and Medium Enterprises
        Document(
            page_content="""
            Specific Support Mechanism for Very Small, Small, and Medium-Sized Enterprises:
            - Targets: Majority of Moroccan entrepreneurial landscape.
            - Support: Financial incentives, plus assistance in financing, training, and project structuring.
            - Status: Application text to be published post-Framework Law.
            """,
            metadata={"type": "financial", "category": "incentives_sme"}
        ),

        # Next Generation Industries
        Document(
            page_content="""
            Next Generation Industries:
            - Digital Technologies: Biotechnology, cybersecurity, blockchain, cloud computing, AI, IoT, nanotechnology, agritech, healthtech, edtech, fintech, govtech, virtual/augmented reality.
            - Pharmaceuticals: Medical cannabis transformation, medical device manufacturing, vaccines.
            - Renewable Energy: Production and storage facilities, energy efficiency technologies.
            - Other Sectors: Semiconductors, smart meters, 3D printing, robotics, shipbuilding, electric and autonomous mobility.
            - Published: Official Bulletin, March 13, 2023.
            """,
            metadata={"type": "sectoral", "category": "next_generation"}
        ),

        # Level-Upgrading Activities
        Document(
            page_content="""
            Level-Upgrading Activities:
            - Automotive: Spare parts for thermal/electric motors, heavy vehicles, pneumatic wheels.
            - Aerospace: Ancillary equipment, aircraft engine parts, aircraft manufacturing/dismantling.
            - Agricultural: Animal feed, baby food, nutritional supplements, health-focused food products, irrigation equipment.
            - Textile/Leather: Technical fabrics and skins.
            - Pharmaceuticals: Medical devices, medicines, vaccines, aromatic/medicinal plants.
            - Mining: High-value derivatives, phosphate-based products.
            - Energy Transition: Seawater desalination equipment.
            - Published: Official Bulletin, March 13, 2023.
            """,
            metadata={"type": "sectoral", "category": "level_upgrading"}
        ),

        # Sustainable Investment Projects
        Document(
            page_content="""
            Sustainable Investment Projects:
            - Mandatory Requirement: Use non-conventional water (recycled, treated wastewater, or desalinated) and water economy systems.
            - Additional Criteria (at least two): Renewable energy use, energy efficiency systems, waste treatment systems, social responsibility programs.
            - Incentive: 3% of investment amount.
            - Published: Official Bulletin, March 13, 2023.
            """,
            metadata={"type": "financial", "category": "incentives_sustainable"}
        ),

        # Local Inclusion Projects
        Document(
            page_content="""
            Local Inclusion Projects:
            - Eligibility: ≥20% local inclusion for agri-food, pharmaceutical, or medical supplies industries; ≥40% for other manufacturing.
            - Calculation: (Local Purchases + Value Added + Crude Margin) / Turnover.
            - Incentive: 3% of investment amount.
            - Published: Official Bulletin, March 13, 2023.
            """,
            metadata={"type": "financial", "category": "incentives_local"}
        ),

        # Territorial Incentives
        Document(
            page_content="""
            Territorial-Based Incentives:
            - Objective: Enhance territorial equity.
            - Excluded Regions: Benslimane, Berrechid, Casablanca, El Jadida, Médiouna, Mohammédia, Nouaceur, Settat, Marrakech, Kénitra, Rabat, Skhirate-Témara, Agadir Ida-Outanane, Fahs-Anjra, Tanger-Assilah.
            - Incentive: Additional support for projects in non-excluded regions to promote equitable development.
            """,
            metadata={"type": "financial", "category": "incentives_territorial"}
        ),

        # Sector-Based Incentives
        Document(
            page_content="""
            Sector-Based Incentives:
            - Targeted Sectors: Industry, tourism, real estate development, education, health, audiovisual industry, cultural industry, digital economy, agriculture, aquaculture, mining, logistics, renewable energy, recycling, waste recovery.
            - Objective: Boost sectors with high growth potential.
            """,
            metadata={"type": "financial", "category": "incentives_sectoral"}
        ),

        # Business Climate Improvements
        Document(
            page_content="""
            Business Climate Improvements (2023-2026 Roadmap):
            - Pillar 1: Simplify investment processes, digitize procedures, decentralize administration, reduce payment terms.
            - Pillar 2: Enhance competitiveness via financing, renewable energy access, industrial decarbonization, logistics improvements.
            - Pillar 3: Promote entrepreneurship, innovation, R&D, and human capital training.
            - Cross-Functional Pillar: Strengthen ethics, integrity, and anti-corruption measures.
            """,
            metadata={"type": "regulatory", "category": "business_climate"}
        ),

        # Investment Governance
        Document(
            page_content="""
            Investment Governance:
            - Ministry of Investment, Convergence and Evaluation of Public Policies: Oversees strategic projects.
            - National Investment Commission: Chaired by the Head of Government, approves investment agreements, grants strategic status (secretariat by AMDIE).
            - Regional Investment Centers (RICs): One-stop shops for investors, manage regional projects (≤250 million MAD agreements at regional level; >250 million MAD at central level).
            """,
            metadata={"type": "regulatory", "category": "governance"}
        ),

        # Contact Information
        Document(
            page_content="""
            Investment Contacts:
            - AMDIE (Moroccan Investment and Export Development Agency):
            - Phone: +212 5372-26400
            - Email: morocconow@amdie.gov.ma
            - Address: Mahaj Riad Center, Avenue Attine, Building N°5 & N°7, Rabat 10100, Morocco
            - Website: www.morocconow.com
            - Ministry of Investment, Convergence and Evaluation of Public Policies:
            - Phone: +212 538061301
            - Address: ANCFCC, Tower 6, Boulevard Abderrahim Bouabid, Guich Oudaya, Agdal-Ryad, Rabat, Morocco
            - Regional Investment Centers (RICs):
            - Casablanca-Settat: +212522494242, info@casainvest.ma, www.casainvest.ma
            - Tangier-Tetouan-Al Hoceima: +212539342301, info@investangier.com, www.investangier.com
            - Fès-Meknes: +212535652057, info@fesmeknesinvest.ma, www.fesmeknesinvest.ma
            - Marrakech-Safi: +212524420493, contact@crimarrakech.ma, www.crimarrakech.ma
            - Souss-Massa: +212528230807, contact@cri-agadir.ma, www.cri-agadir.ma
            - the Rabat-Salé-Kénitra Region: +212537776400,info@rabatinvest.ma,www.rabatinvest.ma
            """,
            metadata={"type": "contacts", "category": "investment_support"}
        ),
         # Contact Information
        Document(
            page_content="""
            Economy and Trade in Morocco
            GDP Growth:
                GDP per capita (current US$): Increased from $3,435 in 2014 to $3,771 in 2023.
                GDP (constant LCU): Grew from 1.00 trillion in 2014 to 1.23 trillion in 2023.
                Trade:
                Merchandise exports to high-income economies: 78.1% of total exports in 2023, up from 75.1% in 2014.
                Manufactures exports: Increased from 66.9% to 74.7% of merchandise exports (2014–2023).
                Medium and high-tech exports: Rose from 48.9% to 64.9% of manufactured exports (2014–2022).
                Commercial service imports: Grew from $7.90 billion in 2014 to $11.03 billion in 2023.
                Foreign Direct Investment (FDI): Net inflows dropped from 2.96% of GDP in 2014 to 0.76% in 2023.
                Remittances: Personal remittances increased from $7.79 billion (6.54% of GDP) in 2014 to $11.75 billion (8.14% of GDP) in 2023.
                Debt Service: Total debt service decreased from 12.9% of exports in 2014 to 9.2% in 2023
            """,
            metadata={"type": "financial", "category": "Economy_and_trade"}
            )
        ])
        
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
        "overview": "Morocco's economic powerhouse, contributing 32% of national GDP, a hub for automotive, aerospace, finance, and logistics.",
        "sectors": "Automotive (Renault, Stellantis), Aerospace (Boeing, Safran), Finance, Logistics, Textiles, Pharmaceuticals",
        "infrastructure": "Port of Casablanca (40M tons capacity), Mohammed V International Airport, industrial zones (e.g., Midparc, Atlantic Free Zone), advanced rail and road networks",
        "workforce": "2.5M active workers, multilingual (Arabic, French, English, Spanish), highly skilled in technical and financial sectors",
        "advantages": "Economic capital, proximity to major ports and airports, established industrial ecosystems, high foreign direct investment (FDI) attractiveness"
    },
    "tangier_tetouan_al_hoceima": {
        "overview": "Strategic gateway to Europe, contributing 8% of GDP, a major manufacturing and logistics hub.",
        "sectors": "Automotive (Renault, Yazaki), Textiles, Logistics, Renewable Energy, Maritime Industries",
        "infrastructure": "Tanger Med Port (Africa’s largest, 9M TEU capacity), Tangier Free Zone, Tanger Automotive City, high-speed rail (Al Boraq)",
        "workforce": "800K active workers, competitive labor costs, multilingual (Arabic, French, Spanish), proximity to Europe",
        "advantages": "Strategic location (14 km from Europe), robust logistics infrastructure, free trade zones, growing renewable energy sector"
    },
    "oriental": {
        "overview": "Emerging region with focus on industrial growth and cross-border trade, contributing to Morocco’s eastern development.",
        "sectors": "Agri-food, Mining, Renewable Energy, Tourism, Logistics",
        "infrastructure": "Nador West Med Port (under development), Oujda-Angad Airport, industrial parks (e.g., Selouane Industrial Zone), road connectivity to Algeria",
        "workforce": "600K active workers, skilled in agriculture and mining, growing technical expertise",
        "advantages": "Proximity to Mediterranean markets, incentives for sustainable projects, potential for cross-border trade"
    },
    "fes_meknes": {
        "overview": "Cultural and industrial hub with a focus on agriculture, manufacturing, and education.",
        "sectors": "Agri-food, Textiles, Leather, Renewable Energy, Education (Edtech)",
        "infrastructure": "Fès-Saïss Airport, industrial zones (e.g., Ain Cheggag), rail and road connectivity to major cities",
        "workforce": "700K active workers, skilled in agriculture, textiles, and emerging tech sectors, supported by universities",
        "advantages": "Rich cultural heritage, growing renewable energy sector, strong academic ecosystem for R&D"
    },
    "rabat_sale_kenitra": {
        "overview": "Administrative and industrial hub, contributing to Morocco’s governance and economic diversification.",
        "sectors": "Automotive, Aerospace, IT (Govtech, Fintech), Public Services, Logistics",
        "infrastructure": "Rabat-Salé Airport, Technopolis Rabat (tech and innovation hub), Kenitra Atlantic Free Zone, high-speed rail",
        "workforce": "1.2M active workers, highly educated, multilingual, strong in administrative and technical skills",
        "advantages": "Political stability, proximity to government institutions, innovation-driven ecosystem, access to Atlantic ports"
    },
    "beni_mellal_khenifra": {
        "overview": "Agricultural and renewable energy hub with focus on sustainable development.",
        "sectors": "Agri-food, Renewable Energy, Mining (Phosphates), Tourism",
        "infrastructure": "Beni Mellal Airport, industrial zones, irrigation networks for agriculture, road connectivity",
        "workforce": "500K active workers, skilled in agriculture and mining, growing renewable energy expertise",
        "advantages": "Rich agricultural resources, phosphate mining potential, incentives for sustainable projects"
    },
    "marrakech_safi": {
        "overview": "Tourism and industrial hub with a focus on sustainable growth and cultural heritage.",
        "sectors": "Tourism, Agri-food, Renewable Energy, Mining, Handicrafts",
        "infrastructure": "Marrakech Menara Airport, industrial zones (e.g., Sidi Ghanem), solar power plants (Noor Ouarzazate nearby)",
        "workforce": "900K active workers, skilled in tourism, agriculture, and artisanal crafts",
        "advantages": "Global tourism destination, renewable energy projects, growing industrial base"
    },
    "souss_massa": {
        "overview": "Agribusiness and tourism hub with strong export potential.",
        "sectors": "Agri-food (citrus, fisheries), Tourism, Renewable Energy, Logistics",
        "infrastructure": "Agadir Al Massira Airport, Port of Agadir, Souss Massa Free Zone, solar and wind projects",
        "workforce": "700K active workers, skilled in agriculture, fishing, and tourism",
        "advantages": "Export-oriented agribusiness, proximity to Atlantic ports, renewable energy incentives"
    },
    "draa_tafilalet": {
        "overview": "Emerging region with focus on agriculture, tourism, and renewable energy.",
        "sectors": "Agri-food, Tourism (desert tourism), Renewable Energy, Mining",
        "infrastructure": "Errachidia Airport, solar power plants (Noor Midelt), oasis irrigation systems",
        "workforce": "400K active workers, skilled in agriculture and tourism, emerging technical skills",
        "advantages": "Unique tourism offerings, renewable energy potential, incentives for territorial equity"
    },
    "laayoune_sakia_el_hamra": {
        "overview": "Strategic southern region with focus on renewable energy and fisheries.",
        "sectors": "Renewable Energy, Fisheries, Mining (Phosphates), Tourism",
        "infrastructure": "Laayoune Port, Hassan I Airport, solar and wind farms, industrial zones",
        "workforce": "300K active workers, skilled in fisheries and renewable energy, growing technical expertise",
        "advantages": "Significant renewable energy projects, phosphate resources, territorial incentives"
    },
    "guelmim_oued_noun": {
        "overview": "Developing region with focus on connectivity and sustainable industries.",
        "sectors": "Agri-food, Fisheries, Tourism, Renewable Energy",
        "infrastructure": "Guelmim Airport, Port of Tan-Tan, renewable energy projects, road connectivity",
        "workforce": "250K active workers, skilled in fisheries and agriculture, emerging renewable energy skills",
        "advantages": "Territorial incentives, gateway to West Africa, sustainable project support"
    },
    "dakhla_oued_ed_dahab": {
        "overview": "Emerging hub for renewable energy, fisheries, and tourism in Morocco’s far south.",
        "sectors": "Renewable Energy, Fisheries, Tourism, Logistics",
        "infrastructure": "Dakhla Atlantic Port (under development), Dakhla Airport, wind farms, free trade zone",
        "workforce": "200K active workers, skilled in fisheries and tourism, growing renewable energy expertise",
        "advantages": "Strategic location for Africa trade, major renewable energy projects, territorial incentives"
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
        sector: Business sector for specific requirements (e.g., manufacturing, renewable energy, agri-food, tourism)
    
    Returns:
        A string detailing the regulatory timeline, key steps, and requirements for setting up the specified business type in the given sector, aligned with Morocco’s Investment Charter.
    """
    # Base timelines and requirements for business types
    base_timeline = {
        "SARL": {
            "timeline": "6-8 weeks",
            "capital": "Minimum 10,000 MAD (no minimum required if single shareholder)",
            "description": "Limited Liability Company, suitable for SMEs, flexible structure."
        },
        "SA": {
            "timeline": "8-10 weeks",
            "capital": "Minimum 300,000 MAD (or 3M MAD if publicly listed)",
            "description": "Public Limited Company, ideal for large enterprises with complex governance."
        },
        "SAS": {
            "timeline": "6-8 weeks",
            "capital": "No minimum capital requirement, flexible for shareholders",
            "description": "Simplified Joint-Stock Company, suitable for startups and joint ventures."
        },
        "Branch": {
            "timeline": "4-6 weeks",
            "capital": "No minimum capital requirement",
            "description": "Extension of a foreign company, no independent legal entity status."
        }
    }

    # Sector-specific additional timelines and requirements, aligned with Investment Charter
    sector_additions = {
        "manufacturing": {
            "additional_time": "+2-4 weeks",
            "requirements": "Industrial permits (e.g., environmental clearance, factory licensing), compliance with local inclusion rates (20-40% per page 13)",
            "charter_incentives": "Eligible for main support mechanism (page 8) and sector-based incentives for automotive, aerospace, textiles, etc."
        },
        "renewable_energy": {
            "additional_time": "+3-5 weeks",
            "requirements": "MASEN (Moroccan Agency for Sustainable Energy) approvals, environmental impact assessments, compliance with sustainable project criteria ",
            "charter_incentives": "Incentives for renewable energy production/storage , capped incentives , territorial incentives in regions like Laayoune or Dakhla "
        },
        "agri_food": {
            "additional_time": "+2-4 weeks",
            "requirements": "Ministry of Agriculture approvals, food safety certifications, 20% local inclusion rate ",
            "charter_incentives": "Support for agri-food projects (page 11), territorial incentives for equitable development "
        },
        "tourism": {
            "additional_time": "+2-3 weeks",
            "requirements": "Ministry of Tourism permits, environmental and cultural heritage compliance",
            "charter_incentives": "Sector-based incentives for tourism projects (page 15), focus on regions like Marrakech-Safi and Dakhla"
        },
        "pharmaceuticals": {
            "additional_time": "+6-8 weeks",
            "requirements": "Ministry of Health approvals, GMP (Good Manufacturing Practice) certification, 20% local inclusion rate ",
            "charter_incentives": "Support for medical device and vaccine manufacturing (page 11), strategic project status for health security "
        },
        "financial": {
            "additional_time": "+4-6 weeks",
            "requirements": "Bank Al-Maghrib or AMMC (Capital Market Authority) licensing, compliance with fintech regulations",
            "charter_incentives": "Support for fintech and govtech projects "
        },
        "education": {
            "additional_time": "+3-5 weeks",
            "requirements": "Ministry of Education permits, accreditation for edtech platforms",
            "charter_incentives": "Support for edtech and training programs "
        },
        "digital_technologies": {
            "additional_time": "+2-4 weeks",
            "requirements": "Digital Development Agency (ADD) compliance, data protection regulations for AI, blockchain, etc.",
            "charter_incentives": "Incentives for next-generation industries like AI, blockchain, and IoT"
        }
    }

    # Default values if business_type or sector not found
    timeline_info = base_timeline.get(business_type, {
        "timeline": "6-8 weeks",
        "capital": "Varies by structure",
        "description": "Standard business structure"
    })
    sector_info = sector_additions.get(sector, {
        "additional_time": "No additional time",
        "requirements": "No sector-specific requirements",
        "charter_incentives": "Eligible for main support mechanism if investment ≥50M MAD or creates ≥50 jobs"
    })

    # Construct the response
    return f"""Regulatory Timeline for {business_type} in {sector}:

Base Timeline and Requirements:
- Timeline: {timeline_info['timeline']}
- Capital: {timeline_info['capital']}
- Description: {timeline_info['description']}

Sector-Specific Additions:
- Additional Time: {sector_info['additional_time']}
- Requirements: {sector_info['requirements']}
- Investment Charter Incentives: {sector_info['charter_incentives']}

Key Steps for Business Setup in Morocco (aligned with Investment Charter, pages 20-22):
1. Business Plan Validation (1-2 weeks): Submit plan to Regional Investment Center (RIC) for initial review. Contact relevant RIC for support.
2. Legal Structure Setup (1 week): Draft articles of association with a notary, ensuring compliance with business type requirements.
3. OMPIC Registration (2-3 days): Register with the Office Marocain de la Propriété Industrielle et Commerciale (OMPIC) for commercial registration.
4. Bank Account and Capital Deposit (1 day): Open a corporate bank account and deposit required capital (if applicable).
5. Commercial Court Registration (5-10 days): Register with the Commercial Court for legal recognition.
6. Tax and Social Security Registration (1 week): Obtain tax ID and register with CNSS (National Social Security Fund).
7. Sector-Specific Permits and Licenses (varies): Obtain approvals from relevant ministries (e.g., Health, Agriculture) or agencies (e.g., MASEN, ADD) as per sector requirements.
8. RIC Support (ongoing): Leverage RIC one-stop shops for streamlined procedures, especially for projects <250M MAD (page 23).

Additional Notes:
- For projects ≥250M MAD, agreements are prepared centrally by the National Investment Commission .
- Strategic projects (≥2B MAD, impacting water/energy/food/health security,) should contact the Ministry of Investment at +212538061301.
- SMEs may access specific support for financing and training.
- Check territorial incentives for regions like Beni Mellal-Khenifra or Dakhla to reduce disparities.
- Contact AMDIE at +212537226400 or morocconow@amdie.gov.ma for export-oriented or diaspora investment support.
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