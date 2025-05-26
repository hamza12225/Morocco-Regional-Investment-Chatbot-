# Morocco Investment Assistant 🇲🇦

A sophisticated AI-powered investment advisory system that provides comprehensive guidance for investing in Morocco. Built with LangChain, FastAPI, and DeepSeek AI, this system features specialized agents for regional analysis, regulatory compliance, and financial planning.

## ✨ Features

### Multi-Agent System
- **Regional Expert**: Analyzes regions, sectors, infrastructure, and workforce
- **Regulatory Specialist**: Handles legal processes, permits, and compliance
- **Financial Advisor**: Provides cost analysis, incentives, and financing options

### Advanced AI Capabilities
- 🤖 **DeepSeek AI Integration** for intelligent responses
- 🔍 **Vector Search** with Chroma DB for accurate information retrieval
- 🧠 **Memory Management** maintains conversation context
- 🛠️ **LangChain Tools** for specialized calculations and data access
- 📊 **Ensemble Retrieval** combining vector and keyword search

### Comprehensive Knowledge Base
- Latest Morocco Investment Charter (2023-2026)
- Regional economic profiles and sector analysis
- Investment incentives and tax benefits
- Regulatory processes and timelines
- Cost calculations and ROI analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Redis server
- DeepSeek API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/hamza12225/Morocco-Regional-Investment-Chatbot.git
cd Morocco-Regional-Investment-Chatbot
```

2. **Create and activate virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the project root:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
REDIS_URL=redis://localhost:6379
```

5. **Start Redis server**
```bash
# On macOS with Homebrew
brew services start redis

# On Ubuntu/Debian
sudo systemctl start redis-server

# On Windows (if using WSL or native Redis)
redis-server
# Or install redis server on Docker
```

6. **Run the application**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
  "query": "What are the investment incentives for manufacturing in Casablanca?",
  "conversation_history": [],
  "preferred_agent": "financial"
}
```

**Response:**
```json
{
  "response": "Detailed response from specialized agents...",
  "agents_used": ["financial", "regional_expert"],
  "routing_reasoning": "Query requires financial and regional expertise",
  "total_tokens": 1250,
  "tools_used": ["calculate_investment_costs", "get_regional_info"],
  "context_documents": 3,
  "timestamp": "2024-01-20T10:30:00"
}
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Main chat interface with intelligent agent routing |
| `/agents` | GET | List available agents and their capabilities |
| `/health` | GET | System health check |
| `/ready` | GET | Readiness probe for deployment |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

## 🛠️ Agent Capabilities

### Regional Expert Agent
- **Regional Comparisons**: Casablanca-Settat vs Tangier-Tetouan analysis
- **Sector Analysis**: Automotive, aerospace, manufacturing, services
- **Infrastructure Assessment**: Ports, airports, industrial zones
- **Workforce Analysis**: Skills, costs, availability

### Regulatory Agent
- **Business Setup**: SARL, SA, SAS structures and requirements
- **Permit Processing**: Industry-specific licenses and approvals
- **Compliance Guidelines**: Tax registration, social security, legal obligations
- **Timeline Estimation**: Step-by-step process duration

### Financial Agent
- **Investment Costing**: Land, construction, equipment calculations
- **Incentive Analysis**: Tax benefits, regional incentives, sector-specific support
- **Financing Options**: Local banks, international funding, government programs
- **ROI Calculations**: Return on investment projections

## 🔧 Configuration

### DeepSeek AI Configuration
```python
class DeepSeekConfig:
    api_key = "your_api_key"
    base_url = "https://api.deepseek.com"
    model = "deepseek-chat"
    max_tokens = 2000
    temperature = 0.7
```

### Vector Store Setup
The system automatically initializes a Chroma vector database with Morocco investment knowledge. The database persists in `./chroma_db/` directory.

### Redis Caching
NOTE : YOU NEED TO HAVE REDIS INSTALLED IN YOURS MACHINE AND SET TO DEFAULT PORT
Responses are cached for 1 hour to improve performance:
- Cache key format: `enhanced_query:{agent_type}:{query_hash}`
- TTL: 3600 seconds (1 hour)

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│   Orchestrator   │────│  Knowledge Base │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
            ┌───────▼────┐ ┌────▼────┐ ┌────▼─────┐
            │ Regional   │ │Regulatory│ │Financial │
            │   Agent    │ │  Agent   │ │  Agent   │
            └────────────┘ └─────────┘ └──────────┘
                    │           │           │
                    └───────────┼───────────┘
                           ┌────▼────┐
                           │ LangChain│
                           │  Tools   │
                           └─────────┘
```

## 🧪 Example Queries

### Investment Cost Analysis
```json
{
  "query": "What would it cost to set up a 5,000 m² manufacturing facility in Tangier with automotive components production?"
}
```

### Regulatory Process
```json
{
  "query": "What are the steps to register a SARL company in Morocco for textile manufacturing, and how long does each step take?"
}
```

### Regional Comparison
```json
{
  "query": "Compare Casablanca-Settat and Tangier-Tetouan regions for establishing a logistics hub. Include infrastructure, costs, and workforce analysis."
}
```

### Investment Incentives
```json
{
  "query": "What tax incentives and financial benefits are available for renewable energy projects in Morocco under the 2023 Investment Charter?"
}
```

## 🔍 Advanced Features

### Intelligent Query Routing
The system automatically routes queries to the most appropriate agents based on content analysis:
- Financial keywords → Financial Agent
- Regulatory terms → Regulatory Agent  
- Location/sector mentions → Regional Expert

### Vector Search Integration
- **Embedding Model**: `sentence-transformers/paraphrase-MiniLM-L6-v2`
- **Vector Store**: Chroma DB with persistence
- **Retrieval Strategy**: Ensemble retrieval (70% vector, 30% BM25)

### Memory Management
- **Window Size**: Last 5 conversation exchanges
- **Context Preservation**: Maintains conversation flow
- **Tool Integration**: Seamless tool usage within conversations

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Environment Variables for Production
```env
DEEPSEEK_API_KEY=your_production_key
REDIS_URL=redis://your-redis-host:6379
LOG_LEVEL=INFO
WORKERS=4
```

### Health Checks
- **Health Endpoint**: `/health` - System status
- **Readiness Endpoint**: `/ready` - Deployment readiness

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .
isort .

# Lint code
flake8
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Visit `/docs` endpoint for interactive API documentation
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Join our GitHub Discussions for questions and ideas

## 🙏 Acknowledgments

- **DeepSeek AI** for providing the language model capabilities
- **LangChain** for the agent framework and tools
- **Chroma DB** for vector storage and retrieval
- **FastAPI** for the high-performance web framework
- **Morocco Investment Authorities** for comprehensive investment data

---

**Built with ❤️ for Morocco's investment ecosystem**

> Transform your Morocco investment journey with AI-powered insights and expert guidance.
