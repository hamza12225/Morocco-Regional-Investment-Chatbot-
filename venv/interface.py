import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Streamlit page configuration
st.set_page_config(
    page_title="Morocco Investment Assistant",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #1976d2;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .agent-info {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        border-left: 3px solid #4caf50;
    }
    
    .metrics-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar-content {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        padding: 0.5rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .status-connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .routing-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        border-left: 3px solid #ffc107;
    }
    
    .processing-animation {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Backend API configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your deployed backend URL

class MoroccoInvestmentChat:
    def __init__(self):
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'total_tokens_used' not in st.session_state:
            st.session_state.total_tokens_used = 0
        if 'queries_count' not in st.session_state:
            st.session_state.queries_count = 0
        if 'agents_used' not in st.session_state:
            st.session_state.agents_used = set()
        if 'api_status' not in st.session_state:
            st.session_state.api_status = 'unknown'
        if 'available_agents' not in st.session_state:
            st.session_state.available_agents = {}
    
    def check_api_health(self) -> Dict:
        """Check backend API health status"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            ready = requests.get(f"{API_BASE_URL}/ready", timeout=5)

            if response.status_code == 200 and ready.status_code==200:
                health_data = response.json()
                st.session_state.api_status = 'connected'
                return health_data
            else:
                st.session_state.api_status = 'error'
                return {"error": f"API returned status {response.status_code}"}
        except requests.exceptions.RequestException as e:
            st.session_state.api_status = 'error'
            return {"error": f"Connection failed: {str(e)}"}
    
    def get_available_agents(self) -> Dict:
        """Get available agents and their capabilities"""
        try:
            response = requests.get(f"{API_BASE_URL}/agents", timeout=10)
            if response.status_code == 200:
                agents_data = response.json()
                st.session_state.available_agents = agents_data
                return agents_data
            else:
                return {"error": f"Failed to get agents: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def send_query(self, user_query: str) -> Dict:
        """Send query to enhanced backend API"""
        try:
            payload = {
                "query": user_query,
                "conversation_history": [
                    {"user" if msg["type"] == "user" else "assistant": msg["content"]} 
                    for msg in st.session_state.conversation_history[-10:]  # Last 10 exchanges
                ]
            }
            
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json=payload,
                timeout=300  # Increased timeout for agent processing
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def display_api_status(self):
        """Display API connection status"""
        if st.session_state.api_status == 'connected':
            st.markdown("""
            <div class="status-indicator status-connected">
                ✅ Connected to Morocco Investment API
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.api_status == 'error':
            st.markdown("""
            <div class="status-indicator status-error">
                ❌ API Connection Error - Check if backend is running
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator">
                🔄 Checking API connection...
            </div>
            """, unsafe_allow_html=True)
    
    def display_session_analytics(self):
        """Display enhanced session analytics"""
        if st.session_state.queries_count > 0:
            # Create metrics dataframe
            agents_list = list(st.session_state.agents_used)
            if agents_list:
                df_agents = pd.DataFrame({
                    'Agent': [agent.replace('_', ' ').title() for agent in agents_list],
                    'Usage': [1] * len(agents_list)  # Simplified for demo
                })
                
                # Agent usage chart
                fig = px.bar(df_agents, x='Agent', y='Usage', 
                           title='Agents Consulted This Session',
                           color='Agent')
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Token usage over time (simplified)
            if len(st.session_state.conversation_history) > 0:
                queries = list(range(1, st.session_state.queries_count + 1))
                tokens = [100, 250, 180, 320, 275][:len(queries)]  # Mock data for demo
                
                fig_tokens = go.Figure()
                fig_tokens.add_trace(go.Scatter(
                    x=queries, y=tokens,
                    mode='lines+markers',
                    name='Tokens per Query',
                    line=dict(color='#2a5298', width=3)
                ))
                fig_tokens.update_layout(
                    title='Token Usage Trend',
                    xaxis_title='Query Number',
                    yaxis_title='Tokens Used',
                    height=300
                )
                st.plotly_chart(fig_tokens, use_container_width=True)
    
    def display_main_interface(self):
        """Display enhanced main chat interface"""
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🇲🇦 Morocco Regional Investment Assistant</h1>
            <p>Powered by Multi-Agent AI System with LangChain & DeepSeek</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check API health on load
        health_status = self.check_api_health()
        available_agents = self.get_available_agents()
        
        # Sidebar with enhanced information
        with st.sidebar:
            st.markdown("## 🔧 System Status")
            self.display_api_status()
            
            if st.button("🔄 Refresh Connection"):
                health_status = self.check_api_health()
                st.rerun()
            
            st.markdown("---")
            
            st.markdown("### 📊 Session Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Queries", st.session_state.queries_count)
            with col2:
                st.metric("Tokens", st.session_state.total_tokens_used)
            with col3:
                st.metric("Agents", len(st.session_state.agents_used))
            
            # Display available agents
            if st.session_state.available_agents and 'agents' in st.session_state.available_agents:
                st.markdown("### 🤖 Available Experts")
                for agent_name, agent_info in st.session_state.available_agents['agents'].items():
                    with st.expander(f"🔍 {agent_name.replace('_', ' ').title()}"):
                        st.write(f"**Role:** {agent_info['description']}")
                        st.write("**Capabilities:**")
                        for capability in agent_info['capabilities']:
                            st.write(f"• {capability}")
            
            st.markdown("### 🎯 What I Can Help With")
            st.markdown("""
            **🏭 Regional Expertise:**
            - Compare investment opportunities across regions
            - Sector-specific market analysis
            - Infrastructure & logistics assessment
            
            **📋 Regulatory Guidance:**
            - Step-by-step investment procedures
            - Required permits and documentation
            - Legal structures and compliance
            
            **💰 Financial Analysis:**
            - Investment cost calculations
            - Available incentives and benefits
            - Financing options and ROI analysis
            """)
            
            st.markdown("### 🗺️ Key Investment Regions")
            regions_info = [
                "🏢 **Casablanca-Settat:** Finance & Industry Hub",
                "🚢 **Tangier-Tétouan:** Manufacturing & Logistics", 
                "🏛️ **Rabat-Salé:** Government & ICT Center",
                "🕌 **Marrakech-Safi:** Tourism & Agriculture",
                "🎓 **Fès-Meknès:** Traditional Industries & Education"
            ]
            
            for region in regions_info:
                st.markdown(region)
            
            st.markdown("---")
            
            if st.button("🗑️ Clear Conversation", type="secondary"):
                st.session_state.conversation_history = []
                st.session_state.total_tokens_used = 0
                st.session_state.queries_count = 0
                st.session_state.agents_used = set()
                st.rerun()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display conversation history
            if st.session_state.conversation_history:
                st.markdown("### 💬 Investment Consultation")
                
                for i, message in enumerate(st.session_state.conversation_history):
                    if message["type"] == "user":
                        st.markdown(f"""
                        <div class="user-message">
                            <strong>🧑‍💼 You:</strong><br>
                            {message['content']}
                            <div style="font-size: 0.7em; color: #666; margin-top: 0.5rem;">
                                {message['timestamp'].strftime('%H:%M:%S')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message">
                            <strong>🤖 Investment Assistant:</strong><br>
                            {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enhanced metadata display
                        if 'metadata' in message:
                            metadata = message['metadata']
                            
                            # Agents used
                            if 'agents_used' in metadata:
                                agents_str = ", ".join([
                                    agent.replace('_', ' ').title() 
                                    for agent in metadata['agents_used']
                                ])
                                st.markdown(f"""
                                <div class="agent-info">
                                    <strong>🔍 Experts Consulted:</strong> {agents_str}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Routing reasoning
                            if 'routing_reasoning' in metadata:
                                st.markdown(f"""
                                <div class="routing-info">
                                    <strong>🎯 Agent Selection:</strong> {metadata['routing_reasoning']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Technical details
                            col_meta1, col_meta2, col_meta3 = st.columns(3)
                            with col_meta1:
                                st.caption(f"Tokens: {metadata.get('total_tokens', 'N/A')}")
                            with col_meta2:
                                st.caption(f"Tools: {len(metadata.get('tools_used', []))}")
                            with col_meta3:
                                st.caption(f"Context Docs: {metadata.get('context_documents', 0)}")
                        
                        st.markdown("---")
            
            else:
                st.markdown("### 👋 Welcome to Morocco Investment Assistant")
                st.info("""
                🎯 **Ready to help you invest in Morocco!** 
                
                Ask me about:
                - Regional comparisons and opportunities
                - Investment costs and financial incentives  
                - Legal procedures and regulatory requirements
                - Sector analysis and market insights
                
                Start by typing your investment question below or choose from the examples.
                """)
            
            # Query input section
            st.markdown("### 💭 Ask Your Investment Question")
            
            # Enhanced example questions
            example_questions = [
                "I have 50 million MAD to invest in automotive manufacturing. Which region offers the best opportunities and what are the setup costs?",
                "What are the complete steps and timeline to establish a textile export company in Morocco?",
                "Compare investment incentives and infrastructure between Casablanca and Tangier for a logistics company",
                "What financing options and government support are available for renewable energy projects over 100 million MAD?",
                "I want to open a food processing plant. What permits do I need and what are the regulatory timelines?",
                "Analyze the tech startup ecosystem in Rabat vs Casablanca for a fintech company",
                "What are the labor costs and workforce quality in different regions for aerospace manufacturing?"
            ]
            
            selected_example = st.selectbox(
                "💡 Choose an example question or write your own:",
                ["Select an example..."] + example_questions,
                key="example_selector"
            )
            
            # Auto-fill example
            if selected_example != "Select an example...":
                user_input = st.text_area(
                    "📝 Your investment question:",
                    value=selected_example,
                    height=120,
                    placeholder="Ask about regions, costs, regulations, incentives, or any investment topic...",
                    key="user_input"
                )
            else:
                user_input = st.text_area(
                    "📝 Your investment question:",
                    height=120,
                    placeholder="Ask about regions, costs, regulations, incentives, or any investment topic...",
                    key="user_input"
                )
            
            # Send button
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                send_button = st.button("🚀 Get Expert Analysis", type="primary", use_container_width=True)
            with col_btn2:
                if st.button("🎲 Ask Random Question", use_container_width=True):
                    import random
                    random_question = random.choice(example_questions)
                    st.session_state.user_input = random_question
                    st.rerun()
            
            if send_button:
                if user_input.strip():
                    self.process_user_query(user_input.strip())
                else:
                    st.warning("⚠️ Please enter a question about investing in Morocco.")
        
        with col2:
            # Enhanced sidebar info
            st.markdown("### 📈 Morocco at a Glance")
            st.info("""
            **🌟 Investment Highlights:**
            - 🌍 Strategic Africa-Europe gateway
            - 💰 Competitive operational costs
            - 🚗 World-class automotive sector
            - ⚓ Major ports (Tanger Med, Casablanca)
            - 💻 Growing digital ecosystem
            - 🏭 Strong manufacturing base
            """)
            
            st.markdown("### 💡 Pro Tips for Better Results")
            st.success("""
            **🎯 For More Accurate Analysis:**
            - Specify your investment amount
            - Mention your business sector
            - Include preferred regions
            - State your timeline requirements
            - Ask about specific challenges
            """)
            
            # Session analytics
            if st.session_state.queries_count > 0:
                st.markdown("### 📊 Session Analytics")
                self.display_session_analytics()
    
    def process_user_query(self, user_query: str):
        """Enhanced query processing with real-time feedback"""
        
        # Add user message immediately
        user_message = {
            "type": "user",
            "content": user_query,
            "timestamp": datetime.now()
        }
        st.session_state.conversation_history.append(user_message)
        
        # Show processing animation
        with st.container():
            st.markdown("""
            <div class="processing-animation">
                <span>🤔</span>
                <span>Analyzing your query and routing to expert agents...</span>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            steps = [
                "🔍 Analyzing query intent...",
                "🎯 Routing to appropriate experts...",
                "💼 Consulting regional expert...",
                "📋 Checking regulatory requirements...",
                "💰 Calculating financial implications...",
                "🔄 Synthesizing expert responses...",
                "✅ Preparing comprehensive analysis..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.3)  # Simulate processing time
        
        # Send query to backend
        result = self.send_query(user_query)
        
        # Clear processing indicators
        progress_bar.empty()
        status_text.empty()
        
        if "error" in result:
            st.error(f"❌ {result['error']}")
            
            # Add error message to conversation
            error_message = {
                "type": "assistant",
                "content": f"I apologize, but I encountered an error: {result['error']}. Please check if the backend service is running and try again.",
                "timestamp": datetime.now(),
                "metadata": {"error": True}
            }
            st.session_state.conversation_history.append(error_message)
        else:
            # Process successful response
            response_text = result.get('response', 'No response received')
            
            # Update session statistics
            st.session_state.queries_count += 1
            st.session_state.total_tokens_used += result.get('total_tokens', 0)
            
            # Track agents used
            agents_used = result.get('agents_used', [])
            for agent in agents_used:
                st.session_state.agents_used.add(agent)
            
            # Add assistant response to conversation
            assistant_message = {
                "type": "assistant",
                "content": response_text,
                "timestamp": datetime.now(),
                "metadata": {
                    "agents_used": agents_used,
                    "routing_reasoning": result.get('routing_reasoning', ''),
                    "total_tokens": result.get('total_tokens', 0),
                    "tools_used": result.get('tools_used', []),
                    "context_documents": result.get('context_documents', 0)
                }
            }
            st.session_state.conversation_history.append(assistant_message)
            
            # Show success message
            st.success("✅ Analysis complete! Check the response above.")
        
        # Rerun to update the interface
        st.rerun()

# Main application execution
def main():
    chat_app = MoroccoInvestmentChat()
    chat_app.display_main_interface()

if __name__ == "__main__":
    main()