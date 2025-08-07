import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.llm import get_gemini_model
from models.embeddings import HREmbedder
from utils.retrieval import hr_rag_query
from utils.anonymize import clean_text
from utils.file_processing import extract_text
import chromadb
from chromadb.utils import embedding_functions
from config.chroma_client import init_chroma
from datetime import datetime
import pandas as pd
from google.api_core import exceptions
from io import StringIO
from docx import Document
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
import numpy as np
from wordcloud import WordCloud
import tempfile
import base64
from faker import Faker  # For generating sample data in demo mode

# Configuration
st.set_page_config(
    page_title="AttriBot HR Analytics Suite",
    page_icon="📊",
    #layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
HR_SYSTEM_PROMPT = """
You are AttriBot, an intelligent HR retention assistant and a data analyst focused on company well-being and growth. Follow these guidelines strictly:

1. First try to answer using the provided HR documents.
2. If information is missing or outdated, you may perform a web search by including:
   <web_search>search query</web_search>
3. For HR policy questions, always verify with latest web sources.
4. When using web results, always cite sources with links.


5.  **Analyze and Explain:**
    * Begin with a clear, bulleted analysis of the data. Explain what the data shows and provide key insights based *only* on the visualization you are about to generate.
    * Your primary goal is to turn data into a story. Explain what the chart means in a practical business context.

6.  **Critique and Suggest:**
    * After your analysis, you **must** provide a "Data Quality & Suggestions" section.
    * Critically evaluate the analysis. If the data is limited (e.g., small sample size, no time-series data), state it clearly.
    * Suggest specific improvements. For example: "For a more accurate trend analysis, we would need data from multiple years. The current view is just a snapshot." or "To understand the root cause, we should segment this data by department."
    * also suggest improvements for company policies or practices based on the data, such as "To improve retention, consider implementing exit interviews to gather more qualitative insights."

7.  *Visualize the Data:**
    * Following your text, you **must** generate a Python code snippet to create a visualization if the query involves trends, statistics, or comparisons.
    * **This is critical:** Your final output must HIDE the Python code from the user. The user should only see your analysis, your suggestions, and the graph itself.
    * **IMPORTANT**: Wrap the Python code in a single markdown block starting with ```python.
    * **CRITICAL**: In your Python code, use libraries like matplotlib or seaborn, but **NEVER call `plt.show()`**. Streamlit will handle rendering the plot.


8.  **Maintain Standards:**
    * Base all analysis and citations on the provided ChromaDB data only.
    * Maintain a professional and empathetic tone suitable for HR.
"""

# Initialize Faker for demo data
fake = Faker()

# Enhanced utility functions
def calculate_retention_risk_score(text):
    """Calculate a retention risk score based on sentiment and keywords"""
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    
    # Keywords that indicate retention risk
    risk_keywords = ['leave', 'quit', 'unhappy', 'frustrated', 'better offer', 
                    'burnout', 'stress', 'toxic', 'culture', 'manager']
    
    keyword_count = sum(1 for word in risk_keywords if word.lower() in text.lower())
    
    # Normalize score between 0-100
    score = max(0, min(100, 50 + (sentiment * 20) + (keyword_count * 5)))
    return round(score, 1)

def generate_word_cloud(text):
    """Generate a word cloud from text"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_download_link(data, filename, file_type):
    """Create a download link for data"""
    if file_type == 'csv':
        data = data.to_csv(index=False)
        b64 = base64.b64encode(data.encode()).decode()
    elif file_type == 'json':
        b64 = base64.b64encode(data.encode()).decode()
    else:
        raise ValueError("Unsupported file type")
    
    href = f'<a href="data:file/{file_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Enhanced file processing
def extract_text(file):
    """Enhanced text extraction with error handling"""
    ext = os.path.splitext(file.name)[-1].lower()
    
    try:
       
        pass  # Placeholder for code that may raise an exception
        if ext == ".pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

        elif ext == ".docx":
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])

        elif ext == ".txt":
            return str(file.read(), "utf-8")

        elif ext == ".csv":
            df = pd.read_csv(file)
            return df.to_string()

        else:
            st.warning(f"Unsupported file format: {ext}")
            return ""
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return ""

# Enhanced clean_text function
def clean_text(text):
    """Enhanced PII cleaning with regex patterns"""
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove phone numbers
    text = re.sub(r'(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}\b', '[PHONE]', text)
    
    # Remove credit card numbers
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CC]', text)
    
    # Remove SSN-like patterns
    text = re.sub(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', '[SSN]', text)
    
    # Basic cleanup
    text = text.replace("\n", " ").replace("\r", " ").strip()
    
    return text

# Enhanced HR Upload Page
def hr_upload_page():
    st.title("🧑💼 HR Data Hub")
    
    # Init ChromaDB
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="hr_docs")

    # Upload section with enhanced options
    with st.expander("📁 Advanced Data Upload", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload HR documents (CSV, PDF, DOCX, TXT)",
                type=["csv", "pdf", "docx", "txt"],
                accept_multiple_files=True,
                help="Upload multiple files at once for batch processing"
            )
            
        with col2:
            data_source = st.selectbox(
                "Data Source",
                options=["HRIS", "ATS", "Survey", "Exit Interview", "Performance", "Other"],
                index=0,
                help="Select the primary source of this data"
            )
            
            department = st.text_input(
                "Department (optional)",
                help="Tag data by department for better segmentation"
            )

    # Enhanced tagging system
    if uploaded_files:
        st.subheader("🔖 Data Categorization")
        with st.form("file_tagging_form"):
            cols = st.columns(3)
            file_tags = {}
            
            for i, file in enumerate(uploaded_files):
                with cols[i % 3]:
                    file_name = file.name
                    tag = st.selectbox(
                        f"Tag for: {file_name}",
                        options=["Auto Detect", "Compensation", "Performance", "Exit", "Engagement", "Diversity", "Other"],
                        key=f"tag_{file_name}",
                        help="Categorize this file for better retrieval"
                    )
                    file_tags[file_name] = tag
            
            # Additional metadata
            confidentiality = st.select_slider(
                "Confidentiality Level",
                options=["Internal", "Confidential", "Restricted"],
                value="Confidential",
                help="Set appropriate access level for this data"
            )
            
            retention_days = st.number_input(
                "Retention Period (days)",
                min_value=30,
                max_value=365*5,
                value=365,
                help="How long should this data be retained?"
            )
            
            if st.form_submit_button("Apply Tags"):
                st.success("Metadata tags applied to all files")

    # Processing logic with progress bar
    if st.button("🚀 Process & Index Data", type="primary"):
        if not uploaded_files:
            st.warning("Please upload files first")
            return
            
        with st.status("Processing files...", expanded=True) as status:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            added_count = 0
            
            for i, file in enumerate(uploaded_files):
                try:
                    file_name = file.name
                    file_ext = os.path.splitext(file_name)[-1].lower()
                    tag = file_tags.get(file_name, "Auto Detect")
                    
                    status.write(f"Processing {file_name}...")
                    
                    if file_ext == ".csv":
                        df = pd.read_csv(file)
                        for idx, row in df.iterrows():
                            doc = row.to_json()
                            doc_id = f"{tag.lower()}_{data_source}_{file_name}_{idx}"
                            metadata = {
                                "source": file_name,
                                "type": tag,
                                "department": department,
                                "data_source": data_source,
                                "confidentiality": confidentiality,
                                "retention_days": retention_days,
                                "upload_time": str(datetime.now())
                            }
                            collection.add(
                                documents=[doc],
                                metadatas=[metadata],
                                ids=[doc_id]
                            )
                            added_count += 1
                    else:
                        text = extract_text(file)
                        cleaned = clean_text(text)
                        
                        # Calculate retention risk score for text documents
                        risk_score = calculate_retention_risk_score(cleaned) if tag in ["Exit", "Engagement"] else None
                        
                        metadata = {
                            "source": file_name,
                            "type": tag,
                            "department": department,
                            "data_source": data_source,
                            "confidentiality": confidentiality,
                            "retention_days": retention_days,
                            "risk_score": risk_score,
                            "upload_time": str(datetime.now())
                        }
                        
                        collection.add(
                            documents=[cleaned],
                            metadatas=[metadata],
                            ids=[f"{tag.lower()}_{data_source}_{file_name}"]
                        )
                        added_count += 1
                    
                    progress_bar.progress((i + 1) / total_files)
                    
                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
                    continue
            
            status.update(label="Processing complete!", state="complete", expanded=False)
        
        st.success(f"✅ Successfully indexed {added_count} documents")
        st.balloons()
        
        # Show summary statistics
        with st.expander("📊 Upload Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Files Processed", len(uploaded_files))
            col2.metric("Total Documents Indexed", added_count)
            col3.metric("Average Risk Score", 
                        f"{risk_score:.1f}" if 'risk_score' in locals() and risk_score else "N/A",
                        delta="-5% from last upload" if 'risk_score' in locals() and risk_score and risk_score < 50 else "+5% from last upload")

# Enhanced chat response function
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st

def get_chat_response(model, messages):
    """Enhanced chat response with better error handling and performance"""
    try:
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif isinstance(msg, AIMessage):
                gemini_messages.append({"role": "model", "parts": [msg.content]})
            elif isinstance(msg, SystemMessage):
                gemini_messages.append({"role": "system", "parts": [msg.content]})
            else:
                st.warning(f"Unknown message type: {type(msg)}")

    except Exception as e:
        st.error(f"Error formatting messages: {str(e)}")
        return "Error preparing the conversation history."

    # Retry logic with exponential backoff
    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = model.generate_content(gemini_messages)

            if not response.candidates:
                return "I'm sorry, the response was blocked due to content policy restrictions."

            finish_reason = response.candidates[0].finish_reason
            if finish_reason.name != "STOP":
                return f"Response incomplete due to: {finish_reason.name}"

            return response.text

        except Exception as e:
            # Specific handling for quota or overload
            if "ResourceExhausted" in str(type(e)):
                if attempt == max_retries - 1:
                    return "I'm currently experiencing high demand. Please try again later."
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                st.error(f"API Error: {str(e)}")
                return "Sorry, I encountered an error processing your request."

    return "Unable to get a response after multiple attempts."

# Enhanced visualization parser
def parse_and_display_visualization(response_text):
    """
    Enhanced visualization parser with better error handling and fallback options
    """
    # Extract all code blocks
    code_blocks = re.findall(r"```python\s*([\s\S]+?)\s*```", response_text)
    
    # Display the text parts
    text_parts = re.split(r"```python\s*[\s\S]+?\s*```", response_text)
    for text in text_parts:
        if text.strip():
            st.markdown(text.strip())
    
    # Execute each code block
    visualization_generated = False
    for code in code_blocks:
        try:
            # Create a clean namespace for execution
            exec_namespace = {
                'plt': plt,
                'sns': sns,
                'pd': pd,
                'np': np,
                'px': px,
                'st': st,
                'os': os
            }
            
            # Check for file operations and provide dummy data if needed
            if "pd.read_csv" in code and not os.path.exists('HR_Analytics.csv'):
                st.warning("Using sample data since 'HR_Analytics.csv' was not found")
                exec_namespace['df'] = pd.DataFrame({
                    'department': ['IT', 'HR', 'Finance', 'IT', 'HR'],
                    'age': [25, 30, 45, 28, 35],
                    'gender': ['M', 'F', 'M', 'F', 'M'],
                    'attrition': [1, 0, 1, 0, 1]
                })
            
            # Execute the code
            exec(code, exec_namespace)
            
            # Handle different visualization outputs
            if 'plt' in exec_namespace:
                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig)
                    visualization_generated = True
                plt.close()
            elif 'fig' in exec_namespace:
                fig = exec_namespace['fig']
                if hasattr(fig, '__plotly_restyle__'):
                    st.plotly_chart(fig)
                    visualization_generated = True
                elif hasattr(fig, 'savefig'):
                    st.pyplot(fig)
                    visualization_generated = True
            elif 'ax' in exec_namespace:
                fig = exec_namespace['ax'].get_figure()
                st.pyplot(fig)
                visualization_generated = True
                plt.close()
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            with st.expander("View problematic code"):
                st.code(code, language="python")
    
    return visualization_generated
def chat_page():
    """Main chat interface with web search and guaranteed visualizations"""
    st.title("🧑‍💼 AttriBot HR Analytics Suite")
    
    # Initialize components
    chat_model = get_gemini_model()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Controls")
        response_mode = st.radio(
            "Response Style",
            ["Detailed", "Concise", "Executive Summary"],
            index=0
        )
        
        visualization_style = st.selectbox(
            "Visualization Style",
            ["Matplotlib", "Seaborn", "Plotly"],
            index=1
        )
        
        enable_web_search = st.toggle("Enable Web Search", True,
                                    help="Allow real-time web searches when needed")
        
        if st.button("Clear Conversation"):
            st.session_state.messages = [{"role": "system", "content": HR_SYSTEM_PROMPT}]
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": HR_SYSTEM_PROMPT}]
    
    # Display conversation history
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                parse_and_display_visualization(msg["content"])
    
    # Main chat interface
    if prompt := st.chat_input("Ask about attrition patterns..."):
        # Add user message to chat
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Perform RAG query
        results = hr_rag_query(prompt)
        
        # Prepare context with metadata
        context = "Relevant documents:\n"
        has_data = bool(results['documents'])
        
        if has_data:
            for doc, metadata in zip(results['documents'], results['metadatas']):
                context += f"\n- Source: {metadata.get('source', 'Unknown')}"
                context += f"\n  Department: {metadata.get('department', 'N/A')}"
                context += f"\n  Content: {str(doc)[:500]}...\n"
        else:
            context += "\nNo relevant documents found in database.\n"
        
        # Format final prompt
        final_prompt = f"{HR_SYSTEM_PROMPT}\n\nUser Query: {prompt}\n\n{context}"
        
        if response_mode == "Concise":
            final_prompt += "\n\nPlease respond concisely with 3-5 bullet points."
        elif response_mode == "Executive Summary":
            final_prompt += "\n\nProvide an executive summary with key insights and recommendations."
        
        # FORCE visualization when data exists
        if has_data:
            final_prompt += "\n\nIMPORTANT: You MUST include a Python code visualization."
            final_prompt += f"\nUse {visualization_style} for the visualization."
            final_prompt += "\nPreferred visualization types:"
            final_prompt += "\n1. Attrition by Department Bar Chart"
            final_prompt += "\n2. Monthly Income vs. Attrition Box Plot"
            final_prompt += "\n3. Age Distribution Histogram"
            final_prompt += "\n4. Job Satisfaction Stacked Bar"
            final_prompt += "\n\nInclude clear titles and axis labels."
        
        # Get and display response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = get_chat_response(
                        chat_model,
                        [HumanMessage(content=final_prompt)]
                    )
                    
                    # Handle web search requests
                    if enable_web_search and "<web_search>" in response:
                        search_query = re.search(r'<web_search>(.*?)</web_search>', response).group(1)
                        with st.spinner(f"Searching web for: {search_query}"):
                            web_results = perform_web_search(search_query)
                            
                            if web_results:
                                response = response.replace(
                                    f"<web_search>{search_query}</web_search>",
                                    "\n\n**Web Search Results:**\n" + 
                                    "\n".join([f"- [{res['title']}]({res['url']})\n  {res['content'][:200]}..." 
                                              for res in web_results])
                                )
                            else:
                                response = response.replace(
                                    f"<web_search>{search_query}</web_search>",
                                    "\n\n⚠️ No web results found for this query"
                                )
                    
                    # Handle visualizations
                    if "```python" in response:
                        viz_success = parse_and_display_visualization(response)
                        
                        if not viz_success and has_data:
                            generate_standard_attrition_visualizations(results, visualization_style)
                    else:
                        st.markdown(response)
                        if has_data and not response_mode == "Concise":
                            generate_standard_attrition_visualizations(results, visualization_style)
                    
                    if results['sources']:
                        st.caption(f"Sources: {', '.join(set(results['sources']))}")
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    if has_data:
                        generate_standard_attrition_visualizations(results, visualization_style)
        
        st.session_state.messages.append({"role": "assistant", "content": response})


def generate_standard_attrition_visualizations(results, style="Plotly"):
    """Generate standard HR attrition visualizations"""
    try:
        df = pd.DataFrame(results['metadatas'])
        if len(df) == 0:
            return
            
        st.warning("Generating standard visualizations from available data...")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["By Department", "By Age", "By Gender"])
        
        with tab1:
            if 'department' in df.columns:
                st.subheader("Attrition by Department")
                if style == "Plotly":
                    fig = px.bar(df, x='department', color='attrition', 
                                title="Attrition by Department", barmode='group')
                    st.plotly_chart(fig)
                else:
                    plt.figure(figsize=(12, 6))
                    sns.countplot(data=df, x='department', hue='attrition')
                    plt.xticks(rotation=45)
                    plt.title("Attrition by Department")
                    st.pyplot(plt.gcf())
        
        with tab2:
            if 'age' in df.columns:
                st.subheader("Age Distribution")
                if style == "Plotly":
                    fig = px.histogram(df, x='age', color='attrition',
                                     title="Age Distribution by Attrition",
                                     barmode='overlay')
                    st.plotly_chart(fig)
                else:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=df, x='age', hue='attrition', 
                                kde=True, bins=20)
                    plt.title("Age Distribution by Attrition")
                    st.pyplot(plt.gcf())
        
        with tab3:
            if 'gender' in df.columns:
                st.subheader("Attrition by Gender")
                if style == "Plotly":
                    fig = px.pie(df, names='gender', color='attrition',
                               title="Attrition by Gender")
                    st.plotly_chart(fig)
                else:
                    plt.figure(figsize=(8, 6))
                    df.groupby(['gender', 'attrition']).size().unstack().plot(
                        kind='pie', subplots=True, legend=False)
                    plt.title("Attrition by Gender")
                    st.pyplot(plt.gcf())
                    
    except Exception as e:
        st.error(f"Failed to generate standard visualizations: {str(e)}")
        
        
def debug_chroma():
    """Enhanced ChromaDB debug interface with advanced deletion options"""
    st.title("🔍 Database Inspector")
    
    try:
        collection = init_chroma()
        docs = collection.get()
        
        if not docs["ids"]:
            st.warning("ChromaDB is empty. Upload data first.")
            return
            
        st.success(f"Found {len(docs['ids'])} documents")
        
        # Create detailed dataframe with proper null handling
        df = pd.DataFrame({
            "ID": docs["ids"],
            "Type": [m.get("type", "unknown") for m in docs["metadatas"]],
            "Source": [m.get("source", "unknown") for m in docs["metadatas"]],
            "Department": [m.get("department", "unknown") for m in docs["metadatas"]],
            "Risk Score": [float(m.get("risk_score", 0)) for m in docs["metadatas"]],
            "Confidentiality": [m.get("confidentiality", "unknown") for m in docs["metadatas"]],
            "Upload Time": [m.get("upload_time", "") for m in docs["metadatas"]],
            "Content Preview": [txt[:80] + "..." if txt else "empty" for txt in docs["documents"]]
        })

        # Convert Upload Time and handle missing dates
        df['Upload Time'] = pd.to_datetime(df['Upload Time'], errors='coerce')
        df['Year'] = df['Upload Time'].dt.year.fillna('Unknown').astype(str)
        df['Month'] = df['Upload Time'].dt.month_name().fillna('Unknown')
        df['Quarter'] = df['Upload Time'].dt.quarter.fillna('Unknown').astype(str)
        
        # Data Management Section
        with st.expander("🗑️ Document Deletion Tools", expanded=True):
            tab1, tab2, tab3 = st.tabs(["By Source File", "By Date Range", "By Metadata Filters"])
            
            with tab1:
                st.subheader("Delete Entire Source Files")
                source_files = sorted(df['Source'].unique().tolist())
                files_to_delete = st.multiselect(
                    "Select source files to delete completely",
                    options=source_files,
                    help="This will delete ALL documents from the selected source files"
                )
                
                if st.button("Delete Selected Files", type="primary"):
                    if files_to_delete:
                        # Get all IDs from selected sources
                        ids_to_delete = df[df['Source'].isin(files_to_delete)]['ID'].tolist()
                        collection.delete(ids=ids_to_delete)
                        st.success(f"Deleted {len(ids_to_delete)} documents from {len(files_to_delete)} files")
                        st.rerun()
                    else:
                        st.warning("Please select at least one source file")
            
            with tab2:
                st.subheader("Delete by Date Range")
                
                # Get valid date range
                valid_dates = df[df['Upload Time'].notna()]
                if len(valid_dates) > 0:
                    min_date = valid_dates['Upload Time'].min().date()
                    max_date = valid_dates['Upload Time'].max().date()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input(
                            "Start date",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date
                        )
                    with col2:
                        end_date = st.date_input(
                            "End date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date
                        )
                    
                    if start_date > end_date:
                        st.error("End date must be after start date")
                    else:
                        mask = (df['Upload Time'] >= pd.Timestamp(start_date)) & \
                               (df['Upload Time'] <= pd.Timestamp(end_date))
                        date_df = df[mask]
                        
                        st.info(f"Found {len(date_df)} documents in selected range")
                        
                        if st.button(f"Delete {len(date_df)} Documents in Date Range", 
                                   type="primary", disabled=len(date_df)==0):
                            collection.delete(ids=date_df['ID'].tolist())
                            st.success(f"Deleted {len(date_df)} documents")
                            st.rerun()
                else:
                    st.warning("No documents with valid dates found")
            
            with tab3:
                st.subheader("Delete by Metadata Filters")
                st.write("Select criteria to delete matching documents")
                
                col1, col2 = st.columns(2)
                with col1:
                    delete_type = st.selectbox(
                        "Document Type",
                        options=["All Types"] + sorted(df['Type'].unique().tolist()))
                    
                    delete_dept = st.selectbox(
                        "Department",
                        options=["All Departments"] + sorted(df['Department'].unique().tolist()))
                
                with col2:
                    delete_year = st.selectbox(
                        "Year",
                        options=["All Years"] + sorted(df['Year'].unique().tolist()))
                    
                    delete_conf = st.selectbox(
                        "Confidentiality Level",
                        options=["All Levels"] + sorted(df['Confidentiality'].unique().tolist()))
                
                # Build query
                query = {}
                if delete_type != "All Types":
                    query['Type'] = delete_type
                if delete_dept != "All Departments":
                    query['Department'] = delete_dept
                if delete_year != "All Years":
                    query['Year'] = delete_year
                if delete_conf != "All Levels":
                    query['Confidentiality'] = delete_conf
                
                if query:
                    mask = pd.Series(True, index=df.index)
                    for k, v in query.items():
                        mask &= (df[k] == v)
                    
                    filtered_df = df[mask]
                    st.info(f"Found {len(filtered_df)} matching documents")
                    
                    if st.button(f"Delete {len(filtered_df)} Matching Documents", 
                               type="primary", disabled=len(filtered_df)==0):
                        collection.delete(ids=filtered_df['ID'].tolist())
                        st.success(f"Deleted {len(filtered_df)} documents")
                        st.rerun()
                else:
                    st.warning("Select at least one filter criteria")
        
        # Display all documents with filters
        st.subheader("📄 All Documents")
        
        # Filter controls
        with st.expander("🔍 Filter Options", expanded=True):
            cols = st.columns(4)
            
            with cols[0]:
                year_filter = st.multiselect(
                    "Filter by Year",
                    options=sorted(df['Year'].unique()),
                    default=sorted(df['Year'].unique())
                )
            
            with cols[1]:
                type_filter = st.multiselect(
                    "Filter by Type",
                    options=sorted(df['Type'].unique()),
                    default=sorted(df['Type'].unique())
                )
            
            with cols[2]:
                dept_filter = st.multiselect(
                    "Filter by Department",
                    options=sorted(df['Department'].unique()),
                    default=sorted(df['Department'].unique())
                )
            
            with cols[3]:
                conf_filter = st.multiselect(
                    "Filter by Confidentiality",
                    options=sorted(df['Confidentiality'].unique()),
                    default=sorted(df['Confidentiality'].unique())
                )
        
        # Apply filters
        filtered_df = df[
            (df['Year'].isin(year_filter)) &
            (df['Type'].isin(type_filter)) &
            (df['Department'].isin(dept_filter)) &
            (df['Confidentiality'].isin(conf_filter))
        ]
        
        # Display data
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "Content Preview": st.column_config.TextColumn(width="large"),
                "Risk Score": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                )
            },
            hide_index=True
        )
        
        # Export options
        with st.expander("💾 Export Options"):
            st.download_button(
                label="Download Filtered Data as CSV",
                data=filtered_df.to_csv(index=False),
                file_name="chroma_filtered_export.csv",
                mime="text/csv"
            )
            
            st.download_button(
                label="Download Full Collection as CSV",
                data=df.to_csv(index=False),
                file_name="chroma_full_export.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error accessing ChromaDB: {str(e)}")
# Main application
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Chat"
    
    # Sidebar navigation with enhanced UI
    with st.sidebar:
        st.image("/Users/siv3sh/Downloads/NeoStats-white-high-res-3.png", width=150)
        st.title("Navigation")
        
        # Page selection with icons
        st.session_state.page = st.radio(
            "Go to:",
            options=[
                "Chat", 
                "Data Upload", 
                "Database Debug"
            ],
            format_func=lambda x: {
                "Chat": "💬 Chat Analysis",
                "Data Upload": "📤 Data Hub",
                "Database Debug": "🔍 Database Inspector"
            }[x],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Demo mode toggle
        demo_mode = st.toggle("Demo Mode", False, help="Use sample data for testing")
        if demo_mode:
            st.warning("Demo mode using synthetic data")
        
        # System status
        st.divider()
        st.caption("System Status")
        try:
            chroma_status = "✅ Connected" if init_chroma() else "❌ Disconnected"
            st.write(f"ChromaDB: {chroma_status}")
            
            model_status = "✅ Ready" if get_gemini_model() else "❌ Unavailable"
            st.write(f"AI Model: {model_status}")
        except:
            st.error("System components not initialized")
    
    # Page router
    if st.session_state.page == "Chat":
        chat_page()
    elif st.session_state.page == "Data Upload":
        hr_upload_page()
    elif st.session_state.page == "Database Debug":
        debug_chroma()

if __name__ == "__main__":
    main()