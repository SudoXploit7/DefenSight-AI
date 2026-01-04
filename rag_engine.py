# rag_engine.py (OPTIMIZED GROQ VERSION - FIXED)
# -------------------------------------------------------------
# DefenSight AI RAG Engine - Optimized for Groq rate limits
# -------------------------------------------------------------

import tiktoken
import time
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from groq import Groq
from collections import Counter

# === SETTINGS ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Get your key from https://console.groq.com/")
GROQ_MODEL = "llama-3.3-70b-versatile"  # ✅ Better model with 128k context window

VECTOR_DB_PATH = "./DefenSight AI_db"
EMBED_MODEL = "multi-qa-mpnet-base-dot-v1"

# ✅ Optimized for Groq free tier (8000 TPM limit)
MAX_CONTEXT_TOKENS = 6000  # Leave room for prompt + response
TOP_K = 40
MAX_OUTPUT_TOKENS = 4000

# === INIT ===
print("🔍 Initializing DefenSight AI RAG Engine (Optimized Mode)...")

model = SentenceTransformer(EMBED_MODEL)
client = PersistentClient(path=VECTOR_DB_PATH)

# Don't cache collection at module level - get it fresh each time
groq_client = Groq(api_key=GROQ_API_KEY)

def get_collection():
    """Get or create collection - always fresh reference"""
    return client.get_or_create_collection("defensight_ai")

# Initial load for startup message
initial_collection = get_collection()
print(f"✅ Loaded: {initial_collection.count()} documents in vector DB")

# === Token Counter ===
def count_tokens(text):
    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(text))
    except:
        # Fallback estimation
        return len(text) // 4


# === Optimized Context Builder ===
def build_context(query, top_k=TOP_K, max_tokens=MAX_CONTEXT_TOKENS):
    """
    Smart context builder that:
    1. Retrieves relevant chunks
    2. Diversifies by type and source
    3. Respects strict token limits
    4. Prioritizes high-quality content
    """
    # Get fresh collection reference
    collection = get_collection()
    
    embedding = model.encode([query])[0]
    
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
    except Exception as e:
        print(f"❌ Query error: {e}")
        return ""

    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    # Categorize by log type
    categorized = {
        "ids": [],
        "config": [],
        "compliance": [],
        "log": [],
        "cert": [],
        "traffic": [],
        "other": []
    }
    
    sources = set()
    token_count = 0
    
    # Reserve tokens for metadata header
    header = "=== SECURITY DATA CONTEXT ===\n\n"
    token_count += count_tokens(header)
    
    # Process chunks with strict token limit
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
            
        meta = metadatas[i] if i < len(metadatas) else {}
        log_type = meta.get("type", "other")
        source = meta.get("source_file", "unknown")
        
        sources.add(source)
        
        # Format chunk with metadata
        formatted_chunk = f"[{log_type.upper()}] {chunk}"
        chunk_tokens = count_tokens(formatted_chunk)
        
        # Stop if we would exceed limit
        if token_count + chunk_tokens > max_tokens:
            break
            
        # Add to category
        if log_type in categorized:
            categorized[log_type].append(formatted_chunk)
        else:
            categorized["other"].append(formatted_chunk)
            
        token_count += chunk_tokens
    
    # Build final context
    context_parts = [header]
    
    # Add categorized content (prioritize important types)
    priority_order = ["ids", "config", "compliance", "cert", "traffic", "log", "other"]
    
    for section in priority_order:
        if categorized[section]:
            context_parts.append(f"--- {section.upper()} ---")
            context_parts.extend(categorized[section][:10])  # Limit per section
            context_parts.append("")
    
    final_context = "\n".join(context_parts).strip()
    actual_tokens = count_tokens(final_context)
    
    print(f"📊 Context: {actual_tokens} tokens, {len(sources)} sources, {sum(len(v) for v in categorized.values())} chunks")
    
    return final_context


# === Smart Groq Wrapper with Rate Limit Handling ===
def ask_groq(messages, max_retries=3, delay=2):
    """
    Smart API wrapper that:
    - Handles rate limits gracefully
    - Implements exponential backoff
    - Validates token counts before sending
    """
    # Calculate total tokens in request
    total_prompt_tokens = sum(count_tokens(m["content"]) for m in messages)
    
    print(f"📤 Request: ~{total_prompt_tokens} prompt tokens")
    
    # If too large, truncate the context
    if total_prompt_tokens > 7000:  # Safety margin
        print("⚠️  Request too large, truncating context...")
        
        # Find and truncate the user message with context
        for msg in messages:
            if msg["role"] == "user" and "Context" in msg["content"]:
                parts = msg["content"].split("===")
                if len(parts) >= 3:
                    # Keep question, reduce context
                    context_part = parts[1]
                    question_part = parts[2] if len(parts) > 2 else ""
                    
                    # Truncate context to fit
                    max_context = 4000  # tokens
                    truncated = context_part[:max_context * 4]  # rough char estimate
                    
                    msg["content"] = f"{parts[0]}===\n{truncated}\n\n===\n{question_part}"
                break
    
    for attempt in range(max_retries):
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=MAX_OUTPUT_TOKENS
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if rate limit error
            if "rate_limit" in error_msg.lower() or "413" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏳ Rate limited, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print("❌ Rate limit exceeded after retries")
                    return "**Error**: Rate limit exceeded. Please try again in a moment."
            
            # Other errors
            if attempt < max_retries - 1:
                print(f"⚠️  Error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                time.sleep(delay)
                continue
            else:
                raise
    
    return "**Error**: Failed to get response after multiple attempts."


# === Optimized Query Engine ===
def query_with_rag(user_query):
    """
    Single-query RAG optimized for token limits
    """
    context = build_context(user_query, top_k=TOP_K, max_tokens=MAX_CONTEXT_TOKENS)
    
    if not context or len(context) < 100:
        return (
            "⚠️ **Insufficient Context**\n\n"
            "No relevant data found in the database for this query. "
            "Please ensure logs have been uploaded and indexed."
        )
    
    system_prompt = """You are DefenSight AI, an expert security analyst. Provide comprehensive, detailed analysis with:
- Specific evidence (IPs, timestamps, log entries)
- Technical explanations
- Security implications
- Actionable recommendations

Use the context strictly. If insufficient, state what's needed."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""===CONTEXT===\n{context}\n\n===QUESTION===\n{user_query}\n\nProvide detailed analysis with specific evidence and recommendations."""
        }
    ]
    
    return ask_groq(messages)


# === Optimized Report Generation (Single-Pass) ===
def generate_summary(mode="technical"):
    """
    Optimized summary generation - single query instead of multi-query
    """
    print(f"📋 Generating {mode} summary...")
    
    # Single optimized query based on mode
    if mode == "technical":
        query = "Analyze all security logs, configurations, IDS alerts, certificates, and threats for technical security report"
    else:
        query = "Summarize top security risks, threats, compliance issues, and critical findings for executive review"
    
    # Build context with higher limit for reports
    context = build_context(query, top_k=50, max_tokens=4500)
    
    if not context or len(context) < 200:
        return f"**No data available** to generate {mode} summary. Please upload and index security logs first."
    
    if mode == "technical":
        prompt = """Generate a comprehensive TECHNICAL SECURITY REPORT with these sections:



## 1. Threat Analysis
- Active threats and attack patterns
- IDS/IPS alerts with severity
- Attack sources and techniques
- Timeline of significant events

## 2. Network Security
- Traffic patterns and anomalies
- Suspicious connections
- Protocol analysis
- Port scanning activities

## 3. Configuration Review
- Firewall rules analysis
- Misconfigurations
- Compliance gaps
- Policy violations

## 4. Certificate & Encryption
- SSL/TLS status
- Certificate issues
- Encryption weaknesses

## 5. Risk Assessment
- Critical vulnerabilities
- Exploitable weaknesses
- Business impact

## 6. Recommendations
- Immediate actions (Priority 1)
- Short-term fixes (Priority 2)
- Long-term improvements (Priority 3)

Include specific IPs, ports, timestamps, and evidence from the logs."""

    else:  # executive
        prompt = """Generate a concise EXECUTIVE SUMMARY for C-level leadership:

## Security Posture
Current security health and key metrics

## Critical Findings
Top 3-5 most critical issues and business impact

## Threat Summary
Active threats and attack attempts

## Compliance Status
Regulatory gaps and audit findings

## Recommendations
Immediate actions, resources needed, timeline, and ROI

Use clear, non-technical language. Focus on business risk and decisions."""

    messages = [
        {
            "role": "system",
            "content": "You are a senior security analyst. Be thorough, specific, and actionable. Use evidence from the provided context."
        },
        {
            "role": "user",
            "content": f"""===SECURITY DATA===\n{context}\n\n===TASK===\n{prompt}\n\nGenerate a complete report with all sections. Include specific findings and evidence."""
        }
    ]
    
    return ask_groq(messages, max_retries=3, delay=3)


# === Similar Events Search ===
def find_similar_events(event_description, top_k=15):
    """Find similar security events for threat hunting"""
    context = build_context(event_description, top_k=top_k, max_tokens=3000)
    
    messages = [
        {
            "role": "system",
            "content": "You are a threat hunting expert analyzing security patterns."
        },
        {
            "role": "user",
            "content": f"""Find events similar to: "{event_description}"

Related Events:
{context}

Analyze:
1. Similar events found
2. Common patterns
3. Security implications
4. Investigation steps"""
        }
    ]
    
    return ask_groq(messages)


# === Database Statistics ===
def get_db_stats():
    """Get statistics about indexed data"""
    collection = get_collection()
    total_docs = collection.count()
    
    if total_docs > 0:
        sample = collection.get(limit=min(1000, total_docs))
        metadatas = sample.get("metadatas", [])
        
        types = Counter(m.get("type", "unknown") for m in metadatas)
        sources = Counter(m.get("source_file", "unknown") for m in metadatas)
        
        return {
            "total_documents": total_docs,
            "log_types": dict(types),
            "sources": dict(sources),
            "embedding_dimension": model.get_sentence_embedding_dimension()
        }
    
    return {"total_documents": 0}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DefenSight AI RAG Engine - Database Stats")
    print("="*60)
    
    stats = get_db_stats()
    print(f"\n📊 Database Statistics:")
    print(f"   Total Documents: {stats.get('total_documents', 0)}")
    print(f"   Embedding Dimensions: {stats.get('embedding_dimension', 'N/A')}")
    
    if stats.get('log_types'):
        print(f"\n📋 Log Types:")
        for log_type, count in stats['log_types'].items():
            print(f"   - {log_type}: {count}")
    
    if stats.get('sources'):
        print(f"\n📁 Top Sources:")
        for source, count in list(stats['sources'].items())[:10]:
            print(f"   - {source}: {count}")
    
    print("\n✅ Ready for queries!")
    print("="*60)