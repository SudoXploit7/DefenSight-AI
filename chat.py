#!/usr/bin/env python3
"""
DefenSight AI_chat_groq.py - OPTIMIZED VERSION (FIXED)
RAG chat loop with enhanced context retrieval and better UX
"""

import os
import sys
import time
import tiktoken
from datetime import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from groq import Groq

# === SETTINGS ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not found in environment")
    print("   Set it with: export GROQ_API_KEY='your-key-here'")
    sys.exit(1)

# Model settings - optimized for quality and speed
GROQ_MODEL = "llama-3.3-70b-versatile"  # ✅ Better quality, 128k context
EMBED_MODEL = "multi-qa-mpnet-base-dot-v1"
VECTOR_DB_PATH = "./DefenSight AI_db"

# Context settings - balanced for free tier
MAX_CONTEXT_TOKENS = 4500  # Safe margin for free tier
TOP_K = 30  # More chunks for better coverage
MAX_COMPLETION_TOKENS = 2000
TEMPERATURE = 0.3  # Slightly higher for more detailed responses

# === INIT ===
print("🔥 Initializing DefenSight AI Chat (Optimized)...")
print(f"   Model: {GROQ_MODEL}")
print(f"   Embeddings: {EMBED_MODEL}")

try:
    model = SentenceTransformer(EMBED_MODEL)
    print(f"✅ Loaded embedding model ({model.get_sentence_embedding_dimension()}D vectors)")
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    sys.exit(1)

try:
    client = PersistentClient(path=VECTOR_DB_PATH)
    
    # Don't cache collection at module level
    def get_collection():
        """Get or create collection - always fresh reference"""
        return client.get_or_create_collection("defensight_ai")
    
    collection = get_collection()
    doc_count = collection.count()
    print(f"✅ Connected to ChromaDB: {doc_count:,} documents indexed")
    
    if doc_count == 0:
        print("⚠️  WARNING: No documents in database! Upload logs first.")
except Exception as e:
    print(f"❌ Failed to connect to ChromaDB: {e}")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_API_KEY)

# === TOKEN COUNTER ===
def count_tokens(text):
    """Accurate token counting"""
    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(text))
    except Exception:
        # Fallback estimation
        return len(text) // 4


# === ENHANCED RAG RETRIEVAL ===
def retrieve_relevant_context(query, top_k=TOP_K, max_tokens=MAX_CONTEXT_TOKENS):
    """
    Enhanced context retrieval with:
    - Smart categorization by log type
    - Source diversity
    - Token budget management
    - Quality metadata
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
        print(f"❌ Vector DB query error: {e}")
        return "", {}

    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    # Categorize by log type
    categorized = {
        "ids": [],
        "config": [],
        "compliance": [],
        "cert": [],
        "traffic": [],
        "log": [],
        "other": []
    }
    
    sources = Counter()
    token_total = 0
    chunks_added = 0
    
    for i, chunk in enumerate(chunks):
        if not chunk or token_total >= max_tokens:
            break
        
        meta = metadatas[i] if i < len(metadatas) else {}
        log_type = meta.get("type", "other")
        source = meta.get("source_file", "unknown")
        timestamp = meta.get("timestamp", "")
        
        # Format chunk with rich metadata
        formatted = f"[{log_type.upper()}|{source}] {chunk}"
        
        chunk_tokens = count_tokens(formatted)
        if token_total + chunk_tokens > max_tokens:
            break
        
        # Add to category
        category = log_type if log_type in categorized else "other"
        categorized[category].append(formatted)
        
        sources[source] += 1
        token_total += chunk_tokens
        chunks_added += 1
    
    # Build structured context
    context_parts = []
    
    # Prioritize important log types
    priority = ["ids", "config", "compliance", "cert", "traffic", "log", "other"]
    
    for cat in priority:
        if categorized[cat]:
            context_parts.append(f"=== {cat.upper()} LOGS ===")
            # Limit per category to ensure diversity
            context_parts.extend(categorized[cat][:8])
            context_parts.append("")
    
    final_context = "\n".join(context_parts).strip()
    
    # Stats for user feedback
    stats = {
        "chunks": chunks_added,
        "tokens": token_total,
        "sources": len(sources),
        "source_list": dict(sources.most_common(5)),
        "types": {k: len(v) for k, v in categorized.items() if v}
    }
    
    return final_context, stats


# === SMART GROQ WRAPPER ===
def ask_groq(messages, retries=3):
    """
    Enhanced Groq API wrapper with:
    - Rate limit handling
    - Exponential backoff
    - Token validation
    - Better error messages
    """
    # Pre-flight token check
    total_tokens = sum(count_tokens(m["content"]) for m in messages)
    
    if total_tokens > 7500:
        print(f"⚠️  Large request (~{total_tokens} tokens), may hit rate limit...")
    
    for attempt in range(retries):
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS
            )
            
            response = completion.choices[0].message.content
            
            # Usage stats
            if hasattr(completion, 'usage'):
                usage = completion.usage
                print(f"📊 Tokens used: {usage.total_tokens} (prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})")
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Rate limit handling
            if "rate_limit" in error_msg or "413" in error_msg:
                if attempt < retries - 1:
                    wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⏳ Rate limited, waiting {wait}s... (attempt {attempt + 1}/{retries})")
                    time.sleep(wait)
                    continue
                else:
                    return "❌ **Rate Limit Error**: Too many requests. Please wait a moment and try again."
            
            # Other errors
            if attempt < retries - 1:
                print(f"⚠️  Error (attempt {attempt + 1}/{retries}): {str(e)[:100]}")
                time.sleep(1)
                continue
            else:
                return f"❌ **Error**: {str(e)[:200]}"
    
    return "❌ Failed after multiple retries."


# === HELPER COMMANDS ===
def show_help():
    """Display available commands"""
    print("\n" + "="*60)
    print("📚 Available Commands:")
    print("="*60)
    print("  exit, quit       - Exit the chat")
    print("  help, ?          - Show this help")
    print("  stats            - Show database statistics")
    print("  clear            - Clear conversation history")
    print("  debug            - Toggle debug mode")
    print("="*60 + "\n")


def show_stats():
    """Display database statistics"""
    collection = get_collection()
    total = collection.count()
    
    print("\n" + "="*60)
    print("📊 Database Statistics")
    print("="*60)
    print(f"  Total Documents: {total:,}")
    
    if total > 0:
        # Sample to get type distribution
        sample = collection.get(limit=min(1000, total))
        metas = sample.get("metadatas", [])
        
        types = Counter(m.get("type", "unknown") for m in metas)
        sources = Counter(m.get("source_file", "unknown") for m in metas)
        
        print(f"\n  📋 Log Types:")
        for log_type, count in types.most_common():
            percentage = (count / len(metas)) * 100
            print(f"     • {log_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n  📁 Top Sources:")
        for source, count in sources.most_common(5):
            print(f"     • {source}: {count}")
    
    print("="*60 + "\n")


# === MAIN CHAT LOOP ===
def start_chat():
    """Enhanced chat loop with better UX"""
    print("\n" + "="*60)
    print("🤖 DefenSight AI Chat - Type 'help' for commands")
    print("="*60 + "\n")
    
    system_prompt = """You are DefenSight AI, an expert security analyst specializing in:
• Firewall configuration and log analysis
• IDS/IPS alert investigation and threat hunting
• Network security and traffic analysis
• Compliance and security policy review
• Certificate and encryption analysis

Provide comprehensive answers with:
- Specific evidence (IPs, timestamps, log entries)
- Technical explanations and context
- Security implications and risk assessment
- Actionable recommendations

Use ONLY the provided context. If insufficient, clearly state what additional data is needed."""

    messages = [{"role": "system", "content": system_prompt}]
    history_limit = 20
    debug_mode = False
    
    try:
        while True:
            # Get user input
            try:
                user_input = input("🧑 You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() in ["help", "?"]:
                show_help()
                continue
            
            elif user_input.lower() == "stats":
                show_stats()
                continue
            
            elif user_input.lower() == "clear":
                messages = [messages[0]]  # Keep only system prompt
                print("✅ Conversation history cleared.\n")
                continue
            
            elif user_input.lower() == "debug":
                debug_mode = not debug_mode
                status = "enabled" if debug_mode else "disabled"
                print(f"🔧 Debug mode {status}\n")
                continue
            
            # Process query
            print("🔍 Searching knowledge base...")
            start_time = time.time()
            
            context, stats = retrieve_relevant_context(user_input, top_k=TOP_K)
            
            retrieval_time = time.time() - start_time
            
            # Show retrieval stats
            if context:
                print(f"✅ Found {stats['chunks']} relevant chunks ({stats['tokens']} tokens)")
                if debug_mode:
                    print(f"   Sources: {list(stats['source_list'].keys())}")
                    print(f"   Types: {stats['types']}")
                    print(f"   Retrieval time: {retrieval_time:.2f}s")
            else:
                print("⚠️  No relevant context found in database")
            
            # Build message
            if context:
                user_content = f"""Context from security logs:
{context}

Question: {user_input}

Provide a detailed analysis with specific evidence and recommendations."""
            else:
                user_content = f"""Question: {user_input}

Note: No relevant context found in the database. Please explain what information would be needed to answer this query."""
            
            messages.append({"role": "user", "content": user_content})
            
            # Maintain history limit
            if len(messages) > history_limit:
                messages = [messages[0]] + messages[-(history_limit-1):]
            
            # Get LLM response
            print("⏳ Generating response...\n")
            
            llm_start = time.time()
            assistant_reply = ask_groq(messages)
            llm_time = time.time() - llm_start
            
            if debug_mode:
                print(f"\n⏱️  LLM response time: {llm_time:.2f}s")
            
            # Display response
            print(f"\n🤖 DefenSight AI:\n")
            print(assistant_reply)
            print("\n" + "-"*60 + "\n")
            
            # Add to history
            messages.append({"role": "assistant", "content": assistant_reply})
            
            # Brief pause to respect rate limits
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


# === ENTRY POINT ===
if __name__ == "__main__":
    # Quick DB check
    collection = get_collection()
    if collection.count() == 0:
        print("\n⚠️  DATABASE IS EMPTY!")
        print("   Please upload and index logs first using:")
        print("   • Web UI: python gui_app.py")
        print("   • CLI: python live_ingest.py --reindex")
        print()
        
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    start_chat()