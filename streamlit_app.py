import streamlit as st
import requests
import json

# --- Configuration ---
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Mini RAG Q&A", layout="centered")
st.title("🧠 Mini RAG Q&A")
st.markdown("Ask questions about your industrial safety documents!")

# --- Sidebar Configuration ---
st.sidebar.subheader("Configuration")
k_value = st.sidebar.slider("Number of Contexts (k):", min_value=1, max_value=10, value=5)
mode = st.sidebar.radio("Search Mode:", ("hybrid", "baseline"))

# Set default alpha for hybrid mode if alpha slider is removed
alpha = 0.6 if mode == "hybrid" else 0.0 # Alpha is only relevant for hybrid mode, set to 0.0 for baseline (or can be ignored by API)

# --- Q&A Interface ---
query = st.text_input("Your Question:", "What is machine safety?")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching for answers..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"q": query, "k": k_value, "mode": mode, "alpha": alpha}
                )
                response.raise_for_status() # Raise an exception for HTTP errors
                result = response.json()

            if result.get("answer") and result["answer"]["answer"]:
                st.subheader("Answer:")
                st.write(result["answer"]["answer"])
                st.markdown(f"**Sources:** {', '.join(result['answer']['sources'])}")
            else:
                st.info(result.get("abstention_reason", "No direct answer found, but here are some relevant contexts:"))
            
            if result.get("contexts"):
                st.subheader("Retrieved Contexts:")
                for i, context in enumerate(result["contexts"]):
                    st.markdown(f"**Context {i+1} (Score: {context['score']:.2f})** from {context['source_title']} (Chunk {context['chunk_index']}):")
                    st.markdown(context['text'])
            else:
                st.write("No contexts retrieved.")

            st.markdown(f"--- \n _Processing Time: {result.get('processing_time', 0):.2f}s, Reranker Used: {result.get('reranker_used', False)}_ ")

        except requests.exceptions.ConnectionError:
            st.error(f"Error: Could not connect to the API at {API_URL}. Please ensure the FastAPI server is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error during API request: {e}")
        except json.JSONDecodeError:
            st.error("Error: Could not decode JSON response from the API. Invalid response.")
