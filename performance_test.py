import requests
import json
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:8000"

def run_performance_test():
    questions = [
        {"q": "What are the main principles of machine safety?", "k": 3, "mode": "hybrid"},
        {"q": "Explain the concept of Performance Level (PL) in safety systems.", "k": 3, "mode": "hybrid"},
        {"q": "What is an emergency stop and how should it be implemented?", "k": 5, "mode": "hybrid"},
        {"q": "What is the role of ISO 13849 in functional safety?", "k": 3, "mode": "hybrid"},
        {"q": "Describe different types of machine guarding.", "k": 4, "mode": "hybrid"},
        {"q": "What are the requirements for robot safety in industrial environments?", "k": 5, "mode": "hybrid"},
        {"q": "How does risk assessment contribute to machinery safety?", "k": 3, "mode": "hybrid"},
        {"q": "What are the common hazards associated with industrial robots?", "k": 4, "mode": "hybrid"}
    ]

    results = []
    print("\n--- Running Performance Test ---")

    for i, question_data in enumerate(questions):
        print(f"\nQuestion {i+1}: {question_data['q']}")
        start_time = datetime.now()
        try:
            response = requests.post(f"{API_URL}/ask", json=question_data)
            response.raise_for_status() # Raise an exception for HTTP errors
            result = response.json()

            processing_time = (datetime.now() - start_time).total_seconds()

            answer = result.get("answer", {}).get("answer", "N/A")
            sources = ", ".join(result.get("answer", {}).get("sources", [])) if result.get("answer") else "N/A"
            top_score = result["contexts"][0]['score'] if result.get("contexts") else "N/A"
            reranker_used = result.get("reranker_used", False)
            status = "Success"
            error_detail = ""

        except requests.exceptions.ConnectionError:
            status = "Failed (Connection Error)"
            error_detail = f"Could not connect to API at {API_URL}"
            answer, sources, top_score, reranker_used, processing_time = "N/A", "N/A", "N/A", False, 0.0
        except requests.exceptions.RequestException as e:
            status = "Failed (Request Error)"
            error_detail = str(e)
            answer, sources, top_score, reranker_used, processing_time = "N/A", "N/A", "N/A", False, 0.0
        except json.JSONDecodeError:
            status = "Failed (JSON Decode Error)"
            error_detail = "Could not decode JSON response from the API."
            answer, sources, top_score, reranker_used, processing_time = "N/A", "N/A", "N/A", False, 0.0
        except Exception as e:
            status = "Failed (Other Error)"
            error_detail = str(e)
            answer, sources, top_score, reranker_used, processing_time = "N/A", "N/A", "N/A", False, 0.0

        results.append({
            "Question": question_data['q'],
            "Answer": answer,
            "Sources": sources,
            "Top Context Score": f"{top_score:.2f}" if isinstance(top_score, (float, int)) else top_score,
            "Processing Time (s)": f"{processing_time:.2f}",
            "Reranker Used": reranker_used,
            "Status": status,
            "Error Detail": error_detail
        })
    
    df = pd.DataFrame(results)
    print("\n--- Performance Test Results ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_performance_test()
