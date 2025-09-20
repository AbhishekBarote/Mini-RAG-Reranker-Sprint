import requests
import json
import re

def test_api():
    """Test the API with sample queries"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print("Health check:", health_data)
        
        chunk_count = 0
        if "message" in health_data:
            match = re.search(r'Database contains (\d+) chunks.', health_data['message'])
            if match:
                chunk_count = int(match.group(1))

        if chunk_count == 0:
            print("WARNING: Database is empty. Run ingestion first.")
            return
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        return
    
    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/stats")
        print("Stats:", response.json())
    except Exception as e:
        print(f"Failed to get stats: {e}")
    
    # Test sources endpoint
    try:
        response = requests.get(f"{base_url}/sources")
        sources = response.json()
        print(f"Found {len(sources)} sources in database")
    except Exception as e:
        print(f"Failed to get sources: {e}")
    
    # Test queries
    test_queries = [
        {"q": "safety", "k": 3, "mode": "baseline"},
        {"q": "machine safety", "k": 3, "mode": "baseline"},
        {"q": "emergency stops", "k": 5, "mode": "hybrid", "alpha": 0.7},
        {"q": "PLd classification", "k": 3, "mode": "hybrid"}
    ]
    
    for query in test_queries:
        try:
            print(f"\nTesting query: {query['q']}")
            response = requests.post(f"{base_url}/ask", json=query)
            
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                print(f"Response: {response.text}")
                continue
                
            result = response.json()
            
            print(f"Status: {response.status_code}")
            
            if result.get("answer"):
                print(f"Answer: {result['answer']['answer']}")
                print(f"Sources: {result['answer']['sources']}")
            else:
                print(f"No answer: {result.get('abstention_reason', 'Unknown reason')}")
            
            if result.get("contexts"):
                print(f"Top context scores: {[ctx['score'] for ctx in result['contexts'][:3]]}")
            print(f"Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"Reranker used: {result.get('reranker_used', False)}")
            
        except Exception as e:
            print(f"Failed to process query: {e}")

if __name__ == "__main__":
    test_api()