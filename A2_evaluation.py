import json
import requests
import time
from datetime import datetime

API_URL = "http://localhost:8002"


def test_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_evaluation():
    if not test_api():
        print("ERROR: API not responding. Start Docker: docker-compose up")
        return

    with open('evaluation.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"Running {len(questions)} queries...")
    print("=" * 60)

    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:50]}...")

        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": q["question"],
                      "top_k": 5, "include_sources": True},
                timeout=30
            )

            data = response.json()

            results.append({
                "question_number": i,
                "question": q["question"],
                "question_type": q.get("question_type", "factual"),
                "ideal_answer": q["ideal_answer"],
                "generated_answer": data["answer"],
                "sources": data.get("sources", []),
                "success": True
            })

            print(f"  OK ({len(data['answer'])} chars)")

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "question_number": i,
                "question": q["question"],
                "error": str(e),
                "success": False
            })

        time.sleep(1)

    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Results saved to evaluation_results.json")
    print(f"Success: {sum(1 for r in results if r['success'])}/{len(results)}")


if __name__ == "__main__":
    run_evaluation()
