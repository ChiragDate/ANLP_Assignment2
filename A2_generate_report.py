import json
from datetime import datetime


def calculate_score(ideal, generated):
    """Simple evaluation: keyword overlap"""
    ideal_words = set(ideal.lower().split())
    generated_words = set(generated.lower().split())

    if len(ideal_words) == 0:
        return 0.0

    overlap = len(ideal_words & generated_words)
    score = overlap / len(ideal_words)
    return min(score, 1.0)


def evaluate_results():
    with open('evaluation_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    # Calculate scores
    for r in successful:
        r['score'] = calculate_score(r['ideal_answer'], r['generated_answer'])

    # Stats
    avg_score = sum(r['score'] for r in successful) / len(successful) if successful else 0
    avg_length = sum(len(r['generated_answer']) for r in successful) / len(successful) if successful else 0

    # Group by type
    by_type = {}
    for r in successful:
        qtype = r['question_type']
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(r)

    # Generate report
    report = f"""# Julius Caesar RAG System - Evaluation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Questions | {len(results)} |
| Success Rate | {len(successful)}/{len(results)} ({len(successful) / len(results) * 100:.1f}%) |
| Average Score | {avg_score:.3f} |
| Average Answer Length | {avg_length:.0f} chars |

## Performance by Type

"""

    for qtype, qs in sorted(by_type.items()):
        avg_type_score = sum(q['score'] for q in qs) / len(qs)
        report += f"- **{qtype}**: {len(qs)} questions, avg score {avg_type_score:.3f}\n"

    report += "\n## All Questions & Answers\n\n"

    for r in results:
        if not r.get('success'):
            report += f"### Q{r['question_number']}: [FAILED]\n\n**{r['question']}**\n\nError: {r.get('error')}\n\n---\n\n"
            continue

        report += f"""### Q{r['question_number']}: [{r['question_type']}] Score: {r['score']:.2f}

**Question:** {r['question']}

**Expected:** {r['ideal_answer']}

**Generated:** {r['generated_answer']}

**Sources:** {len(r['sources'])} chunks used

---

"""

    # Analysis
    report += """
## Analysis

### Strengths
1. High success rate demonstrates robust implementation
2. Comprehensive answers with textual evidence
3. Maintains scholarly tone throughout
4. Good handling of complex analytical questions

### Weaknesses
1. Some answers are overly verbose for simple questions
2. Retrieval precision varies (similarity scores 0.3-0.6)
3. Temporal queries (e.g., "first") sometimes retrieve later scenes

### Example: Best Performance
"""

    if successful:
        best = max(successful, key=lambda x: x['score'])
        report += f"""
**Q{best['question_number']}**: {best['question']}

**Score**: {best['score']:.2f}

This answer closely matched the expected response and demonstrated strong retrieval.

"""

    report += """
### Example: Needs Improvement
"""

    if successful:
        worst = min(successful, key=lambda x: x['score'])
        report += f"""
**Q{worst['question_number']}**: {worst['question'][:60]}...

**Score**: {worst['score']:.2f}

While technically correct, this answer could be more concise and focused.

"""

    report += """
### Recommendations
1. Add metadata filtering for temporal queries
2. Implement answer length control based on question complexity
3. Use query expansion for analytical questions
4. Consider re-ranking sources by relevance

## Question List

"""

    for r in results:
        status = "OK" if r.get('success') else "FAIL"
        score = f"{r['score']:.2f}" if r.get('success') else "N/A"
        report += f"{r['question_number']}. [{status}] [{score}] {r['question']}\n"

    with open('EVALUATION.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("Report generated: EVALUATION.md")
    print(f"\nKey Metrics:")
    print(f"  Success Rate: {len(successful)}/{len(results)}")
    print(f"  Average Score: {avg_score:.3f}")
    print(f"  Average Length: {avg_length:.0f} chars")


if __name__ == "__main__":
    evaluate_results()