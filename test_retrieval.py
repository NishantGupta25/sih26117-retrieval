from retrieval_module import search_sop

test_queries = [
    "How do I check a valve?",
    "How often should gauges be calibrated?",
    "What should I photograph during inspection?",
    "Is it safe to touch a hot pressurized valve?",
    "My gauge glass looks cracked, what do I do?"
]

for q in test_queries:
    print(f"\nQuery: {q}")
    results = search_sop(q)
    for r in results:
        print(f" - {r[:100]}...")