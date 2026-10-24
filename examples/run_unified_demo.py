"""Run a tiny multi-domain ReliaGraph demo without downloading model weights.

The sample texts are original synthetic examples, not official university policies or
statements about any real institution. They exercise the same pipeline across domains.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json

from pipeline import TopicWeaveEngine

DEMO_DOCS = [
    ("education_portal.txt", "The Student Portal provides enrollment records. Enrollment records support attendance reporting."),
    ("education_assessment.txt", "Assessment rubrics define evaluation criteria. Evaluation criteria support transparent grading."),
    ("software_cache.txt", "A cache stores recent results. The service uses the cache to reduce repeated computation."),
    ("software_api.txt", "The API validates request data. Request validation protects the service from malformed input."),
    ("ecology_river.txt", "Wetlands filter runoff. Filtered runoff improves river water quality."),
    ("ecology_habitat.txt", "Coral reefs shelter juvenile fish. Habitat protection supports biodiversity."),
    ("energy_battery.txt", "Battery systems store electrical energy. Battery management systems monitor charge cycles."),
    ("research_papers.txt", "Research papers describe methods and findings. Research libraries organize publications by topic."),
    ("research_reproducibility.txt", "Reproducible experiments record their methods. Recorded methods help other researchers verify results."),
    ("operations_audit.txt", "Audit logs record system changes. Change records support incident investigation."),
]


def main() -> None:
    with TemporaryDirectory(prefix="topicweave_demo_") as temp_dir:
        with TopicWeaveEngine(temp_dir, enable_topic_embeddings=False,
                              enable_dense_retrieval=False, enable_crosslingual=False) as engine:
            metadata = {
                name: {"source_title": name.replace("_", " ").replace(".txt", "").title(),
                       "source_url": f"https://example.invalid/synthetic/{name}",
                       "source_kind": "original_synthetic_demo"}
                for name, _ in DEMO_DOCS
            }
            added = engine.upsert_files([(name, text.encode("utf-8")) for name, text in DEMO_DOCS],
                                        source_metadata=metadata)
            print("=== ReliaGraph multi-domain demo ===")
            print(json.dumps({"documents": len(engine.registry.docs),
                              "chunks": len(engine.registry.chunks),
                              "topics": len(added.snapshot.topics),
                              "nodes": len(added.snapshot.nodes),
                              "edges": len(added.snapshot.edges),
                              "timings_seconds": added.timings}, indent=2))

            for question in (
                "What are the main themes across these documents?",
                "How is Student Portal connected to attendance reporting?",
                "What protects the service from malformed input?",
            ):
                result = engine.query(question, top_k=5)
                print("\nQUESTION:", question)
                print("ROUTE:", result["route"]["route"])
                print("ANSWER:", result["answer"])
                print("SOURCE CHUNKS:", [h["chunk_id"] for h in result["hits"][:5]])
                print("GRAPH PATHS:", len(result["graph_paths"]))

            # Demonstrate an edit; this text is fictional and does not describe MUJ.
            revised_body = "The Student Portal provides enrollment records. Enrollment records support attendance reporting. A revised retention rule requires records to be archived after review."
            changed = engine.upsert_files([("education_portal.txt", revised_body.encode("utf-8"))],
                                          source_metadata={"education_portal.txt": metadata["education_portal.txt"]})
            print("\nEDIT SUMMARY:", json.dumps(changed.change_summary, indent=2))
            print("GRAPH DIFF:", json.dumps(changed.to_dict()["diff"], indent=2)[:1600])
            print("METRICS:", json.dumps(engine.metrics(), indent=2, default=str)[:1600])


if __name__ == "__main__":
    main()
