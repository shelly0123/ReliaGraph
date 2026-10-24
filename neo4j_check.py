"""Standalone Neo4j connectivity check for ReliaGraph.

Run from the project folder (venv active):   python neo4j_check.py
It reads settings from reliagraph.env (if present) or from environment variables,
then tells you exactly what is wrong and how to fix it.
"""
import os, sys
from pathlib import Path

def load_env_file():
    p = Path(__file__).with_name("reliagraph.env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
        print(f"[ok] loaded settings from {p.name}")
    else:
        print("[info] reliagraph.env not found, using environment variables only")

load_env_file()
uri = os.environ.get("RELIAGRAPH_NEO4J_URI", "bolt://127.0.0.1:7687")
user = os.environ.get("RELIAGRAPH_NEO4J_USER", "neo4j")
pw = os.environ.get("RELIAGRAPH_NEO4J_PASSWORD", "")
print(f"URI={uri}  USER={user}  PASSWORD={'set' if pw else 'MISSING'}")

try:
    import neo4j
    print(f"[ok] neo4j python driver {neo4j.__version__}")
except ImportError:
    sys.exit("[FAIL] neo4j package missing. Run: pip install -r requirements-stores.txt")

if not pw:
    sys.exit("[FAIL] No password. Put RELIAGRAPH_NEO4J_PASSWORD=... in reliagraph.env")

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
try:
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
except AuthError:
    sys.exit("[FAIL] Wrong username/password. Use the password you set when creating the DBMS in Neo4j Desktop.")
except ServiceUnavailable:
    sys.exit("[FAIL] Cannot reach Neo4j. In Neo4j Desktop the DBMS must show 'Running' (click Start). "
             "Check the port in the DBMS details (default 7687).")
except Exception as e:
    sys.exit(f"[FAIL] {type(e).__name__}: {e}")

with driver.session() as s:
    ver = s.run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] AS v").single()
    print(f"[ok] connected to {ver['name']} {ver['v']}")
    n = s.run("MATCH (n:ReliaGraphNode) RETURN count(n) AS c").single()["c"]
    r = s.run("MATCH (:ReliaGraphNode)-[r:RELATES_TO]->(:ReliaGraphNode) RETURN count(r) AS c").single()["c"]
    print(f"[ok] ReliaGraph data in Neo4j: {n} nodes, {r} relationships")
    if n == 0:
        print("[info] 0 nodes is normal until you upload/import data in the app.")
driver.close()
print("\nNeo4j is working. In Neo4j Browser run:  MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 50")
