"""Optional production-style graph/vector stores for ReliaGraph.

SQLite + NetworkX remain the zero-configuration local fallback. Set RELIAGRAPH_NEO4J_URI,
RELIAGRAPH_NEO4J_USER and RELIAGRAPH_NEO4J_PASSWORD to mirror graph snapshots to Neo4j.
Set RELIAGRAPH_CHROMA=1 to persist chunk text in a local Chroma collection.
These are additive stores; this adapter never claims they are active unless connected.
"""
from __future__ import annotations
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def _load_env_file():
    f = ROOT / "reliagraph.env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
_load_env_file()
_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None
_LAST = {"neo4j": {"configured": False, "connected": False, "message": "Using built-in SQLite/NetworkX store"},
         "chroma": {"configured": False, "connected": False, "message": "Vector store disabled; lexical/local retrieval remains available"}}

def _obj(x):
    if is_dataclass(x): return asdict(x)
    if isinstance(x, dict): return x
    try: return vars(x)
    except Exception: return {}

def optional_store_status():
    return {k: dict(v) for k,v in _LAST.items()}

def sync_optional_stores(snapshot: Any, chunks: list[Any]):
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    uri = os.environ.get("RELIAGRAPH_NEO4J_URI", "").strip()
    user = os.environ.get("RELIAGRAPH_NEO4J_USER", "neo4j").strip()
    password = os.environ.get("RELIAGRAPH_NEO4J_PASSWORD", "")
    _LAST["neo4j"]["configured"] = bool(uri and password)
    if uri and password:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            snap = _obj(snapshot) if snapshot else {}
            nodes = snap.get("nodes", []) or []
            edges = snap.get("edges", []) or []
            with driver.session() as session:
                session.run("CREATE INDEX rg_node_id IF NOT EXISTS FOR (n:ReliaGraphNode) ON (n.id)")
                session.run("MATCH (n:ReliaGraphNode) DETACH DELETE n")
                node_rows = []
                for raw in nodes:
                    n = _obj(raw); nid = str(n.get("node_id", ""))
                    if nid:
                        node_rows.append(dict(id=nid, label=str(n.get("label", nid))[:500],
                                              kind=str(n.get("node_type", "entity")),
                                              payload=str(n)[:12000]))
                for i in range(0, len(node_rows), 500):
                    session.run("UNWIND $rows AS r CREATE (n:ReliaGraphNode {id:r.id, label:r.label, kind:r.kind, payload:r.payload})",
                                rows=node_rows[i:i+500])
                edge_rows = []
                for raw in edges:
                    e = _obj(raw); a = str(e.get("source_node_id", "")); b = str(e.get("target_node_id", ""))
                    if a and b:
                        edge_rows.append(dict(a=a, b=b, kind=str(e.get("relation_type", "related_to")),
                                              status=str(e.get("status", "candidate")), payload=str(e)[:12000]))
                for i in range(0, len(edge_rows), 500):
                    session.run("UNWIND $rows AS r MATCH (a:ReliaGraphNode {id:r.a}), (b:ReliaGraphNode {id:r.b}) "
                                "CREATE (a)-[:RELATES_TO {kind:r.kind, status:r.status, payload:r.payload}]->(b)",
                                rows=edge_rows[i:i+500])
                # Read back from the database: proof that Neo4j really holds the data.
                got_n = session.run("MATCH (n:ReliaGraphNode) RETURN count(n) AS c").single()["c"]
                got_e = session.run("MATCH (:ReliaGraphNode)-[r:RELATES_TO]->(:ReliaGraphNode) RETURN count(r) AS c").single()["c"]
            driver.close(); _LAST["neo4j"]={"configured":True,"connected":True,"verified_nodes":got_n,"verified_edges":got_e,"uri":uri,"message":f"Verified in Neo4j by read-back: {got_n} nodes, {got_e} relationships"}
        except Exception as exc:
            _LAST["neo4j"]={"configured":True,"connected":False,"uri":uri,"message":f"Neo4j unavailable: {type(exc).__name__}: {str(exc)[:220]}"}
    else:
        _LAST["neo4j"]={"configured":False,"connected":False,"message":"Optional: set RELIAGRAPH_NEO4J_URI/USER/PASSWORD; fallback active"}

    chroma_enabled = os.environ.get("RELIAGRAPH_CHROMA", "0") == "1"
    _LAST["chroma"]["configured"] = chroma_enabled
    if chroma_enabled:
        try:
            import chromadb
            path = Path(os.environ.get("RELIAGRAPH_CHROMA_PATH", str(ROOT / "data" / "chroma")))
            _CHROMA_CLIENT = chromadb.PersistentClient(path=str(path))
            try: _CHROMA_CLIENT.delete_collection("reliagraph_chunks")
            except Exception: pass
            _CHROMA_COLLECTION = _CHROMA_CLIENT.get_or_create_collection("reliagraph_chunks", metadata={"hnsw:space":"cosine"})
            ids=[]; docs=[]; metas=[]
            for i, raw in enumerate(chunks):
                c=_obj(raw); text=str(c.get("text", "") or "")
                if not text.strip(): continue
                cid=str(c.get("chunk_id", c.get("id", f"chunk-{i}")))
                ids.append(cid[:500]); docs.append(text[:12000]); metas.append({"doc_id":str(c.get("doc_id", ""))[:500],"source_title":str(c.get("source_title", c.get("filename", "")))[:500]})
            for start in range(0,len(ids),256):
                _CHROMA_COLLECTION.add(ids=ids[start:start+256],documents=docs[start:start+256],metadatas=metas[start:start+256])
            _LAST["chroma"]={"configured":True,"connected":True,"message":f"Persisted {len(ids)} source chunks; Chroma embedding function configured by library defaults"}
        except Exception as exc:
            _LAST["chroma"]={"configured":True,"connected":False,"message":f"Chroma unavailable: {type(exc).__name__}: {str(exc)[:200]}; fallback retrieval active"}
    else:
        _LAST["chroma"]={"configured":False,"connected":False,"message":"Optional: set RELIAGRAPH_CHROMA=1; local retrieval fallback active"}

def clear_optional_stores():
    # Full replacement occurs on next sync; status will show connectivity if configured.
    if _CHROMA_CLIENT is not None:
        try: _CHROMA_CLIENT.delete_collection("reliagraph_chunks")
        except Exception: pass


def check_neo4j():
    """Cheap live probe (no writes): connects and counts what Neo4j currently holds."""
    uri = os.environ.get("RELIAGRAPH_NEO4J_URI", "").strip()
    user = os.environ.get("RELIAGRAPH_NEO4J_USER", "neo4j").strip()
    password = os.environ.get("RELIAGRAPH_NEO4J_PASSWORD", "")
    if not (uri and password) or password == "PUT_YOUR_NEO4J_PASSWORD_HERE":
        _LAST["neo4j"].update(configured=False, connected=False,
                              message="Not configured: edit reliagraph.env and restart")
        return dict(_LAST["neo4j"])
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        with driver.session() as s:
            n = s.run("MATCH (n:ReliaGraphNode) RETURN count(n) AS c").single()["c"]
            e = s.run("MATCH (:ReliaGraphNode)-[r:RELATES_TO]->(:ReliaGraphNode) RETURN count(r) AS c").single()["c"]
        driver.close()
        _LAST["neo4j"] = {"configured": True, "connected": True, "verified_nodes": n, "verified_edges": e,
                          "uri": uri, "message": f"Connected. Neo4j holds {n} nodes, {e} relationships"}
    except Exception as exc:
        _LAST["neo4j"] = {"configured": True, "connected": False, "uri": uri,
                          "message": f"Neo4j unavailable: {type(exc).__name__}: {str(exc)[:220]}"}
    return dict(_LAST["neo4j"])
