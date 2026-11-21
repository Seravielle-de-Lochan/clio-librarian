"""Clio v0.2 sanity test — run from the repo root: python tests/sanity_v02.py"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clio_v02.access_store import AccessStore
from clio_v02.ranker import rank, SessionStore, is_protected

tmp = tempfile.mktemp(suffix='.json')
acc = AccessStore(tmp); sessions = SessionStore()
sid = sessions.start(); sess = sessions.get(sid)
cands = [
  {"id":"a","text":"the day we finally named the garden and both laughed about it","distance":0.3,
   "metadata":{"type":"letter","tags":"naming,garden","date_num":20260306,"source":"Letters/Letter4.md"}},
  {"id":"b","text":"grocery list and errands for tuesday","distance":0.9,
   "metadata":{"type":"summary","tags":"errands","date_num":20260715,"source":"Summaries/x.md"}},
  {"id":"c","text":"moving day at the new flat, the first night in","distance":0.5,
   "metadata":{"type":"summary","tags":"milestone","date_num":20260707,"source":"Summaries/2026-07-07.md"}},
  {"id":"d","text":"an old archive chunk with no date about naming the garden","distance":0.35,
   "metadata":{"type":"archive","tags":"","source":"Archive/old.md"}},
]
q = "the day we named the garden"
r1 = rank(cands, q, acc, session=sess, n_results=3)
assert len(r1) == 3 and r1[0]["score"] >= r1[-1]["score"]
acc.record_retrieval([x["id"] for x in r1], {x["id"]: x["protected"] for x in r1})
r2 = rank(cands, q, acc, session=sess, n_results=3)
assert all(x["id"] == "b" for x in r2), "dedup failed"
assert acc.current_strength("a") > 0.3, "strengthening failed"
assert is_protected(cands[2]["metadata"]), "protection failed"
assert len(acc.serendipity_sample(["a","b","c","d"], 2)) == 2
os.remove(tmp); print("ALL SANITY CHECKS PASSED")
