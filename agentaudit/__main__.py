"""CLI: python -m agentaudit {serve,seed,audit}"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="agentaudit", description="QA governance for LLM agents")
    parser.add_argument("--db", default=os.environ.get("AGENTAUDIT_DB", "data/agentaudit.db"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--seed", action="store_true", help="load sample traces (offline judge) before starting")

    seed = sub.add_parser("seed", help="audit the built-in sample traces")
    seed.add_argument("--judge", default="heuristic", choices=["heuristic", "claude", "auto"])

    audit = sub.add_parser("audit", help="audit trace JSON file(s); a file may hold one trace or a list")
    audit.add_argument("files", nargs="+", type=Path)
    audit.add_argument("--judge", default=None, choices=["heuristic", "claude", "auto"])
    audit.add_argument("--no-store", action="store_true", help="print only; do not save to the dashboard database")

    args = parser.parse_args(argv)

    from .graph import Auditor
    from .judges import claude_available, get_judge
    from .schemas import Trace
    from .seed import sample_traces
    from .store import Store

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        app = create_app(db_path=args.db)
        if args.seed:
            Auditor(get_judge("heuristic"), Store(args.db)).audit_many(sample_traces())
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    mode = args.judge
    if mode == "claude" and not claude_available():
        print("error: --judge claude needs credentials (set ANTHROPIC_API_KEY)", file=sys.stderr)
        return 2
    store = None if getattr(args, "no_store", False) else Store(args.db)
    auditor = Auditor(get_judge(mode), store)

    if args.cmd == "seed":
        results = auditor.audit_many(sample_traces())
        print(f"Audited {len(results)} sample traces with the {auditor.judge.name} judge into {args.db}")
        return 0

    traces: list[Trace] = []
    for f in args.files:
        data = json.loads(f.read_text())
        traces += [Trace.model_validate(t) for t in (data if isinstance(data, list) else [data])]
    exit_code = 0
    for r in auditor.audit_many(traces):
        print(f"{r.verdict.upper():5} {r.composite:3}  {r.agent}/{r.trace_id}  {r.summary}")
        if r.review:
            print(f"        needs review: {'; '.join(r.review.reasons)}")
        exit_code = max(exit_code, {"pass": 0, "warn": 0, "fail": 1}[r.verdict])
    return exit_code  # non-zero when any run fails, so `agentaudit audit` can gate CI


if __name__ == "__main__":
    raise SystemExit(main())
