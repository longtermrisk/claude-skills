#!/usr/bin/env python3
"""Zero-dependency client for the forecasting AI's /api/v1 endpoints.

For externally managed agents (r-docs runs, Claude Code, cron jobs). Standard
library only, so it can be copied as a single file into any environment.

Configuration (environment or flags):
    FAI_BASE_URL   e.g. http://100.x.y.z:14160  (the service's tailnet address)
    FAI_TOKEN      the bearer token a human minted on the /agents page or `fai mint`

Usage:
    fai_client.py me                          # account, portfolio value, positions
    fai_client.py questions                   # tracked questions + latest quotes
    fai_client.py question <id>               # one question, quote, tags, resolution
    fai_client.py clusters                    # all clusters
    fai_client.py cluster <id>                # members, scenario order, my decomposition
    fai_client.py create-cluster --title T --question-id A --question-id B
    fai_client.py submit <cluster_id> --file payload.json [--run-link URL]
    fai_client.py report <question_id> --file report.json

`submit` payload shape (see the skill / cluster endpoint for the scenario order):
    {"latents": [{"name": "...", "question": "..."}, ...],
     "rows": [{"id": "<question_id>", "probs": [p, ...], "confidence": 0.8,
               "reasoning": "..."}, ...],
     "note": "optional"}
Typed (categorical/numeric/date) members use "distributions" instead of "probs".
If --run-link is omitted, $GDOCS_RUN_URL is used when set, so r-docs runs link
back to their session automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _request(base: str, token: str, method: str, path: str, body: dict | None = None):
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (ValueError, AttributeError):
            pass
        sys.exit(f"HTTP {e.code} on {method} {path}: {detail}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=os.environ.get("FAI_BASE_URL", ""))
    p.add_argument("--token", default=os.environ.get("FAI_TOKEN", ""))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("me")
    sub.add_parser("questions")
    q = sub.add_parser("question"); q.add_argument("id")
    sub.add_parser("clusters")
    c = sub.add_parser("cluster"); c.add_argument("id")
    cc = sub.add_parser("create-cluster")
    cc.add_argument("--title", required=True)
    cc.add_argument("--question-id", action="append", required=True,
                    dest="question_ids")
    s = sub.add_parser("submit")
    s.add_argument("cluster_id")
    s.add_argument("--file", required=True, help="JSON payload (see module docstring)")
    s.add_argument("--run-link", default=os.environ.get("GDOCS_RUN_URL", ""))
    s.add_argument("--model", default="", help="optional model name for attribution")
    r = sub.add_parser("report")
    r.add_argument("question_id")
    r.add_argument("--file", required=True,
                   help='JSON: {"distribution": ..., "confidence": 0-1, "rationale"}')

    a = p.parse_args(argv)
    if not a.base_url:
        sys.exit("set FAI_BASE_URL (or --base-url)")
    if not a.token:
        sys.exit("set FAI_TOKEN (or --token); a human mints one on the /agents page")

    def call(method, path, body=None):
        print(json.dumps(_request(a.base_url, a.token, method, path, body), indent=2))

    if a.cmd == "me":
        call("GET", "/api/v1/me")
    elif a.cmd == "questions":
        call("GET", "/api/v1/questions")
    elif a.cmd == "question":
        call("GET", f"/api/v1/questions/{urllib.parse.quote(a.id, safe='')}")
    elif a.cmd == "clusters":
        call("GET", "/api/v1/clusters")
    elif a.cmd == "cluster":
        call("GET", f"/api/v1/clusters/{urllib.parse.quote(a.id, safe='')}")
    elif a.cmd == "create-cluster":
        call("POST", "/api/v1/clusters",
             {"title": a.title, "question_ids": a.question_ids})
    elif a.cmd == "submit":
        payload = json.load(open(a.file))
        if a.run_link:
            payload["run_link"] = a.run_link
        if a.model:
            payload["model"] = a.model
        call("POST",
             f"/api/v1/clusters/{urllib.parse.quote(a.cluster_id, safe='')}/submit",
             payload)
    elif a.cmd == "report":
        body = json.load(open(a.file))
        body["question_id"] = a.question_id
        call("POST", "/api/v1/reports", body)


if __name__ == "__main__":
    main()
