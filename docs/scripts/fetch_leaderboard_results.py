"""Fetch TDL Challenge 2026 leaderboard results from labeled PRs.

Walks the GitHub PRs of ``geometric-intelligence/TopoBench`` that carry the
``track-1-gnn`` or ``track-2-tnn`` labels, locates a ``results.json`` file
under ``2026_tdl_challenge/`` in each PR head, extracts the in-distribution
test metrics row-by-row, and writes a consolidated
``docs/_static/leaderboard/data/leaderboard.json`` file consumed by the static
leaderboard page.

Usage
-----

Local run (uses ``GITHUB_TOKEN`` if exported, or falls back to unauthenticated
requests with the lower rate limit)::

    python docs/scripts/fetch_leaderboard_results.py

Re-seed from an example file when the API is unreachable / no PRs yet::

    python docs/scripts/fetch_leaderboard_results.py \
        --seed-from path/to/example_results.json \
        --seed-pr-number 0 --seed-pr-title "Example GIN baseline" \
        --seed-track track-1-gnn

The script is idempotent: running it twice without upstream changes produces
the same output bytes.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = "geometric-intelligence/TopoBench"
TRACK_LABELS = ("track-1-gnn", "track-2-tnn")
RESULTS_DIR_HINT = "2026_tdl_challenge"
RESULTS_FILENAME = "results.json"

TASK_METRIC_KEY: dict[str, str] = {
    "community_detection": "test_best_rerun_accuracy",
    "triangle_counting": "test_mse_by_total_triangles",
}
TASK_METRIC_NAME: dict[str, str] = {
    "community_detection": "accuracy",
    "triangle_counting": "mse_per_triangle",
}
TASK_METRIC_DIRECTION: dict[str, str] = {
    "community_detection": "max",
    "triangle_counting": "min",
}

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "_static"
    / "leaderboard"
    / "data"
    / "leaderboard.json"
)


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------


class GitHubClient:
    """Minimal GitHub REST client built on the standard library."""

    API_ROOT = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "topobench-leaderboard-fetcher/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, url: str) -> tuple[int, dict[str, str], bytes]:
        req = urllib.request.Request(url, headers=self._headers())
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.status, dict(resp.headers), resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429) and attempt < 2:
                    reset = exc.headers.get("X-RateLimit-Reset")
                    wait = 2**attempt
                    if reset and reset.isdigit():
                        wait = max(
                            wait, min(60, int(reset) - int(time.time()))
                        )
                    print(
                        f"[rate-limit] HTTP {exc.code}; sleeping {wait}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                if exc.code == 404:
                    return exc.code, dict(exc.headers), b""
                raise
            except urllib.error.URLError:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError(f"Exceeded retries for {url}")

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.API_ROOT}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        status, _headers, body = self._request(url)
        if status == 404:
            return None
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def paginate(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Iterator[Any]:
        page = 1
        per_page = (params or {}).get("per_page", 100)
        while True:
            merged = dict(params or {})
            merged["page"] = page
            merged["per_page"] = per_page
            batch = self.get_json(path, merged)
            if not batch:
                return
            yield from batch
            if len(batch) < per_page:
                return
            page += 1


# ---------------------------------------------------------------------------
# PR discovery + results.json extraction
# ---------------------------------------------------------------------------


def list_labeled_prs(
    client: GitHubClient, labels: Iterable[str]
) -> list[dict[str, Any]]:
    """Return a deduplicated list of PR records carrying any of ``labels``.

    The Issues API surface returns both issues and PRs; we filter to PRs and
    enrich each entry with the PR-specific metadata (head sha, merge state).
    """

    seen: dict[int, dict[str, Any]] = {}
    for label in labels:
        params = {
            "state": "all",
            "labels": label,
            "per_page": 100,
            "sort": "updated",
            "direction": "desc",
        }
        for issue in client.paginate(f"/repos/{REPO}/issues", params):
            if "pull_request" not in issue:
                continue
            number = issue["number"]
            if number in seen:
                seen[number].setdefault("_labels", set()).add(label)
                continue
            pr_full = client.get_json(f"/repos/{REPO}/pulls/{number}")
            if pr_full is None:
                continue
            pr_full["_labels"] = {label}
            pr_full["_issue"] = issue
            seen[number] = pr_full
    return list(seen.values())


def find_results_json_blobs(
    client: GitHubClient, pr: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return git tree entries for any results.json in the PR head tree.

    Uses the recursive git-tree endpoint so we capture results.json files at
    arbitrary depth under ``2026_tdl_challenge/``.
    """

    head = pr.get("head") or {}
    sha = head.get("sha")
    repo_obj = head.get("repo") or {}
    full_name = repo_obj.get("full_name") or REPO
    if not sha:
        return []
    tree = client.get_json(
        f"/repos/{full_name}/git/trees/{sha}", {"recursive": "1"}
    )
    if not tree or not isinstance(tree, dict):
        return []
    entries = tree.get("tree") or []
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path", "")
        if (
            not path.endswith(f"/{RESULTS_FILENAME}")
            and path != RESULTS_FILENAME
        ):
            continue
        if RESULTS_DIR_HINT not in path:
            continue
        candidates.append({**entry, "_repo": full_name})
    candidates.sort(key=lambda e: e.get("path", ""))
    return candidates


def fetch_blob_json(
    client: GitHubClient, blob: dict[str, Any]
) -> dict[str, Any] | None:
    repo_full = blob.get("_repo", REPO)
    sha = blob.get("sha")
    if not sha:
        return None
    obj = client.get_json(f"/repos/{repo_full}/git/blobs/{sha}")
    if obj is None:
        return None
    encoding = obj.get("encoding")
    content = obj.get("content", "")
    if encoding == "base64":
        raw = base64.b64decode(content)
    else:
        raw = content.encode("utf-8")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[skip] could not decode blob {sha}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------


def _coerce_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a results.json payload into per-(task, setting, seed) rows."""

    out: list[dict[str, Any]] = []
    for entry in payload.get("results", []) or []:
        task = entry.get("experiment")
        if task not in TASK_METRIC_KEY:
            continue
        score = _coerce_finite(entry.get(TASK_METRIC_KEY[task]))
        if score is None:
            continue
        try:
            seed = int(entry["train_seed"])
        except (KeyError, TypeError, ValueError):
            continue
        homophily = entry.get("homophily")
        avg_degree = entry.get("avg_degree")
        power_law = entry.get("power_law")
        if not (homophily and avg_degree and power_law):
            continue
        out.append(
            {
                "task": task,
                "homophily": homophily,
                "avg_degree": avg_degree,
                "power_law": power_law,
                "train_seed": seed,
                "score": score,
            }
        )
    return out


def select_track(labels: Iterable[str]) -> str:
    """Return the canonical track tag, preferring track-1 if both are set."""

    label_set = {l.lower() for l in labels}
    for candidate in TRACK_LABELS:
        if candidate in label_set:
            return candidate
    return ""


def build_submission(
    pr: dict[str, Any], payload: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    metadata = payload.get("metadata", {}) or {}
    user = pr.get("user") or {}
    return {
        "pr_number": pr.get("number"),
        "pr_title": pr.get("title"),
        "pr_url": pr.get("html_url"),
        "pr_author": user.get("login"),
        "pr_state": "merged"
        if pr.get("merged_at")
        else pr.get("state", "open"),
        "track": select_track(pr.get("_labels", set())),
        "model_config": metadata.get("model_config"),
        "study_id": metadata.get("study_id"),
        "n_runs": metadata.get("n_runs"),
        "submitted_at_utc": metadata.get("generated_at_utc")
        or pr.get("updated_at")
        or pr.get("created_at"),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def collect_submissions(client: GitHubClient) -> list[dict[str, Any]]:
    submissions: list[dict[str, Any]] = []
    prs = list_labeled_prs(client, TRACK_LABELS)
    print(f"Found {len(prs)} PRs with track labels.", file=sys.stderr)
    for pr in prs:
        number = pr.get("number")
        blobs = find_results_json_blobs(client, pr)
        if not blobs:
            print(
                f"[skip] PR #{number}: no results.json under {RESULTS_DIR_HINT}/",
                file=sys.stderr,
            )
            continue
        # Prefer the lexicographically latest path so the most recent
        # study_id wins when several JSONs are committed.
        chosen = blobs[-1]
        payload = fetch_blob_json(client, chosen)
        if payload is None:
            print(
                f"[skip] PR #{number}: could not load {chosen.get('path')}",
                file=sys.stderr,
            )
            continue
        rows = extract_rows(payload)
        if not rows:
            print(
                f"[skip] PR #{number}: no usable rows in {chosen.get('path')}",
                file=sys.stderr,
            )
            continue
        submissions.append(build_submission(pr, payload, rows))
        print(
            f"[ok] PR #{number} ({chosen.get('path')}): {len(rows)} rows",
            file=sys.stderr,
        )
    return submissions


def write_leaderboard(
    output_path: Path, submissions: list[dict[str, Any]]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tasks": {
            task: {
                "metric": TASK_METRIC_NAME[task],
                "direction": TASK_METRIC_DIRECTION[task],
            }
            for task in TASK_METRIC_KEY
        },
        "submissions": submissions,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {output_path} ({len(submissions)} submissions).",
        file=sys.stderr,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the consolidated leaderboard JSON (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--seed-from",
        type=Path,
        default=None,
        help="Path to an example results.json. When given, the script does NOT call "
        "the GitHub API and instead emits a single seeded submission. Useful "
        "for offline preview / initial commit.",
    )
    p.add_argument("--seed-pr-number", type=int, default=0)
    p.add_argument(
        "--seed-pr-title", type=str, default="Example submission (seed)"
    )
    p.add_argument(
        "--seed-pr-url", type=str, default=f"https://github.com/{REPO}"
    )
    p.add_argument("--seed-pr-author", type=str, default="example")
    p.add_argument(
        "--seed-track",
        type=str,
        default=TRACK_LABELS[0],
        choices=list(TRACK_LABELS),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.seed_from is not None:
        payload = json.loads(args.seed_from.read_text(encoding="utf-8"))
        rows = extract_rows(payload)
        if not rows:
            print(
                f"error: no usable rows in {args.seed_from}",
                file=sys.stderr,
            )
            return 2
        metadata = payload.get("metadata", {}) or {}
        seeded = {
            "pr_number": args.seed_pr_number,
            "pr_title": args.seed_pr_title,
            "pr_url": args.seed_pr_url,
            "pr_author": args.seed_pr_author,
            "pr_state": "seed",
            "track": args.seed_track,
            "model_config": metadata.get("model_config"),
            "study_id": metadata.get("study_id"),
            "n_runs": metadata.get("n_runs"),
            "submitted_at_utc": metadata.get("generated_at_utc"),
            "rows": rows,
        }
        write_leaderboard(args.output, [seeded])
        return 0

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    client = GitHubClient(token=token)
    submissions = collect_submissions(client)
    write_leaderboard(args.output, submissions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
