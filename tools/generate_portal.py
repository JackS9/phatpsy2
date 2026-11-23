#!/usr/bin/env python3
"""Generate the PHATPSY 2.0 interactive project portal (docs/dashboard/index.html).

Updates the live repository telemetry badges (total commits, branch count,
tag count, latest commit SHA/URL) and the embedded ``snapshotCommits`` timeline
in the PHATPSY 2.0 dashboard HTML while preserving the full visual design,
architecture explorer, quick-start tabs, and client-side GitHub sync logic.

Data sources (``--source``):
  - ``auto`` (default): read from local ``git`` history if available, otherwise
    query the GitHub REST API, otherwise keep the template's embedded snapshot.
  - ``git``: read telemetry and recent commits (including touched paths) from a
    local git checkout — fast, offline, and immune to API rate limits.
  - ``github``: fetch live telemetry and commits from the GitHub REST API using
    ``per_page=1`` ``Link`` header pagination and a ``certifi``-backed SSL
    context when installed.
  - ``snapshot``: preserve the template's embedded snapshot verbatim (useful
    for deterministic parity checks).

Usage examples::

    # Refresh docs/dashboard/index.html from local git history (offline)
    python3 tools/generate_portal.py --source git

    # Refresh from live GitHub API into a preview file
    python3 tools/generate_portal.py --source github -o /tmp/portal.html

    # Verify byte-for-byte parity with the template snapshot
    python3 tools/generate_portal.py --source snapshot -o /tmp/portal_snap.html
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import html
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_REPO = "JackS9/phatpsy2"
DEFAULT_BRANCH = "master"
DEFAULT_COMMIT_LIMIT = 14
DEFAULT_TIMEOUT_S = 15

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = REPO_ROOT / "docs" / "dashboard" / "index.html"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "dashboard" / "index.html"

_MERGE_PR_RE = re.compile(
    r"^Merge pull request #(\d+) from [^/\s]+/([^\s]+)",
    re.IGNORECASE,
)
_LINK_LAST_PAGE_RE = re.compile(r'[?&]page=(\d+)[^>]*>;\s*rel="last"')


@dataclasses.dataclass(frozen=True)
class CommitEntry:
    """Single commit entry rendered in the dashboard activity feed."""

    sha: str
    full_sha: str
    time: str
    message: str
    ref: str
    paths: tuple[str, ...]
    category: str
    url: str


@dataclasses.dataclass(frozen=True)
class PortalTelemetry:
    """Live repository telemetry injected into the dashboard HTML."""

    repo: str
    branch: str
    total_commits: int
    branch_count: int
    tag_count: int
    latest_sha: str
    latest_full_sha: str
    commits: tuple[CommitEntry, ...]
    source: str


def build_ssl_context() -> ssl.SSLContext:
    """Return an SSL context that uses ``certifi`` CA roots when available.

    Python.org macOS builds do not always bind the system keychain automatically;
    using ``certifi`` avoids silent ``CERTIFICATE_VERIFY_FAILED`` failures.
    """
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def parse_total_count_from_link_header(
    link_header: str | None,
    page_items: int,
    per_page: int = 1,
) -> int:
    """Compute exact item count from a GitHub REST API ``Link`` header.

    When called with ``per_page=1``, ``rel="last"`` gives the exact total
    count. When ``Link`` is absent (single-page response), returns
    ``page_items``.
    """
    if link_header:
        match = _LINK_LAST_PAGE_RE.search(link_header)
        if match:
            last_page = int(match.group(1))
            if per_page == 1:
                return last_page
            return last_page * per_page
    return page_items


def format_relative_time(
    timestamp: dt.datetime,
    now: dt.datetime | None = None,
) -> str:
    """Format a UTC timestamp as a concise human-readable relative label."""
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    diff_sec = max(0, int((now - timestamp).total_seconds()))
    if diff_sec < 3600:
        mins = max(1, diff_sec // 60)
        return "1 min ago" if mins == 1 else f"{mins} mins ago"
    if diff_sec < 86400:
        hours = diff_sec // 3600
        return "1 hour ago" if hours == 1 else f"{hours} hours ago"
    days = diff_sec // 86400
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def parse_iso_timestamp(iso_str: str) -> dt.datetime:
    """Parse an ISO-8601 timestamp (including trailing ``Z``) into UTC."""
    normalized = iso_str.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def extract_pr_ref(subject: str, default_branch: str) -> str:
    """Extract ``PR #N · branch`` from a GitHub merge commit subject."""
    match = _MERGE_PR_RE.match(subject)
    if match:
        return f"PR #{match.group(1)} · {match.group(2)}"
    return default_branch


def summarize_paths(file_paths: Iterable[str], limit: int = 3) -> tuple[str, ...]:
    """Collapse raw git file paths into up to ``limit`` top-level path badges."""
    seen: list[str] = []
    for raw in file_paths:
        cleaned = raw.strip().lstrip("./")
        if not cleaned:
            continue
        if "/" in cleaned:
            top = cleaned.split("/", 1)[0] + "/"
        else:
            top = cleaned
        if top not in seen:
            seen.append(top)
        if len(seen) >= limit:
            break
    return tuple(seen)


def classify_commit(subject: str, paths: Sequence[str] = ()) -> str:
    """Classify a commit into one of the four dashboard feed categories.

    Categories match the dashboard filter tabs:
      - ``docs`` (Docs & Specs)
      - ``vb`` (VB & Catalysis)
      - ``fortran`` (Fortran Core)
      - ``build`` (Build & Tests)
    """
    lower = subject.lower()
    if (
        lower.startswith("docs")
        or "merge pull request" in lower
        or "license" in lower
        or "roadmap" in lower
    ):
        return "docs"
    if (
        "(vb)" in lower
        or "valence-bond" in lower
        or "phase18" in lower
        or "no/ni" in lower
        or "assembler" in lower
        or "catalysis" in lower
        or "flagship" in lower
    ):
        return "vb"
    if (
        "gradient" in lower
        or "scf" in lower
        or "diagnostics" in lower
        or "methodology" in lower
        or "fortran" in lower
    ):
        return "fortran"
    if paths:
        doc_roots = {"docs/", ".agents/", "AGENTS.md", "ROADMAP.md", "README.md"}
        if all(p in doc_roots for p in paths):
            return "docs"
        if any(p == "src/" for p in paths):
            return "fortran"
    return "build"


def _run_git(args: Sequence[str], git_dir: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(git_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def fetch_telemetry_from_git(
    git_dir: Path,
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    commit_limit: int = DEFAULT_COMMIT_LIMIT,
    now: dt.datetime | None = None,
) -> PortalTelemetry:
    """Read live telemetry and recent commit history from a local git repo."""
    # Resolve target ref: if both local `branch` and `origin/<branch>` exist,
    # prefer whichever has more commits so a stale local tracking branch never
    # shadows a freshly fetched `origin/<branch>` (or vice versa).
    valid_refs: list[tuple[int, str]] = []
    for candidate in (f"origin/{branch}", branch):
        try:
            _run_git(["rev-parse", "--verify", "--quiet", candidate], git_dir)
            count = int(_run_git(["rev-list", "--count", candidate], git_dir))
            valid_refs.append((count, candidate))
        except subprocess.CalledProcessError:
            continue

    if valid_refs:
        total_commits, target_ref = max(valid_refs, key=lambda pair: pair[0])
    else:
        target_ref = "HEAD"
        total_commits = int(_run_git(["rev-list", "--count", target_ref], git_dir))

    # Prefer remote branches (excluding HEAD symbolic pointer) to match GitHub count
    remote_branches = [
        line.strip()
        for line in _run_git(
            ["for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"],
            git_dir,
        ).splitlines()
        if line.strip() and line.strip() != "origin/HEAD" and line.strip() != "origin"
    ]
    if remote_branches:
        branch_count = len(remote_branches)
    else:
        local_branches = [
            line.strip()
            for line in _run_git(
                ["for-each-ref", "--format=%(refname:short)", "refs/heads"],
                git_dir,
            ).splitlines()
            if line.strip()
        ]
        branch_count = max(1, len(local_branches))

    tags = [
        line.strip()
        for line in _run_git(["tag", "--list"], git_dir).splitlines()
        if line.strip()
    ]
    tag_count = len(tags)

    # Fetch commit metadata + changed files in a single git log invocation.
    record_sep = "\x1e"
    field_sep = "\x1f"
    raw_log = _run_git(
        [
            "log",
            target_ref,
            f"-n{commit_limit}",
            f"--pretty=format:{record_sep}%H{field_sep}%cI{field_sep}%s",
            "--name-only",
        ],
        git_dir,
    )

    commits: list[CommitEntry] = []
    for block in raw_log.split(record_sep):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        header_parts = lines[0].split(field_sep)
        if len(header_parts) < 3:
            continue
        full_sha, iso_date, subject = (
            header_parts[0].strip(),
            header_parts[1].strip(),
            header_parts[2].strip(),
        )
        file_lines = [ln.strip() for ln in lines[1:] if ln.strip()]
        paths = summarize_paths(file_lines)
        short_sha = full_sha[:7]
        commits.append(
            CommitEntry(
                sha=short_sha,
                full_sha=full_sha,
                time=format_relative_time(parse_iso_timestamp(iso_date), now=now),
                message=subject,
                ref=extract_pr_ref(subject, branch),
                paths=paths,
                category=classify_commit(subject, paths),
                url=f"https://github.com/{repo}/commit/{full_sha}",
            )
        )

    if not commits:
        raise RuntimeError(f"No commits found for ref {target_ref!r} in {git_dir}")

    return PortalTelemetry(
        repo=repo,
        branch=branch,
        total_commits=total_commits,
        branch_count=branch_count,
        tag_count=tag_count,
        latest_sha=commits[0].sha,
        latest_full_sha=commits[0].full_sha,
        commits=tuple(commits),
        source="git",
    )


def _github_get_json(
    url: str,
    ssl_ctx: ssl.SSLContext,
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> tuple[Any, str | None]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "PHATPSY2-Portal-Generator",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
        link_header = resp.headers.get("Link")
        payload = json.loads(resp.read().decode("utf-8"))
        return payload, link_header


def fetch_telemetry_from_github(
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    commit_limit: int = DEFAULT_COMMIT_LIMIT,
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    now: dt.datetime | None = None,
) -> PortalTelemetry:
    """Fetch exact repository telemetry and commits from the GitHub REST API."""
    ssl_ctx = build_ssl_context()
    base = f"https://api.github.com/repos/{repo}"

    # Use per_page=1 so Link rel="last" gives the exact total count.
    commits_one, commits_link = _github_get_json(
        f"{base}/commits?sha={branch}&per_page=1",
        ssl_ctx=ssl_ctx,
        token=token,
        timeout=timeout,
    )
    total_commits = parse_total_count_from_link_header(
        commits_link,
        page_items=len(commits_one) if isinstance(commits_one, list) else 0,
        per_page=1,
    )

    branches_one, branches_link = _github_get_json(
        f"{base}/branches?per_page=1",
        ssl_ctx=ssl_ctx,
        token=token,
        timeout=timeout,
    )
    branch_count = parse_total_count_from_link_header(
        branches_link,
        page_items=len(branches_one) if isinstance(branches_one, list) else 0,
        per_page=1,
    )

    tags_one, tags_link = _github_get_json(
        f"{base}/tags?per_page=1",
        ssl_ctx=ssl_ctx,
        token=token,
        timeout=timeout,
    )
    tag_count = parse_total_count_from_link_header(
        tags_link,
        page_items=len(tags_one) if isinstance(tags_one, list) else 0,
        per_page=1,
    )

    raw_commits, _ = _github_get_json(
        f"{base}/commits?sha={branch}&per_page={commit_limit}",
        ssl_ctx=ssl_ctx,
        token=token,
        timeout=timeout,
    )
    if not isinstance(raw_commits, list) or not raw_commits:
        raise RuntimeError(f"GitHub API returned no commits for {repo}@{branch}")

    commits: list[CommitEntry] = []
    for item in raw_commits[:commit_limit]:
        full_sha = str(item.get("sha", ""))
        commit_obj = item.get("commit") or {}
        raw_msg = str(commit_obj.get("message") or "Commit")
        subject = raw_msg.splitlines()[0].strip()
        committer = commit_obj.get("committer") or commit_obj.get("author") or {}
        iso_date = str(committer.get("date") or "")
        rel_time = (
            format_relative_time(parse_iso_timestamp(iso_date), now=now)
            if iso_date
            else "recent"
        )
        url = str(item.get("html_url") or f"https://github.com/{repo}/commit/{full_sha}")
        commits.append(
            CommitEntry(
                sha=full_sha[:7],
                full_sha=full_sha,
                time=rel_time,
                message=subject,
                ref=extract_pr_ref(subject, branch),
                paths=(),
                category=classify_commit(subject),
                url=url,
            )
        )

    return PortalTelemetry(
        repo=repo,
        branch=branch,
        total_commits=total_commits,
        branch_count=branch_count,
        tag_count=tag_count,
        latest_sha=commits[0].sha,
        latest_full_sha=commits[0].full_sha,
        commits=tuple(commits),
        source="github",
    )


def _replace_once(
    text: str,
    pattern: str,
    replacement: str,
    slot_name: str,
    flags: int = 0,
) -> str:
    """Replace exactly one occurrence of ``pattern`` or raise ``ValueError``."""
    updated, count = re.subn(pattern, lambda _m: replacement, text, count=1, flags=flags)
    if count != 1:
        raise ValueError(
            f"Dashboard template drift: could not locate slot {slot_name!r}"
        )
    return updated


def _escape_js_html_text(value: str) -> str:
    """HTML-escape text destined for ``innerHTML`` template literals in JS."""
    return html.escape(value, quote=True)


def serialize_snapshot_commits_js(commits: Sequence[CommitEntry]) -> str:
    """Serialize ``snapshotCommits`` as a safe JS array literal.

    Escapes HTML entities for ``innerHTML`` safety and replaces ``</`` with
    ``<\\/`` so a commit message containing ``</script>`` cannot break out of
    the enclosing ``<script>`` tag.
    """
    payload = [
        {
            "sha": _escape_js_html_text(c.sha),
            "time": _escape_js_html_text(c.time),
            "message": _escape_js_html_text(c.message),
            "ref": _escape_js_html_text(c.ref),
            "paths": [_escape_js_html_text(p) for p in c.paths],
            "category": c.category,
            "url": _escape_js_html_text(c.url),
        }
        for c in commits
    ]
    raw_json = json.dumps(payload, indent=4, ensure_ascii=False)
    safe_json = raw_json.replace("</", "<\\/")
    indented = "\n".join(
        f"            {line}" if idx > 0 else line
        for idx, line in enumerate(safe_json.splitlines())
    )
    return (
        f"// 3. Verified Repository Snapshot Commits ({len(commits)} real hashes)\n"
        f"            const snapshotCommits = {indented};"
    )


def render_portal_html(template_html: str, telemetry: PortalTelemetry) -> str:
    """Inject ``telemetry`` into all dynamic slots of ``template_html``."""
    commit_label = "Commit" if telemetry.total_commits == 1 else "Commits"
    branch_label = "Branch" if telemetry.branch_count == 1 else "Branches"
    tag_label = "Tag" if telemetry.tag_count == 1 else "Tags"
    n_commits = len(telemetry.commits)

    out = template_html

    # 1. Hero telemetry pill: total commits
    out = _replace_once(
        out,
        r"(>\s*)\d+\s+Commits?(\s*</span>)",
        f">\n                    {telemetry.total_commits} {commit_label}\n                </span>",
        slot_name="hero_total_commits",
    )

    # 2. Hero telemetry pill: branches
    out = _replace_once(
        out,
        r"(>\s*)\d+\s+Branch(?:es)?(\s*</span>)",
        f">\n                    {telemetry.branch_count} {branch_label}\n                </span>",
        slot_name="hero_branch_count",
    )

    # 3. Hero telemetry pill: tags
    out = _replace_once(
        out,
        r"(>\s*)\d+\s+Tags?(\s*</span>)",
        f">\n                    {telemetry.tag_count} {tag_label}\n                </span>",
        slot_name="hero_tag_count",
    )

    # 4. Hero telemetry pill: latest commit link + short SHA
    latest_url = html.escape(
        f"https://github.com/{telemetry.repo}/commit/{telemetry.latest_full_sha}",
        quote=True,
    )
    out = _replace_once(
        out,
        r'<a href="https://github\.com/[^"]+/commit/[0-9a-f]{7,40}"(\s+target="_blank"[^>]*>\s*)Latest:\s*[0-9a-f]{7,40}(\s*</a>)',
        (
            f'<a href="{latest_url}"\n'
            f'                    target="_blank" rel="noopener noreferrer"\n'
            f'                    class="inline-flex items-center px-3 py-1 rounded-full text-xs font-mono bg-slate-900 hover:bg-slate-800 text-[#fb923c] border border-slate-700 transition-colors">\n'
            f"                    Latest: {html.escape(telemetry.latest_sha)}\n"
            f"                </a>"
        ),
        slot_name="hero_latest_commit",
    )

    # 5. Activity feed "All (N)" filter count
    out = _replace_once(
        out,
        r'(<span id="count-all">)\d+(</span>)',
        f'<span id="count-all">{n_commits}</span>',
        slot_name="feed_count_all",
    )

    # 6. Feed status banner: branch + total commits
    branch_escaped = html.escape(telemetry.branch)
    out = _replace_once(
        out,
        r'<span id="feed-status-text">Showing verified repository snapshot commits on <code\s+class="font-mono text-slate-800">[^<]+</code> \(\d+ total commits\)\.</span>',
        (
            f'<span id="feed-status-text">Showing verified repository snapshot commits on <code\n'
            f'                        class="font-mono text-slate-800">{branch_escaped}</code> ({telemetry.total_commits} total commits).</span>'
        ),
        slot_name="feed_status_banner",
    )

    # 7. Feed status banner: "View all N commits on GitHub" link
    commits_page_url = html.escape(
        f"https://github.com/{telemetry.repo}/commits/{telemetry.branch}/",
        quote=True,
    )
    out = _replace_once(
        out,
        r'<a href="https://github\.com/[^"]+/commits/[^"]+/" target="_blank" rel="noopener noreferrer"\s+class="font-medium text-sky-700 hover:underline">\s*View all \d+ commits on GitHub &rarr;\s*</a>',
        (
            f'<a href="{commits_page_url}" target="_blank" rel="noopener noreferrer"\n'
            f'                    class="font-medium text-sky-700 hover:underline">\n'
            f"                    View all {telemetry.total_commits} commits on GitHub &rarr;\n"
            f"                </a>"
        ),
        slot_name="feed_view_all_link",
    )

    # 8. Embedded JS snapshotCommits array
    out = _replace_once(
        out,
        r"// 3\. Verified Repository Snapshot Commits \(\d+ real hashes\)\s*\n\s*const snapshotCommits = \[.*?\];",
        serialize_snapshot_commits_js(telemetry.commits),
        slot_name="js_snapshot_commits",
        flags=re.DOTALL,
    )

    # 9. Offline fallback status line in JS
    out = _replace_once(
        out,
        r'Displaying \d+ verified repository snapshot commits on <code class="font-mono text-slate-800">[^<]+</code>\.',
        f'Displaying {n_commits} verified repository snapshot commits on <code class="font-mono text-slate-800">{branch_escaped}</code>.',
        slot_name="js_fallback_text",
    )

    return out


def resolve_telemetry(
    source: str,
    git_dir: Path,
    repo: str,
    branch: str,
    commit_limit: int,
    token: str | None,
) -> PortalTelemetry | None:
    """Fetch telemetry according to ``source`` (``None`` means keep snapshot)."""
    if source == "snapshot":
        return None

    if source == "git":
        return fetch_telemetry_from_git(
            git_dir=git_dir,
            repo=repo,
            branch=branch,
            commit_limit=commit_limit,
        )

    if source == "github":
        return fetch_telemetry_from_github(
            repo=repo,
            branch=branch,
            commit_limit=commit_limit,
            token=token,
        )

    # source == "auto"
    try:
        return fetch_telemetry_from_git(
            git_dir=git_dir,
            repo=repo,
            branch=branch,
            commit_limit=commit_limit,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as exc:
        print(
            f"Warning: local git telemetry unavailable ({exc}); trying GitHub API...",
            file=sys.stderr,
        )

    try:
        return fetch_telemetry_from_github(
            repo=repo,
            branch=branch,
            commit_limit=commit_limit,
            token=token,
        )
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"Warning: GitHub API telemetry unavailable ({exc}); preserving template snapshot.",
            file=sys.stderr,
        )
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the PHATPSY 2.0 interactive project portal HTML."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"Dashboard HTML template path (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "git", "github", "snapshot"),
        default="auto",
        help="Telemetry source (default: auto = git -> github -> snapshot)",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub owner/repo slug (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"Target branch name (default: {DEFAULT_BRANCH})",
    )
    parser.add_argument(
        "-n",
        "--commits",
        type=int,
        default=DEFAULT_COMMIT_LIMIT,
        help=f"Number of snapshot commits to embed (default: {DEFAULT_COMMIT_LIMIT})",
    )
    parser.add_argument(
        "--git-dir",
        type=Path,
        default=REPO_ROOT,
        help=f"Local git repository root for --source git/auto (default: {REPO_ROOT})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="Optional GitHub API token (defaults to $GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve telemetry and validate rendering without writing the output file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.commits < 1:
        print("Error: --commits must be >= 1", file=sys.stderr)
        return 2

    template_html = args.template.read_text(encoding="utf-8")

    try:
        telemetry = resolve_telemetry(
            source=args.source,
            git_dir=args.git_dir,
            repo=args.repo,
            branch=args.branch,
            commit_limit=args.commits,
            token=args.token,
        )
    except Exception as exc:  # noqa: BLE001 — explicit CLI failure
        print(f"Error: failed to fetch telemetry ({args.source}): {exc}", file=sys.stderr)
        return 1

    if telemetry is None:
        rendered_html = template_html
        source_label = "snapshot"
    else:
        rendered_html = render_portal_html(template_html, telemetry)
        source_label = telemetry.source

    if args.dry_run:
        if telemetry is None:
            print(f"[dry-run] source=snapshot bytes={len(rendered_html.encode('utf-8'))}")
        else:
            print(
                f"[dry-run] source={source_label} repo={telemetry.repo} "
                f"branch={telemetry.branch} commits={telemetry.total_commits} "
                f"branches={telemetry.branch_count} tags={telemetry.tag_count} "
                f"latest={telemetry.latest_sha} embedded={len(telemetry.commits)} "
                f"bytes={len(rendered_html.encode('utf-8'))}"
            )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered_html, encoding="utf-8")

    if telemetry is None:
        print(
            f"Generated {args.output} ({len(rendered_html.encode('utf-8'))} bytes, "
            f"source=snapshot)"
        )
    else:
        print(
            f"Generated {args.output} ({len(rendered_html.encode('utf-8'))} bytes, "
            f"source={source_label}, commits={telemetry.total_commits}, "
            f"branches={telemetry.branch_count}, tags={telemetry.tag_count}, "
            f"latest={telemetry.latest_sha})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
