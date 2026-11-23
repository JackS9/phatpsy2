"""Unit tests for tools/generate_portal.py."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "tools" / "generate_portal.py"
TEMPLATE_PATH = REPO_ROOT / "docs" / "dashboard" / "index.html"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_portal", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gp = _load_module()


def test_parse_total_count_from_link_header_per_page_1() -> None:
    link = (
        '<https://api.github.com/repositories/1/commits?sha=master&per_page=1&page=2>; rel="next", '
        '<https://api.github.com/repositories/1/commits?sha=master&per_page=1&page=277>; rel="last"'
    )
    assert gp.parse_total_count_from_link_header(link, page_items=1, per_page=1) == 277


def test_parse_total_count_from_link_header_no_header() -> None:
    assert gp.parse_total_count_from_link_header(None, page_items=0, per_page=1) == 0
    assert gp.parse_total_count_from_link_header(None, page_items=3, per_page=1) == 3


def test_extract_pr_ref_and_summarize_paths() -> None:
    assert (
        gp.extract_pr_ref(
            "Merge pull request #12 from JackS9/docs/dashboard-landing", "master"
        )
        == "PR #12 · docs/dashboard-landing"
    )
    assert gp.extract_pr_ref("feat: add portal generator", "master") == "master"
    assert gp.summarize_paths(
        ["docs/dashboard/index.html", "docs/assets/logo.svg", "AGENTS.md", "tools/a.py"]
    ) == ("docs/", "AGENTS.md", "tools/")


def test_classify_commit() -> None:
    assert gp.classify_commit("docs: update dashboard") == "docs"
    assert gp.classify_commit("Merge pull request #12 from JackS9/x") == "docs"
    assert gp.classify_commit("feat(vb): add valence-bond assembler") == "vb"
    assert gp.classify_commit("fix: SCF gradient convergence diagnostics") == "fortran"
    assert gp.classify_commit("chore: update CMake build flags") == "build"
    assert gp.classify_commit("update files", paths=("docs/", "ROADMAP.md")) == "docs"


def test_snapshot_mode_byte_for_byte_parity(tmp_path: Path) -> None:
    out_file = tmp_path / "portal_snapshot.html"
    rc = gp.main(
        [
            "--source",
            "snapshot",
            "--template",
            str(TEMPLATE_PATH),
            "-o",
            str(out_file),
        ]
    )
    assert rc == 0
    assert out_file.read_bytes() == TEMPLATE_PATH.read_bytes()


def test_render_portal_html_updates_all_slots_and_escapes_xss() -> None:
    template_html = TEMPLATE_PATH.read_text(encoding="utf-8")
    malicious_msg = 'fix: handle </script><img src=x onerror="alert(1)"> safely'
    telemetry = gp.PortalTelemetry(
        repo="JackS9/phatpsy2",
        branch="master",
        total_commits=280,
        branch_count=9,
        tag_count=2,
        latest_sha="abcdef1",
        latest_full_sha="abcdef1234567890abcdef1234567890abcdef12",
        commits=(
            gp.CommitEntry(
                sha="abcdef1",
                full_sha="abcdef1234567890abcdef1234567890abcdef12",
                time="2 hours ago",
                message=malicious_msg,
                ref="master",
                paths=("tools/", "tests/"),
                category="build",
                url="https://github.com/JackS9/phatpsy2/commit/abcdef1234567890abcdef1234567890abcdef12",
            ),
        ),
        source="git",
    )

    rendered = gp.render_portal_html(template_html, telemetry)

    assert "280 Commits" in rendered
    assert "9 Branches" in rendered
    assert "2 Tags" in rendered
    assert "Latest: abcdef1" in rendered
    assert (
        "https://github.com/JackS9/phatpsy2/commit/abcdef1234567890abcdef1234567890abcdef12"
        in rendered
    )
    assert '<span id="count-all">1</span>' in rendered
    assert "(280 total commits)" in rendered
    assert "View all 280 commits on GitHub &rarr;" in rendered
    assert "Verified Repository Snapshot Commits (1 real hashes)" in rendered
    assert "Displaying 1 verified repository snapshot commits on" in rendered

    # Ensure XSS payload cannot break out of <script> or inject raw HTML
    assert "</script><img" not in rendered
    assert "&lt;/script&gt;&lt;img" in rendered
    assert rendered.count("</script>") == template_html.count("</script>")

    # Core dashboard invariants remain intact
    assert "examples/n2.yaml" in rendered
    assert "examples/N2.yaml" not in rendered


def test_render_portal_html_raises_on_template_drift() -> None:
    telemetry = gp.PortalTelemetry(
        repo="JackS9/phatpsy2",
        branch="master",
        total_commits=10,
        branch_count=1,
        tag_count=0,
        latest_sha="1234567",
        latest_full_sha="1234567" * 5 + "12345",
        commits=(),
        source="git",
    )
    with pytest.raises(ValueError, match="Dashboard template drift"):
        gp.render_portal_html("<html><body>broken</body></html>", telemetry)


def test_format_relative_time() -> None:
    now = dt.datetime(2026, 11, 23, 22, 0, 0, tzinfo=dt.timezone.utc)
    assert gp.format_relative_time(now - dt.timedelta(minutes=1), now=now) == "1 min ago"
    assert gp.format_relative_time(now - dt.timedelta(minutes=25), now=now) == "25 mins ago"
    assert gp.format_relative_time(now - dt.timedelta(hours=3), now=now) == "3 hours ago"
    assert gp.format_relative_time(now - dt.timedelta(days=1, hours=2), now=now) == "yesterday"
    assert gp.format_relative_time(now - dt.timedelta(days=5), now=now) == "5 days ago"
