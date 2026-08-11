"""Manual entrypoint for the fixed Europe PMC public-OA release smoke."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from paper_agent.smoke import run_public_oa_download_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("PAPER_AGENT_RUN_LIVE_SMOKE") != "1":
        parser.error("set PAPER_AGENT_RUN_LIVE_SMOKE=1 to permit the public OA smoke")
    contact = os.environ.get("PAPER_AGENT_SMOKE_CONTACT")
    email = os.environ.get("PAPER_AGENT_SMOKE_UNPAYWALL_EMAIL")
    if not contact or not email:
        parser.error("PAPER_AGENT_SMOKE_CONTACT and PAPER_AGENT_SMOKE_UNPAYWALL_EMAIL are required")
    result = run_public_oa_download_smoke(
        args.output_dir,
        contact=contact,
        unpaywall_email=email,
        source_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[1],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip(),
    )
    print(result.evidence_path)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
