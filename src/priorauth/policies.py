"""Load coverage policies from YAML files and check them before anything uses them.

Policies are data, not code: adding a service means adding a file to policies/. Each policy cites its
coverage document, whose extracted text lives in policies/sources/, and every criterion quotes that
text. Every rule here runs at load time, so a bad policy file fails at startup instead of in the middle
of a request.
"""

import hashlib
import re
from itertools import dropwhile
from pathlib import Path

import yaml
from pydantic import ValidationError

from priorauth.ingest import normalize
from priorauth.schemas import Policy

POLICY_ID = re.compile(r"[a-z0-9_]+")
SOURCE_HEADER = "# source: "


class PolicyError(ValueError):
    """A policy file is malformed or breaks one of the corpus rules."""


def load_policy(path: Path) -> Policy:
    """Load one policy file and enforce the rules that keep policies from colliding."""
    try:
        policy = Policy.model_validate(yaml.safe_load(path.read_text()))
    except (yaml.YAMLError, ValidationError) as e:
        raise PolicyError(f"{path.name}: {e}") from e

    pid = policy.policy_id
    if not POLICY_ID.fullmatch(pid):
        raise PolicyError(f"{path.name}: policy_id {pid!r} must use only lowercase letters, digits and _")
    if pid != path.stem:
        raise PolicyError(f"{path.name}: policy_id {pid!r} must match the file name")

    ids = [c.criterion_id for c in policy.criteria]
    if len(ids) != len(set(ids)):
        raise PolicyError(f"{path.name}: criterion IDs must be unique")
    if bad := [cid for cid in ids if not re.fullmatch(rf"{pid}\.[a-z0-9_]+", cid)]:
        raise PolicyError(f"{path.name}: criterion IDs must look like '{pid}.<name>': {bad}")
    return policy


def load_sources(directory: Path) -> dict[str, str]:
    """Normalized text of each file in directory/sources, keyed by the URL in its '# source:' header."""
    sources = {}
    for path in sorted((directory / "sources").glob("*.txt")):
        lines = path.read_text().splitlines()
        if not lines or not lines[0].startswith(SOURCE_HEADER):
            raise PolicyError(f"sources/{path.name}: first line must be '{SOURCE_HEADER}<url>'")
        url = lines[0].removeprefix(SOURCE_HEADER).strip()
        if url in sources:
            raise PolicyError(f"sources/{path.name}: another source file already has {url}")
        sources[url] = normalize("\n".join(dropwhile(lambda line: line.startswith("#"), lines)))
    return sources


def check_quotes(policy: Policy, sources: dict[str, str]) -> None:
    """Every criterion must quote the policy's coverage document, and every quote must appear in it."""
    pid = policy.policy_id
    if policy.source not in sources:
        raise PolicyError(f"{pid}: no extracted text in sources/ for {policy.source}")
    if missing := [c.criterion_id for c in policy.criteria if not c.source_quotes]:
        raise PolicyError(f"{pid}: criteria without source_quotes: {missing}")
    text = sources[policy.source]
    if bad := [(c.criterion_id, q) for c in policy.criteria for q in c.source_quotes if normalize(q) not in text]:
        raise PolicyError(f"{pid}: quotes not found in the source text: {bad}")


def load_policies(directory: Path) -> dict[str, Policy]:
    """Load every *.yaml file in a directory, keyed by policy_id, and check its quotes.

    Policy IDs must equal their file names, so the directory itself guarantees they are unique, and
    criterion IDs are prefixed with their policy ID, so they are unique across the whole corpus.
    """
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise PolicyError(f"no policy files in {directory}")
    sources = load_sources(directory)
    policies = {}
    for policy in map(load_policy, paths):
        check_quotes(policy, sources)
        policies[policy.policy_id] = policy
    return policies


def fingerprint(policy: Policy) -> str:
    """Short hash of the policy's content, recorded in the audit trail.

    It hashes the validated policy, not the raw file, so it changes when a criterion's text, quotes or
    required flag changes but not when the YAML is only reformatted or re-commented.
    """
    return hashlib.sha256(policy.model_dump_json().encode()).hexdigest()[:16]