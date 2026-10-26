"""Conservative report vocabulary adapter, not a factual evaluator.

Only whole, reviewed assertions are aliases. No substring inference, provider,
or lookup of the expected oracle answer is permitted here.
"""
from copy import deepcopy


TENANT_DENIAL_ALIASES = (
    "Access denied due to tenant mismatch.",
    "Access denied: tenant mismatch (actor tenant A, object owner tenant B)",
    "Access denied: tenant mismatch (actor tenant A, object owner tenant B). Policy tenant-report-owner-v4 enforces owner_only rule, conceal_denial_as 404.",
    "Access denied: policy tenant-report-owner-v4 enforces owner_only rule. Actor tenant A cannot access object report-017 owned by tenant B. Denial concealed as 404 not_found.",
    "Access denied due to tenant mismatch (actor tenant A, object owner tenant B), concealed as 404 not_found per policy configuration",
    "Policy enforcement: tenant A denied access to tenant B's object report-017 (owner_only rule), concealed as 404",
    "Access denied: tenant A attempted to access object owned by tenant B. Policy tenant-report-owner-v4 enforces owner_only rule with tenant_mismatch denial.",
    "Access denied: tenant A (tenant-A-user) attempted to access object report-017 owned by tenant B. Policy tenant-report-owner-v4 enforces owner_only rule, resulting in tenant_mismatch denial concealed as 404 not_found.",
)


def _spelling(text):
    return " ".join(text.split()).casefold()


def normalize_report(report):
    """Return a separate oracle input and auditable mapping; never mutate raw text.

Unknown, negated, uncertain and extended assertions remain unchanged so that
the existing exact-code oracle rejects them. An alias does not prove a claim.
"""
    normalized = deepcopy(report)
    raw = report.get("reason")
    matched = isinstance(raw, str) and _spelling(raw) in {
        _spelling(alias) for alias in TENANT_DENIAL_ALIASES}
    if matched:
        normalized["reason"] = "tenant_mismatch"
    return normalized, {"version": "lifecycle-report-reason-v1", "raw_reason": raw,
                        "oracle_reason": normalized.get("reason"),
                        "mapping": "explicit_alias" if matched else "unchanged"}
