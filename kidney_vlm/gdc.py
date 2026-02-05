import json
from typing import Any, Dict, Iterable, List

import requests


GDC_API_ROOT = "https://api.gdc.cancer.gov"


def gdc_post(endpoint: str, payload: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    """POST JSON to a GDC API endpoint and return parsed JSON."""
    url = endpoint
    if not endpoint.startswith("http"):
        url = f"{GDC_API_ROOT.rstrip('/')}/{endpoint.lstrip('/')}"

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def iter_gdc_files(
    filters: Dict[str, Any],
    fields: List[str],
    page_size: int = 500,
    timeout_s: int = 60,
) -> Iterable[Dict[str, Any]]:
    """Iterate over /files hits with pagination (from + size)."""
    fields_str = ",".join(fields)
    start = 0

    while True:
        payload = {
            "filters": filters,
            "fields": fields_str,
            "format": "JSON",
            "size": str(page_size),
            "from": str(start),
        }
        out = gdc_post("/files", payload, timeout_s=timeout_s)
        hits = out.get("data", {}).get("hits", [])
        if not hits:
            break

        for h in hits:
            yield h

        start += len(hits)

        # Stop if we've reached the reported total.
        total = out.get("data", {}).get("pagination", {}).get("total")
        if total is not None and start >= int(total):
            break


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)
