#!/usr/bin/env python3
"""Write a deterministic SplaTAM container qualification manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["rocm", "cuda"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--gpu-arch", default="")
    parser.add_argument("--status", required=True)
    parser.add_argument("--amd-port-audit-sha256", default="")
    parser.add_argument("--base-image-digest", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not OCI_DIGEST_RE.fullmatch(args.digest):
        raise SystemExit("--digest must be an immutable OCI sha256 digest")
    if args.base_image_digest and not OCI_DIGEST_RE.fullmatch(args.base_image_digest):
        raise SystemExit("--base-image-digest must be an OCI sha256 digest")
    if args.backend == "rocm" and not SHA256_RE.fullmatch(args.amd_port_audit_sha256):
        raise SystemExit("ROCm qualification requires canonical AMD/MOAT audit SHA-256")

    payload = {
        "schema_version": "researchflow-container-qualification/v1",
        "method": "splatam",
        "backend": args.backend,
        "image": args.image,
        "digest": args.digest,
        "base_image_digest": args.base_image_digest or None,
        "gpu_arch": args.gpu_arch or None,
        "status": args.status,
        "amd_port_audit_sha256": args.amd_port_audit_sha256 or None,
        "amd_moat_revision": (
            "f69bb67d70e7a47af00095dab7029c230a30be73" if args.backend == "rocm" else None
        ),
        "upstream_revision": "da6bbcd24c248dc884ac7f49d62e91b841b26ccc",
        "depth_rasterizer_revision": "cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110",
        "amd_gaussian_splatting_reference_revision": (
            "dfb1dd1cbcc508e08c7395b27953620121dad1fc" if args.backend == "rocm" else None
        ),
        "github": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "sha": os.getenv("GITHUB_SHA"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        },
        "capabilities": ["depth_rasterizer_forward_backward_gpu"],
    }
    identity = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["qualification_id"] = hashlib.sha256(identity).hexdigest()
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
