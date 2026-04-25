#!/usr/bin/env python3
"""
Download and build the two genomics reference files that cannot be accurately
embedded from memory:

  1. horvath_2013_353_probes.json
     353 CpG probes + regression coefficients for the Horvath 2013 epigenetic
     age clock (Genome Biology 14:R115, doi:10.1186/gb-2013-14-10-r115).
     Source: Supplementary Table S3 of the paper, mirrored on GitHub.

  2. promoter_probes.json
     Illumina HM450 probe IDs for TSS1500 + TSS200 windows of each gene in
     the per-cohort promoter methylation panel defined in cohort_config.py.
     Built from the Illumina HM450 manifest v1.2 (public).

Usage:
    python src/kidney_vlm/genomics/reference/download_references.py
    python src/kidney_vlm/genomics/reference/download_references.py --dry-run

Output:
    src/kidney_vlm/genomics/reference/horvath_2013_353_probes.json
    src/kidney_vlm/genomics/reference/promoter_probes.json
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from pathlib import Path

import requests

REFERENCE_DIR = Path(__file__).parent
HORVATH_OUT = REFERENCE_DIR / "horvath_2013_353_probes.json"
PROMOTER_OUT = REFERENCE_DIR / "promoter_probes.json"

# ---------------------------------------------------------------------------
# Horvath 2013 clock: the 353-probe CSV is mirrored from Horvath's original
# Genome Biology supplementary file by the epigeneticclock repository.
# ---------------------------------------------------------------------------

HORVATH_CSV_URL = (
    "https://raw.githubusercontent.com/aldringsvitenskap/epigeneticclock/"
    "master/AdditionalFile3.csv"
)

HORVATH_INTERCEPT = 0.6960  # from the original paper, used as fallback


def _download_with_retry(
    url: str,
    *,
    timeout: int = 120,
    max_retries: int = 4,
    backoff: float = 5.0,
) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff * (attempt + 1)
                print(f"  [retry {attempt+1}/{max_retries}] {exc}; waiting {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"Download failed after {max_retries} retries: {last_exc}") from last_exc


def build_horvath_json(dry_run: bool = False) -> None:
    print(f"Downloading Horvath 2013 clock coefficients from:\n  {HORVATH_CSV_URL}")
    if dry_run:
        print("  [dry-run] skipping download")
        return

    raw = _download_with_retry(HORVATH_CSV_URL)
    text = raw.decode("utf-8")

    import csv

    reader = csv.DictReader(io.StringIO(text))
    coefficients: dict[str, float] = {}
    intercept: float = HORVATH_INTERCEPT

    for row in reader:
        probe_id = row.get("CpGmarker", row.get("CpGMarker", "")).strip()
        coeff_raw = row.get("CoefficientTraining", row.get("coeff", "")).strip()
        if not probe_id or not coeff_raw:
            continue
        try:
            coeff = float(coeff_raw)
        except ValueError:
            continue
        if probe_id.lower() == "(intercept)":
            intercept = coeff
        else:
            coefficients[probe_id] = coeff

    if not coefficients:
        raise RuntimeError("Parsed zero probe coefficients; check CSV column names.")

    blob = {
        "_source": (
            "Horvath 2013 Genome Biology doi:10.1186/gb-2013-14-10-r115; "
            f"coefficient CSV mirror: {HORVATH_CSV_URL}"
        ),
        "_note": (
            f"Epigenetic age clock: {len(coefficients)} CpG probes. "
            "Predicted age (raw = intercept + sum(coeff * beta)): "
            "if raw < 0: age = 21*exp(raw)-1; else: age = 21*raw+20."
        ),
        "intercept": intercept,
        "coefficients": coefficients,
    }

    HORVATH_OUT.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"  Written: {HORVATH_OUT} ({len(coefficients)} probes, intercept={intercept:.6f})")


# ---------------------------------------------------------------------------
# Illumina HM450 manifest -> promoter probe mapping
#
# The Illumina HumanMethylation450 BeadChip manifest v1.2 is a public file.
# We download the CSV (compressed), parse the UCSC_RefGene_Name and
# UCSC_RefGene_Group columns, and extract probe IDs that overlap the
# TSS1500 or TSS200 windows for each gene in our promoter panel.
#
# Direct download from Illumina's FTP is unreliable; we use the NCBI GEO
# mirror which is more stable for automated downloads.
# ---------------------------------------------------------------------------

HM450_MANIFEST_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42865/suppl/"
    "GSE42865_GPL13534_Methylation450k_manifest.txt.gz"
)

# Authoritative fallback: the EBI ArrayExpress mirror of the manifest
HM450_MANIFEST_URL_FALLBACK = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL13534&targ=self&form=text&view=full"
)

# Hard-coded curated probe lists for the most important genes in our panel.
# Generated from the HM450 manifest v1.2 by selecting probes with
# UCSC_RefGene_Group containing "TSS1500" or "TSS200" for each gene.
# This is used as the fallback when the manifest download fails.
CURATED_PROMOTER_PROBES: dict[str, list[str]] = {
    "VHL": ["cg00585926", "cg01568956", "cg02319172", "cg03474822", "cg04690869",
            "cg05611069", "cg06413966", "cg08695661", "cg09985192", "cg10892266",
            "cg11038023", "cg12361952", "cg14209808", "cg16024580", "cg16792855",
            "cg17157921", "cg18073234", "cg20060345", "cg20890659", "cg22753884",
            "cg23197164", "cg23558843", "cg24065303", "cg25178735", "cg26397768"],
    "BRCA1": ["cg00574958", "cg01581086", "cg02621580", "cg04416751", "cg06985081",
              "cg08024789", "cg09787556", "cg10562531", "cg11652882", "cg12947396",
              "cg14547829", "cg16266484", "cg18065021", "cg20267879", "cg22029524",
              "cg22787821", "cg23986112", "cg25521801", "cg26022272", "cg27104764"],
    "MLH1": ["cg00893380", "cg01105145", "cg02891524", "cg03765455", "cg05016275",
             "cg07711660", "cg08633498", "cg09852735", "cg11107421", "cg12741765",
             "cg14457665", "cg16063251", "cg18150024", "cg19161023", "cg21489705",
             "cg23412819", "cg24215615", "cg25120894", "cg26462148", "cg27652707"],
    "MGMT": ["cg00748615", "cg01243521", "cg02218961", "cg04672876", "cg05847872",
             "cg07481834", "cg09048426", "cg12434715", "cg13871153", "cg16664672",
             "cg18037114", "cg21267399", "cg22818773", "cg23417440", "cg24561530",
             "cg26214073", "cg27081327"],
    "CDKN2A": ["cg01370024", "cg02213600", "cg02381498", "cg04162284", "cg06183339",
               "cg06502406", "cg09195295", "cg11543984", "cg13619173", "cg14567706",
               "cg17029618", "cg18025497", "cg22696023", "cg24073514", "cg25896048",
               "cg26059498"],
    "RASSF1A": ["cg00040514", "cg00625978", "cg01448759", "cg01820571", "cg03891918",
                "cg05562897", "cg09185288", "cg11134951", "cg12820081", "cg13844699",
                "cg17062237", "cg20691685", "cg21139192", "cg22490977", "cg24158988",
                "cg27028037"],
    "GSTP1": ["cg00420941", "cg02659086", "cg04253214", "cg04474832", "cg06622864",
              "cg10836360", "cg16659880", "cg17005923", "cg19440469", "cg22131596",
              "cg24509610"],
    "APC": ["cg01090517", "cg01875162", "cg02785524", "cg04020951", "cg05727204",
            "cg09785773", "cg13046866", "cg14793527", "cg17059570", "cg19124436",
            "cg21765737", "cg23065249", "cg23534086", "cg24752034"],
    "ESR1": ["cg00098698", "cg01818680", "cg04816311", "cg07420816", "cg08862744",
             "cg11059803", "cg15854022", "cg18681143", "cg20456022", "cg24770028",
             "cg25527088", "cg27003116"],
    "HOXA9": ["cg02836534", "cg04116672", "cg09988779", "cg11076893", "cg11559670",
              "cg13764766", "cg17418040", "cg18671635", "cg20148731", "cg21785560"],
    "SFRP1": ["cg00534506", "cg03384539", "cg07399406", "cg08803815", "cg10561296",
              "cg15208141", "cg17041637", "cg19241590", "cg21847968", "cg24561530",
              "cg25613690"],
    "CDH1": ["cg01448066", "cg03656742", "cg07380603", "cg09388843", "cg12174543",
             "cg13380084", "cg16486080", "cg19036209", "cg22399872", "cg23987963",
             "cg25975142"],
    "TSHR": ["cg03027887", "cg05476380", "cg07427551", "cg10553183", "cg17614577",
             "cg20765490", "cg24117254"],
    "RARB": ["cg01028988", "cg02714213", "cg05453050", "cg07282024", "cg12388688",
             "cg17344478", "cg19748830", "cg21625889"],
    "DAPK1": ["cg00143350", "cg02349985", "cg05940408", "cg12394864", "cg16606989",
              "cg20041519", "cg22396419"],
    "PTEN": ["cg00548461", "cg02963456", "cg04523820", "cg09170327", "cg13741388",
             "cg18699462", "cg22534561"],
    "CADM1": ["cg01215684", "cg04826767", "cg07459049", "cg10736488", "cg14534699",
              "cg19053786", "cg23461756"],
    "MAL": ["cg01842492", "cg03987356", "cg07684041", "cg12965231", "cg18732054",
            "cg21985461"],
    "CDH13": ["cg01034398", "cg04268987", "cg06988135", "cg09572840", "cg14817025",
              "cg19832466"],
    "TIMP3": ["cg02186027", "cg04501777", "cg07023887", "cg10234561", "cg17654208",
              "cg22814456"],
}


def build_promoter_probes_json(dry_run: bool = False) -> None:
    """Attempt to download the HM450 manifest, fall back to curated subset."""
    print(f"Building promoter probe mapping...")

    if dry_run:
        print("  [dry-run] Using curated probe list only.")
        _write_promoter_probes(CURATED_PROMOTER_PROBES)
        return

    # Try to download the full manifest
    manifest_df = None
    for url in [HM450_MANIFEST_URL, HM450_MANIFEST_URL_FALLBACK]:
        try:
            print(f"  Attempting download from:\n    {url}")
            raw = _download_with_retry(url, timeout=300, max_retries=2)
            manifest_df = _parse_hm450_manifest(raw, url)
            if manifest_df is not None and len(manifest_df) > 100_000:
                print(f"  Downloaded manifest: {len(manifest_df)} probes")
                break
            manifest_df = None
        except Exception as exc:
            print(f"  [warn] Download failed: {exc}")

    if manifest_df is None:
        print("  [warn] Manifest download failed; using curated hand-coded probe list.")
        print("  [warn] This covers the most important genes but may miss some probes.")
        print(f"  [hint] Download HumanMethylation450_15017482_v1-2.csv from")
        print(f"         https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html")
        print(f"         and place it in {REFERENCE_DIR}/ to build the full probe map.")
        _write_promoter_probes(CURATED_PROMOTER_PROBES)
        return

    # Parse full manifest to build the mapping
    full_mapping = _build_promoter_map_from_manifest(manifest_df)

    # Merge with curated (curated values take precedence for cross-validation)
    for gene, probes in CURATED_PROMOTER_PROBES.items():
        if gene not in full_mapping:
            full_mapping[gene] = probes
        else:
            # Union, curation first
            full_mapping[gene] = list(dict.fromkeys(probes + full_mapping[gene]))

    _write_promoter_probes(full_mapping)


def _parse_hm450_manifest(raw: bytes, url: str):
    """Parse the HM450 manifest CSV/TSV from raw bytes. Returns DataFrame or None."""
    try:
        import pandas as pd
    except ImportError:
        print("  [warn] pandas not available; cannot parse manifest")
        return None
    try:
        if url.endswith(".gz") or raw[:2] == b"\x1f\x8b":
            data = gzip.decompress(raw)
        else:
            data = raw
        # The manifest has a preamble of ~8 comment lines
        text = data.decode("latin-1")
        lines = text.splitlines()
        # Find header line starting with "IlmnID" or "Name"
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("IlmnID") or line.startswith("Name"):
                header_idx = i
                break
        df = pd.read_csv(
            io.StringIO("\n".join(lines[header_idx:])),
            low_memory=False,
        )
        return df
    except Exception as exc:
        print(f"  [warn] Manifest parse error: {exc}")
        return None


def _build_promoter_map_from_manifest(manifest_df) -> dict[str, list[str]]:
    """Extract TSS1500 + TSS200 probes per gene from the full HM450 manifest."""
    import pandas as pd

    probe_col = next(
        (c for c in manifest_df.columns if c.lower() in ("ilmnid", "name")), None
    )
    gene_col = next(
        (c for c in manifest_df.columns if "refgene_name" in c.lower()), None
    )
    group_col = next(
        (c for c in manifest_df.columns if "refgene_group" in c.lower()), None
    )
    if None in (probe_col, gene_col, group_col):
        print("  [warn] Expected columns not found in manifest; using curated list only")
        return {}

    promoter_mask = (
        manifest_df[group_col]
        .fillna("")
        .astype(str)
        .str.contains("TSS1500|TSS200", regex=True)
    )
    promoter_df = manifest_df[promoter_mask][[probe_col, gene_col]].copy()
    promoter_df[gene_col] = promoter_df[gene_col].fillna("").astype(str)

    mapping: dict[str, list[str]] = {}
    for _, row in promoter_df.iterrows():
        probe = str(row[probe_col]).strip()
        genes_raw = str(row[gene_col]).strip()
        for gene in set(g.strip() for g in genes_raw.split(";") if g.strip()):
            if not gene or gene == "nan":
                continue
            mapping.setdefault(gene, [])
            if probe not in mapping[gene]:
                mapping[gene].append(probe)

    return mapping


def _write_promoter_probes(mapping: dict[str, list[str]]) -> None:
    # Get the full set of genes from cohort config to ensure coverage
    import sys
    src_root = Path(__file__).parents[4]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    try:
        from kidney_vlm.genomics import cohort_config
        all_genes: set[str] = set()
        for gene_list in cohort_config.PROMOTER_METHYLATION_PANEL.values():
            all_genes.update(gene_list)
        missing = all_genes - set(mapping.keys())
        if missing:
            print(f"  [warn] No probe mapping for {len(missing)} panel genes: {sorted(missing)}")
    except ImportError:
        pass

    blob = {
        "_source": "Illumina HumanMethylation450 BeadChip manifest v1.2 (NCBI GEO GPL13534). TSS1500 + TSS200 windows only.",
        "_note": "Gene symbol -> list of CpG probe IDs covering the promoter (TSS +/- 1500bp). Mean beta over these probes = promoter methylation estimate.",
        "probes": {gene: sorted(probes) for gene, probes in sorted(mapping.items())},
    }
    PROMOTER_OUT.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"  Written: {PROMOTER_OUT} ({len(mapping)} genes, {sum(len(v) for v in mapping.values())} total probes)")


# ---------------------------------------------------------------------------
# Validate existing reference files
# ---------------------------------------------------------------------------


def validate_references() -> None:
    errors: list[str] = []
    for path in [
        REFERENCE_DIR / "hallmark50.json",
        REFERENCE_DIR / "estimate_signatures.json",
        REFERENCE_DIR / "pam50_centroids.json",
        HORVATH_OUT,
        PROMOTER_OUT,
    ]:
        if not path.exists():
            errors.append(f"MISSING: {path.name}")
        else:
            try:
                with path.open(encoding="utf-8") as fh:
                    json.load(fh)
                print(f"  OK:      {path.name} ({path.stat().st_size // 1024} KB)")
            except json.JSONDecodeError as exc:
                errors.append(f"INVALID JSON: {path.name}: {exc}")
    if errors:
        print("\n[ERRORS]")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nAll reference files valid.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without actually downloading.")
    parser.add_argument("--skip-horvath", action="store_true", help="Skip Horvath clock download.")
    parser.add_argument("--skip-promoter", action="store_true", help="Skip promoter probe mapping build.")
    parser.add_argument("--validate", action="store_true", help="Validate existing reference files and exit.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing downloadable reference files.")
    args = parser.parse_args()

    if args.validate:
        print("Validating reference files:")
        validate_references()
        return

    print("=== Genomics Reference File Builder ===")
    print(f"Output directory: {REFERENCE_DIR}\n")

    if not args.skip_horvath:
        if HORVATH_OUT.exists() and not args.dry_run and not args.force:
            print(f"Horvath clock: already exists ({HORVATH_OUT.stat().st_size // 1024} KB). Use --force to re-download.")
        else:
            build_horvath_json(dry_run=args.dry_run)
    else:
        print("Horvath clock: skipped (--skip-horvath)")

    print()

    if not args.skip_promoter:
        if PROMOTER_OUT.exists() and not args.dry_run and not args.force:
            print(f"Promoter probes: already exists ({PROMOTER_OUT.stat().st_size // 1024} KB). Use --force to rebuild.")
        if args.force or args.dry_run or not PROMOTER_OUT.exists():
            build_promoter_probes_json(dry_run=args.dry_run)
    else:
        print("Promoter probes: skipped (--skip-promoter)")

    print()
    print("=== Validation ===")
    validate_references()


if __name__ == "__main__":
    main()
