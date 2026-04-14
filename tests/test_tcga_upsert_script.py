from __future__ import annotations

import importlib.util
import io
import json
from functools import lru_cache
from pathlib import Path
import sys
import zipfile

import pytest
from omegaconf import OmegaConf

from kidney_vlm.data.sources import tcga as tcga_module
from kidney_vlm.data.sources.tcga import GDCClient, TCIAClient


@lru_cache(maxsize=1)
def _load_script_module():
    pytest.importorskip("hydra")
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "data" / "01_upsert_tcga_registry_rows.py"
    spec = importlib.util.spec_from_file_location("tcga_upsert_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tcga_cfg(*, restrict_to_radiology_cases: bool, fetch_series_metadata: bool = True):
    return OmegaConf.create(
        {
            "upsert_mode": "replace",
            "patient_subset_ids": [],
            "patient_chunk": {"index": 0, "size": None},
            "gdc": {
                "max_cases": None,
                "pathology_data_formats": ["SVS"],
                "pathology_data_types": ["Slide Image"],
                "max_pathology_files": None,
                "report_data_formats": ["PDF"],
                "report_data_types": ["Pathology Report"],
                "report_data_categories": ["Clinical"],
                "max_report_files": None,
                "fetch_ssm_mutations": False,
                "mutation_gene_panel": ["VHL"],
            },
            "tcia": {
                "enabled": True,
                "restrict_to_radiology_cases": restrict_to_radiology_cases,
                "fetch_series_metadata": fetch_series_metadata,
                "qualifying_modalities": ["CT"],
                "max_studies_per_collection": None,
                "max_series_per_study_metadata": None,
            },
        }
    )


def _make_studies() -> dict[str, list[dict[str, str]]]:
    return {
        "TCGA-AA-0001": [
            {
                "collection": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "study_instance_uid": "1.2.3",
                "modalities_in_study": ["CT", "SEG"],
            }
        ]
    }


def test_fetch_tcga_payloads_does_not_filter_gdc_cases_when_radiology_restriction_disabled() -> None:
    module = _load_script_module()

    class FakeGDCClient:
        def __init__(self) -> None:
            self.fetch_cases_submitter_ids = "unset"

        def fetch_cases(self, project_ids, *, submitter_ids=None, fields=None, max_cases=None):
            del project_ids, fields, max_cases
            self.fetch_cases_submitter_ids = submitter_ids
            return [{"case_id": "case-1", "submitter_id": "TCGA-BB-0002"}]

        def fetch_pathology_files(self, project_ids, *, case_ids=None, submitter_ids=None, data_formats=None, data_types=None, fields=None, max_files=None):
            del project_ids, data_formats, data_types, fields, max_files
            assert case_ids == ["case-1"]
            assert submitter_ids == ["TCGA-BB-0002"]
            return []

        def fetch_report_files(self, *, project_ids, case_ids=None, data_formats=None, data_types=None, data_categories=None, fields=None, max_files=None):
            del project_ids, case_ids, data_formats, data_types, data_categories, fields, max_files
            return []

    class FakeTCIAClient:
        def fetch_studies_by_patient(self, collections, max_studies_per_collection=None):
            del collections, max_studies_per_collection
            return _make_studies()

        def fetch_series_by_patient(self, studies_by_patient, max_series_per_study=None):
            del studies_by_patient, max_series_per_study
            return {
                "TCGA-AA-0001": [
                    {
                        "collection": "TCGA-KIRC",
                        "patient_id": "TCGA-AA-0001",
                        "study_instance_uid": "1.2.3",
                        "series_instance_uid": "1.2.3.4",
                        "modality": "CT",
                    }
                ]
            }

    fake_gdc = FakeGDCClient()
    module._fetch_tcga_payloads(
        tcga_cfg=_tcga_cfg(restrict_to_radiology_cases=False),
        project_ids=["TCGA-KIRC"],
        gdc_client=fake_gdc,
        tcia_client=FakeTCIAClient(),
        tcia_collections=["TCGA-KIRC"],
    )

    assert fake_gdc.fetch_cases_submitter_ids is None


def test_fetch_tcga_payloads_raises_when_restricted_radiology_cohort_is_empty() -> None:
    module = _load_script_module()

    class FakeGDCClient:
        def fetch_cases(self, *args, **kwargs):
            raise AssertionError("GDC case fetch should not run when the restricted radiology cohort is empty.")

    class FakeTCIAClient:
        def fetch_studies_by_patient(self, collections, max_studies_per_collection=None):
            del collections, max_studies_per_collection
            return {}

        def fetch_series_by_patient(self, studies_by_patient, max_series_per_study=None):
            del studies_by_patient, max_series_per_study
            return {}

    with pytest.raises(ValueError, match="avoid replacing the TCGA registry slice with an empty cohort"):
        module._fetch_tcga_payloads(
            tcga_cfg=_tcga_cfg(restrict_to_radiology_cases=True),
            project_ids=["TCGA-KIRC"],
            gdc_client=FakeGDCClient(),
            tcia_client=FakeTCIAClient(),
            tcia_collections=["TCGA-KIRC"],
        )


def test_ensure_tcia_series_metadata_for_download_fetches_filtered_series_on_demand() -> None:
    module = _load_script_module()

    class FakeTCIAClient:
        def __init__(self) -> None:
            self.max_series_per_study = "unset"

        def fetch_series_by_patient(self, studies_by_patient, max_series_per_study=None):
            assert studies_by_patient == _make_studies()
            self.max_series_per_study = max_series_per_study
            return {
                "TCGA-AA-0001": [
                    {
                        "collection": "TCGA-KIRC",
                        "patient_id": "TCGA-AA-0001",
                        "study_instance_uid": "1.2.3",
                        "series_instance_uid": "1.2.3.4",
                        "modality": "CT",
                    },
                    {
                        "collection": "TCGA-KIRC",
                        "patient_id": "TCGA-AA-0001",
                        "study_instance_uid": "1.2.3",
                        "series_instance_uid": "1.2.3.5",
                        "modality": "SEG",
                    },
                ]
            }

    fake_tcia = FakeTCIAClient()
    resolved = module._ensure_tcia_series_metadata_for_download(
        tcga_cfg=_tcga_cfg(restrict_to_radiology_cases=False, fetch_series_metadata=False),
        download_cfg=OmegaConf.create({"max_series_per_study": 2}),
        tcia_client=fake_tcia,
        tcia_studies_by_patient=_make_studies(),
        tcia_series_by_patient={},
    )

    assert fake_tcia.max_series_per_study == 2
    assert list(resolved) == ["TCGA-AA-0001"]
    assert [entry["modality"] for entry in resolved["TCGA-AA-0001"]] == ["CT"]


def test_fetch_tcga_payloads_filters_to_requested_patient_chunk() -> None:
    module = _load_script_module()
    tcga_cfg = _tcga_cfg(restrict_to_radiology_cases=True)
    tcga_cfg.patient_chunk.size = 1
    tcga_cfg.patient_chunk.index = 1

    class FakeGDCClient:
        def __init__(self) -> None:
            self.fetch_cases_submitter_ids = None

        def fetch_cases(self, project_ids, *, submitter_ids=None, fields=None, max_cases=None):
            del project_ids, fields, max_cases
            self.fetch_cases_submitter_ids = submitter_ids
            return [{"case_id": "case-2", "submitter_id": "TCGA-BB-0002"}]

        def fetch_pathology_files(self, project_ids, *, case_ids=None, submitter_ids=None, data_formats=None, data_types=None, fields=None, max_files=None):
            del project_ids, case_ids, submitter_ids, data_formats, data_types, fields, max_files
            return []

        def fetch_report_files(self, *, project_ids, case_ids=None, data_formats=None, data_types=None, data_categories=None, fields=None, max_files=None):
            del project_ids, case_ids, data_formats, data_types, data_categories, fields, max_files
            return []

    class FakeTCIAClient:
        def fetch_studies_by_patient(self, collections, max_studies_per_collection=None):
            del collections, max_studies_per_collection
            return {
                "TCGA-AA-0001": [{"collection": "TCGA-KIRC", "patient_id": "TCGA-AA-0001", "study_instance_uid": "1.2.3"}],
                "TCGA-BB-0002": [{"collection": "TCGA-KIRC", "patient_id": "TCGA-BB-0002", "study_instance_uid": "2.3.4"}],
            }

        def fetch_series_by_patient(self, studies_by_patient, max_series_per_study=None):
            del studies_by_patient, max_series_per_study
            return {
                "TCGA-AA-0001": [{"collection": "TCGA-KIRC", "patient_id": "TCGA-AA-0001", "study_instance_uid": "1.2.3", "series_instance_uid": "1.2.3.4", "modality": "CT"}],
                "TCGA-BB-0002": [{"collection": "TCGA-KIRC", "patient_id": "TCGA-BB-0002", "study_instance_uid": "2.3.4", "series_instance_uid": "2.3.4.5", "modality": "CT"}],
            }

    fake_gdc = FakeGDCClient()
    cases, _pathology, _reports, tcia_studies, tcia_series, _ssm, _ok, _collections = module._fetch_tcga_payloads(
        tcga_cfg=tcga_cfg,
        project_ids=["TCGA-KIRC"],
        gdc_client=fake_gdc,
        tcia_client=FakeTCIAClient(),
        tcia_collections=["TCGA-KIRC"],
    )

    assert fake_gdc.fetch_cases_submitter_ids == ["TCGA-BB-0002"]
    assert [case["submitter_id"] for case in cases] == ["TCGA-BB-0002"]
    assert list(tcia_studies) == ["TCGA-BB-0002"]
    assert list(tcia_series) == ["TCGA-BB-0002"]


def test_build_tcga_genomics_artifact_index_tracks_patient_text_files(tmp_path: Path) -> None:
    module = _load_script_module()
    genomics_by_patient = {
        "TCGA-AA-0001": {
            "patient_id": "TCGA-AA-0001",
            "cancer_code": "KIRC",
            "genomics_text": "[GENOMICS]\ncancer_type: KIRC\n[/GENOMICS]",
        }
    }

    written = module._write_tcga_genomics_text_files(
        genomics_by_patient,
        tmp_path / "tcga_genomics_text",
    )

    by_case, by_patient, manifest_path, manifest_entries = module._build_tcga_genomics_artifact_index(
        cases=[
            {
                "case_id": "case-1",
                "submitter_id": "TCGA-AA-0001",
                "project": {"project_id": "TCGA-KIRC"},
            }
        ],
        genomics_text_paths=written,
        output_manifest_path=tmp_path / "tcga_genomics_downloads.jsonl",
    )

    assert written["TCGA-AA-0001"].read_text(encoding="utf-8").strip() == "[GENOMICS]\ncancer_type: KIRC\n[/GENOMICS]"
    assert list(by_case) == ["case-1"]
    assert list(by_patient) == ["TCGA-AA-0001"]
    assert manifest_path == tmp_path / "tcga_genomics_downloads.jsonl"

    manifest_lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 1
    record = json.loads(manifest_lines[0])
    assert record["case_id"] == "case-1"
    assert record["patient_id"] == "TCGA-AA-0001"
    assert record["payload_key"] == "genomics_text"
    assert record["data_type"] == "Patient Genomics Text"
    assert len(manifest_entries) == 1


def test_genomics_raw_payload_specs_cover_guideline_sources() -> None:
    module = _load_script_module()
    specs = module.GENOMICS_RAW_PAYLOAD_SPECS

    assert set(specs) == {
        "masked_somatic_mutation",
        "gene_expression_quantification",
        "copy_number_segments",
        "masked_copy_number_segments",
        "gene_level_copy_number",
        "mirna_expression_quantification",
        "clinical_supplement",
        "biospecimen_supplement",
        "methylation_beta_value",
    }
    assert specs["gene_expression_quantification"]["query_variants"][0]["workflow_types"] == ["STAR - Counts"]
    assert "BCR OMF XML" in specs["clinical_supplement"]["data_formats"]
    assert "BCR Biotab" in specs["biospecimen_supplement"]["data_formats"]


def test_merge_genomics_download_entry_indexes_preserves_raw_and_text_entries() -> None:
    module = _load_script_module()
    raw_index = {
        "TCGA-AA-0001": [
            {
                "payload_key": "gene_expression_quantification",
                "file_id": "file-1",
                "file_name": "expr.tsv",
                "case_id": "case-1",
                "patient_id": "TCGA-AA-0001",
                "local_path": "raw/tcga/genomics/gene_expression_quantification/expr.tsv",
                "relative_path": "raw/tcga/genomics/gene_expression_quantification/expr.tsv",
            }
        ]
    }
    text_index = {
        "TCGA-AA-0001": [
            {
                "payload_key": "genomics_text",
                "file_id": "",
                "file_name": "TCGA-AA-0001.txt",
                "case_id": "case-1",
                "patient_id": "TCGA-AA-0001",
                "local_path": "staging/tcga_genomics_text/TCGA-AA-0001.txt",
                "relative_path": "staging/tcga_genomics_text/TCGA-AA-0001.txt",
            }
        ]
    }

    merged = module._merge_genomics_download_entry_indexes(raw_index, text_index)

    assert list(merged) == ["TCGA-AA-0001"]
    assert [entry["payload_key"] for entry in merged["TCGA-AA-0001"]] == [
        "gene_expression_quantification",
        "genomics_text",
    ]


def test_write_tcga_msi_metadata_sidecars_creates_patient_tsv(tmp_path: Path) -> None:
    module = _load_script_module()

    by_case, by_patient, manifest_entries, count = module._write_tcga_msi_metadata_sidecars(
        msi_hits=[
            {
                "file_id": "bam-1",
                "file_name": "sample_wxs_gdc_realn.bam",
                "data_type": "Aligned Reads",
                "data_format": "BAM",
                "experimental_strategy": "WXS",
                "access": "controlled",
                "msi_score": 0.42,
                "msi_status": "MSS",
                "cases": [
                    {
                        "case_id": "case-1",
                        "submitter_id": "TCGA-AA-0001",
                        "project": {"project_id": "TCGA-KIRC"},
                    }
                ],
            }
        ],
        raw_root=tmp_path / "raw",
        source_name="tcga",
    )

    assert count == 1
    assert list(by_case) == ["case-1"]
    assert list(by_patient) == ["TCGA-AA-0001"]
    assert manifest_entries[0]["payload_key"] == module.MSI_METADATA_PAYLOAD_KEY
    output_path = tmp_path / "raw" / "tcga" / "genomics" / module.MSI_METADATA_PAYLOAD_KEY / "TCGA-KIRC" / "TCGA-AA-0001" / "msi_scores.tsv"
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split("\t") == [
        "file_id",
        "file_name",
        "data_type",
        "data_format",
        "experimental_strategy",
        "access",
        "msi_score",
        "msi_status",
    ]
    assert "bam-1" in lines[1]
    assert "MSS" in lines[1]


def test_gdc_download_data_file_redownloads_existing_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "slide.svs"
    output_path.write_bytes(b"x")
    payload = b"fresh-content"

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            del chunk_size
            yield payload

        def close(self) -> None:
            return None

    monkeypatch.setattr(tcga_module.requests, "get", lambda *args, **kwargs: FakeResponse())

    client = GDCClient(max_retries=1)
    resolved = client.download_data_file(
        file_id="file-1",
        output_path=output_path,
        skip_existing=True,
        expected_size=len(payload),
    )

    assert resolved == output_path
    assert output_path.read_bytes() == payload
    assert not Path(f"{output_path}.part").exists()


def test_tcia_download_series_zip_redownloads_invalid_existing_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "series.zip"
    output_path.write_bytes(b"not-a-zip")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("image-1.dcm", b"dicom")
    payload = buffer.getvalue()

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            del chunk_size
            yield payload

        def close(self) -> None:
            return None

    monkeypatch.setattr(tcga_module.requests, "get", lambda *args, **kwargs: FakeResponse())

    client = TCIAClient(max_retries=1)
    resolved = client.download_series_zip(
        series_instance_uid="1.2.3.4",
        output_path=output_path,
        skip_existing=True,
    )

    assert resolved == output_path
    assert zipfile.is_zipfile(output_path)
    assert not Path(f"{output_path}.part").exists()
    with zipfile.ZipFile(output_path) as handle:
        assert handle.namelist() == ["image-1.dcm"]
