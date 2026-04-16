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


def test_fetch_tcga_payloads_filters_to_requested_patient_subset_without_radiology_restriction() -> None:
    module = _load_script_module()
    tcga_cfg = _tcga_cfg(restrict_to_radiology_cases=False)
    tcga_cfg.patient_subset_ids = ["TCGA-BB-0002"]

    class FakeGDCClient:
        def __init__(self) -> None:
            self.fetch_cases_submitter_ids = None

        def fetch_cases(self, project_ids, *, submitter_ids=None, fields=None, max_cases=None):
            del project_ids, fields, max_cases
            self.fetch_cases_submitter_ids = submitter_ids
            return [{"case_id": "case-2", "submitter_id": "TCGA-BB-0002"}]

        def fetch_pathology_files(self, project_ids, *, case_ids=None, submitter_ids=None, data_formats=None, data_types=None, fields=None, max_files=None):
            del project_ids, data_formats, data_types, fields, max_files
            assert case_ids == ["case-2"]
            assert submitter_ids == ["TCGA-BB-0002"]
            return []

        def fetch_report_files(self, *, project_ids, case_ids=None, data_formats=None, data_types=None, data_categories=None, fields=None, max_files=None):
            del project_ids, case_ids, data_formats, data_types, data_categories, fields, max_files
            return []

    class FakeTCIAClient:
        def fetch_studies_by_patient(self, collections, max_studies_per_collection=None):
            del collections, max_studies_per_collection
            return {}

        def fetch_series_by_patient(self, studies_by_patient, max_series_per_study=None):
            del studies_by_patient, max_series_per_study
            return {}

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
    assert tcia_studies == {}
    assert tcia_series == {}


def test_fetch_tcga_payloads_filters_to_requested_patient_chunk_without_radiology_restriction() -> None:
    module = _load_script_module()
    tcga_cfg = _tcga_cfg(restrict_to_radiology_cases=False)
    tcga_cfg.patient_chunk.size = 1
    tcga_cfg.patient_chunk.index = 1

    class FakeGDCClient:
        def __init__(self) -> None:
            self.fetch_cases_submitter_ids = None

        def fetch_cases(self, project_ids, *, submitter_ids=None, fields=None, max_cases=None):
            del project_ids, fields, max_cases
            self.fetch_cases_submitter_ids = submitter_ids
            return [
                {"case_id": "case-1", "submitter_id": "TCGA-AA-0001"},
                {"case_id": "case-2", "submitter_id": "TCGA-BB-0002"},
            ]

        def fetch_pathology_files(self, project_ids, *, case_ids=None, submitter_ids=None, data_formats=None, data_types=None, fields=None, max_files=None):
            del project_ids, data_formats, data_types, fields, max_files
            assert case_ids == ["case-2"]
            assert submitter_ids == ["TCGA-BB-0002"]
            return []

        def fetch_report_files(self, *, project_ids, case_ids=None, data_formats=None, data_types=None, data_categories=None, fields=None, max_files=None):
            del project_ids, case_ids, data_formats, data_types, data_categories, fields, max_files
            return []

    class FakeTCIAClient:
        def fetch_studies_by_patient(self, collections, max_studies_per_collection=None):
            del collections, max_studies_per_collection
            return {}

        def fetch_series_by_patient(self, studies_by_patient, max_series_per_study=None):
            del studies_by_patient, max_series_per_study
            return {}

    fake_gdc = FakeGDCClient()
    cases, _pathology, _reports, tcia_studies, tcia_series, _ssm, _ok, _collections = module._fetch_tcga_payloads(
        tcga_cfg=tcga_cfg,
        project_ids=["TCGA-KIRC"],
        gdc_client=fake_gdc,
        tcia_client=FakeTCIAClient(),
        tcia_collections=["TCGA-KIRC"],
    )

    assert fake_gdc.fetch_cases_submitter_ids is None
    assert [case["submitter_id"] for case in cases] == ["TCGA-BB-0002"]
    assert tcia_studies == {}
    assert tcia_series == {}


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


def test_extract_tcia_series_zip_preserves_existing_pngs_when_reextracting(tmp_path: Path) -> None:
    module = _load_script_module()
    zip_path = tmp_path / "series.zip"
    extracted_root = tmp_path / "series"
    extracted_root.mkdir(parents=True, exist_ok=True)
    png_path = extracted_root / "00000001.png"
    png_path.write_bytes(b"png")

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("00000001.dcm", b"dicom")

    resolved = module._extract_tcia_series_zip(
        zip_path=zip_path,
        extracted_root=extracted_root,
        skip_existing=True,
    )

    assert resolved == extracted_root
    assert png_path.exists()
    assert (extracted_root / "00000001.dcm").exists()


def test_cleanup_tcia_series_source_artifacts_deletes_zip_and_source_files_but_keeps_pngs(tmp_path: Path) -> None:
    module = _load_script_module()
    radiology_root = tmp_path / "raw" / "tcga" / "radiology"
    series_dir = radiology_root / "TCGA-KIRC" / "TCGA-AA-0001" / "1.2.3" / "4.5.6"
    series_dir.mkdir(parents=True, exist_ok=True)
    dicom_path = series_dir / "00000001.dcm"
    dicom_path.write_bytes(b"dicom")
    license_path = series_dir / "LICENSE"
    license_path.write_text("license", encoding="utf-8")
    png_path = series_dir / "00000001.png"
    png_path.write_bytes(b"png")
    zip_path = series_dir.parent / "4.5.6.zip"
    zip_path.write_bytes(b"zip")

    entry = {
        "accepted": True,
        "local_path": str(zip_path),
        "series_zip_path": str(zip_path),
        "extracted_root": str(series_dir),
        "selected_series_dir": str(series_dir),
        "extracted_path": str(series_dir),
        "source_file_paths": [str(dicom_path), str(license_path)],
        "usable_dicom_paths": [str(dicom_path)],
        "png_paths": [str(png_path)],
        "slice_count": 1,
    }

    result = module._cleanup_tcia_series_source_artifacts(
        entry,
        cleanup_settings={
            "enabled": True,
            "delete_series_zip_files": True,
            "delete_source_dicom_files": True,
            "prune_empty_directories": True,
        },
        radiology_root=radiology_root,
        feature_extraction_enabled=False,
        segmentation_enabled=False,
    )

    assert result["zip_deleted"] is True
    assert result["source_files_deleted"] == 2
    assert png_path.exists()
    assert not dicom_path.exists()
    assert not license_path.exists()
    assert entry["local_path"] == ""
    assert entry["series_zip_path"] == ""
    assert entry["selected_series_dir"] == str(series_dir)


def test_cleanup_tcia_series_source_artifacts_removes_rejected_series_root(tmp_path: Path) -> None:
    module = _load_script_module()
    radiology_root = tmp_path / "raw" / "tcga" / "radiology"
    series_dir = radiology_root / "TCGA-KIRC" / "TCGA-BB-0002" / "7.8.9" / "1.2.3"
    series_dir.mkdir(parents=True, exist_ok=True)
    (series_dir / "00000001.dcm").write_bytes(b"dicom")
    zip_path = series_dir.parent / "1.2.3.zip"
    zip_path.write_bytes(b"zip")

    entry = {
        "accepted": False,
        "local_path": str(zip_path),
        "series_zip_path": str(zip_path),
        "extracted_root": str(series_dir),
        "selected_series_dir": str(series_dir),
        "extracted_path": str(series_dir),
        "source_file_paths": [str(series_dir / "00000001.dcm")],
        "usable_dicom_paths": [],
        "png_paths": [],
        "slice_count": 0,
    }

    result = module._cleanup_tcia_series_source_artifacts(
        entry,
        cleanup_settings={
            "enabled": True,
            "delete_series_zip_files": True,
            "delete_source_dicom_files": True,
            "prune_empty_directories": True,
        },
        radiology_root=radiology_root,
        feature_extraction_enabled=False,
        segmentation_enabled=False,
    )

    assert result["zip_deleted"] is True
    assert result["extracted_root_removed"] is True
    assert not series_dir.exists()
    assert not zip_path.exists()
    assert entry["selected_series_dir"] == ""
    assert entry["extracted_path"] == ""


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


def test_download_tcia_series_extracts_and_qcs_downloaded_series(tmp_path: Path) -> None:
    module = _load_script_module()

    class FakeTCIAClient:
        def download_series_zip(self, *, series_instance_uid: str, output_path: Path, skip_existing: bool = True) -> Path:
            del skip_existing
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output_path, "w") as handle:
                handle.writestr(f"{series_instance_uid}/image-1.dcm", b"dicom")
            return output_path

    qc_calls: list[Path] = []

    def fake_run_tcia_series_qc(*, extracted_root: Path, qc_cfg) -> dict[str, object]:
        del qc_cfg
        qc_calls.append(extracted_root)
        assert extracted_root.exists()
        selected_dir = extracted_root / "1.2.3.4"
        if not selected_dir.exists():
            selected_dir = next(path for path in extracted_root.iterdir() if path.is_dir())
        return {
            "accepted": True,
            "selected_series_dir": str(selected_dir),
            "candidate_series_dirs": [str(selected_dir)],
            "usable_image_paths": [str(selected_dir / "image-1.dcm")],
            "series_reject_record": None,
            "image_reject_records": [],
        }

    original_run_tcia_series_qc = module._run_tcia_series_qc
    original_build_feature_extractor = module._build_tcia_radiology_feature_extractor

    class FakeFeatureExtractor:
        def extract_series(
            self,
            *,
            series_dir: Path,
            usable_image_paths,
            project_id: str,
            patient_id: str,
            study_instance_uid: str,
            series_instance_uid: str,
            modality: str,
        ):
            del usable_image_paths, project_id, patient_id, study_instance_uid, modality

            class _Result:
                png_dir = str(tmp_path / "pngs" / series_instance_uid)
                embedding_ref = f"data/features/tcga_all_slices.h5::series={series_dir}"
                slice_count = 1

            return _Result()

    try:
        module._run_tcia_series_qc = fake_run_tcia_series_qc
        module._build_tcia_radiology_feature_extractor = lambda **_kwargs: FakeFeatureExtractor()
        downloaded_by_patient, total_downloaded, qc_entries, accepted_count = module._download_tcia_series(
            FakeTCIAClient(),
            tcga_cfg=OmegaConf.create(
                {
                    "tcia": {
                        "qc": {
                            "detail_root": str(tmp_path / "qc_details"),
                            "keep_rejected_extracted_series": False,
                        },
                        "feature_extraction": {"enabled": True},
                    }
                }
            ),
            tcia_series_by_patient={
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
                        "modality": "MR",
                    },
                ]
            },
            patient_ids={"TCGA-AA-0001"},
            raw_root=tmp_path / "raw",
            source_name="tcga",
            staging_root=tmp_path / "staging",
            skip_existing=True,
            max_series_per_study=None,
            max_series_total=None,
        )
    finally:
        module._run_tcia_series_qc = original_run_tcia_series_qc
        module._build_tcia_radiology_feature_extractor = original_build_feature_extractor

    assert total_downloaded == 2
    assert accepted_count == 2
    assert list(downloaded_by_patient) == ["TCGA-AA-0001"]
    assert len(downloaded_by_patient["TCGA-AA-0001"]) == 2
    assert downloaded_by_patient["TCGA-AA-0001"][0]["embedding_ref"].startswith("data/features/tcga_all_slices.h5::series=")
    assert downloaded_by_patient["TCGA-AA-0001"][0]["png_dir"].endswith("/pngs/1.2.3.4")
    assert downloaded_by_patient["TCGA-AA-0001"][0]["slice_count"] == 1
    assert all(entry["accepted"] is True for entry in downloaded_by_patient["TCGA-AA-0001"])
    assert all(Path(entry["local_path"]).suffix == ".zip" for entry in downloaded_by_patient["TCGA-AA-0001"])
    assert all(Path(entry["extracted_path"]).exists() for entry in downloaded_by_patient["TCGA-AA-0001"])
    assert len(qc_entries) == 2
    assert len(qc_calls) == 2
    assert all(Path(entry["qc_detail_path"]).exists() for entry in qc_entries)


def test_download_tcia_series_skips_qc_when_disabled(tmp_path: Path) -> None:
    module = _load_script_module()

    class FakeTCIAClient:
        def download_series_zip(self, *, series_instance_uid: str, output_path: Path, skip_existing: bool = True) -> Path:
            del skip_existing
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output_path, "w") as handle:
                handle.writestr(f"{series_instance_uid}/image-1.dcm", b"dicom")
            return output_path

    original_run_tcia_series_qc = module._run_tcia_series_qc

    def fail_if_called(*args, **kwargs):
        raise AssertionError("QC should not run when data.source.tcga.tcia.qc.enabled=false")

    try:
        module._run_tcia_series_qc = fail_if_called
        downloaded_by_patient, total_downloaded, qc_entries, accepted_count = module._download_tcia_series(
            FakeTCIAClient(),
            tcga_cfg=OmegaConf.create(
                {
                    "tcia": {
                        "qc": {
                            "enabled": False,
                            "detail_root": str(tmp_path / "qc_details"),
                            "keep_rejected_extracted_series": False,
                        },
                        "feature_extraction": {"enabled": False},
                    }
                }
            ),
            tcia_series_by_patient={
                "TCGA-AA-0001": [
                    {
                        "collection": "TCGA-KIRC",
                        "patient_id": "TCGA-AA-0001",
                        "study_instance_uid": "1.2.3",
                        "series_instance_uid": "1.2.3.4",
                        "modality": "CT",
                    }
                ]
            },
            patient_ids={"TCGA-AA-0001"},
            raw_root=tmp_path / "raw",
            source_name="tcga",
            staging_root=tmp_path / "staging",
            skip_existing=True,
            max_series_per_study=None,
            max_series_total=None,
        )
    finally:
        module._run_tcia_series_qc = original_run_tcia_series_qc

    assert total_downloaded == 1
    assert accepted_count == 1
    assert list(downloaded_by_patient) == ["TCGA-AA-0001"]
    assert downloaded_by_patient["TCGA-AA-0001"][0]["accepted"] is True
    assert downloaded_by_patient["TCGA-AA-0001"][0]["reject_reason"] == ""
    assert Path(downloaded_by_patient["TCGA-AA-0001"][0]["extracted_path"]).name == "1.2.3.4"
    assert len(qc_entries) == 1
    assert qc_entries[0]["accepted"] is True
    assert Path(qc_entries[0]["qc_detail_path"]).exists()


def test_configure_portable_radiology_chunk_outputs_routes_paths_into_chunk_bundle() -> None:
    module = _load_script_module()
    cfg = OmegaConf.create(
        {
            "data": {
                "staging_root": "/tmp/project/data/staging",
                "registry_root": "/tmp/project/data/registry",
                "unified_registry_path": "/tmp/project/data/registry/unified.parquet",
                "manifests_root": "/tmp/project/data/registry/manifests",
                "source": {
                    "tcga": {
                        "radiology_process_root": "/tmp/project/data/processes/radiology",
                        "patient_chunk": {"index": 2, "size": 64},
                        "genomics": {
                            "download_manifest_jsonl_path": "/tmp/project/data/staging/tcga_genomics_downloads.jsonl",
                            "sidecar_jsonl_path": "/tmp/project/data/staging/tcga_genomics.jsonl",
                            "text_output_dir": "/tmp/project/data/staging/tcga_genomics_text",
                        },
                        "tcia": {
                            "qc": {
                                "detail_root": "/tmp/project/data/staging/tcga_radiology_qc",
                                "report_jsonl_path": "/tmp/project/data/staging/tcga_radiology_qc.jsonl",
                            },
                            "feature_extraction": {
                                "feature_store_path": "/tmp/project/data/features/radiology/tcga_all_slices.h5",
                            },
                        },
                    }
                },
            }
        }
    )

    layout = module._configure_portable_radiology_chunk_outputs(cfg)

    assert layout is not None
    assert layout.chunk.label == "chunk3"
    assert str(layout.bundle_root).endswith("/data/processes/radiology/chunk3")
    assert str(layout.registry_path).endswith("/data/processes/radiology/chunk3/registry/tcga.parquet")
    assert str(layout.manifest_path).endswith("/data/processes/radiology/chunk3/chunk_manifest.json")
    assert str(cfg.data.source.tcga.tcia.feature_extraction.feature_store_path).endswith(
        "/data/processes/radiology/chunk3/features_medsiglip448/chunk3.h5"
    )
    assert str(cfg.data.source.tcga.tcia.qc.detail_root).endswith(
        "/data/processes/radiology/chunk3/qc"
    )
    assert str(cfg.data.source.tcga.tcia.qc.report_jsonl_path).endswith(
        "/data/processes/radiology/chunk3/qc_report.jsonl"
    )
    assert str(cfg.data.source.tcga.tcia.feature_extraction.png_root).endswith(
        "/data/processes/radiology/chunk3/pngs"
    )
    assert str(cfg.data.source.tcga.tcia.segmentation.mask_root).endswith(
        "/data/processes/radiology/chunk3/mask_medicalsam3"
    )


def test_configure_portable_radiology_chunk_outputs_is_noop_without_chunk_size() -> None:
    module = _load_script_module()
    cfg = OmegaConf.create(
        {
            "data": {
                "staging_root": "/tmp/project/data/staging",
                "registry_root": "/tmp/project/data/registry",
                "unified_registry_path": "/tmp/project/data/registry/unified.parquet",
                "manifests_root": "/tmp/project/data/registry/manifests",
                "source": {
                    "tcga": {
                        "patient_chunk": {"index": 0, "size": None},
                        "tcia": {
                            "qc": {},
                            "feature_extraction": {
                                "feature_store_path": "/tmp/project/data/features/radiology/tcga_all_slices.h5",
                            },
                        },
                    }
                },
            }
        }
    )

    assert module._configure_portable_radiology_chunk_outputs(cfg) is None
    assert str(cfg.data.unified_registry_path) == "/tmp/project/data/registry/unified.parquet"


def test_radiology_path_builders_fall_back_when_config_paths_are_null(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    captured: dict[str, Path] = {}

    class FakeFeatureExtractor:
        def __init__(self, **kwargs) -> None:
            captured["feature_store_path"] = kwargs["feature_store_path"]

    class FakeSegmentationExtractor:
        def __init__(self, **kwargs) -> None:
            captured["mask_root"] = kwargs["mask_root"]
            captured["keyword_map_path"] = kwargs["keyword_map_path"]
            captured["checkpoint_path"] = kwargs["checkpoint_path"]
            captured["medical_sam3_root"] = kwargs["medical_sam3_root"]
            captured["sam3_root"] = kwargs["sam3_root"]

    monkeypatch.setattr(module, "TCGARadiologyFeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(module, "TCGARadiologySegmentationExtractor", FakeSegmentationExtractor)

    tcga_cfg = OmegaConf.create(
        {
            "tcia": {
                "feature_extraction": {
                    "enabled": True,
                    "feature_store_path": None,
                    "model_name": "google/medsiglip-448",
                    "input_size": 448,
                    "batch_size": 16,
                    "device": "cpu",
                    "skip_existing_features": True,
                },
                "segmentation": {
                    "enabled": True,
                    "mask_root": None,
                    "keyword_map_path": None,
                    "checkpoint_path": None,
                    "medical_sam3_root": None,
                    "sam3_root": None,
                    "input_size": 448,
                    "confidence_threshold": 0.1,
                    "device": "cpu",
                    "overwrite_masks": False,
                    "skip_existing_masks": True,
                    "min_mask_pixels": 16,
                },
            }
        }
    )

    raw_root = module.ROOT / "data" / "raw"
    module._build_tcia_radiology_feature_extractor(tcga_cfg=tcga_cfg, raw_root=raw_root)
    module._build_tcia_radiology_segmentation_extractor(tcga_cfg=tcga_cfg, raw_root=raw_root)

    assert captured["feature_store_path"] == (
        module.ROOT / "data" / "features" / "radiology_features_medsiglip_448" / "tcga_all_slices.h5"
    ).resolve()
    assert captured["mask_root"] == (module.ROOT / "data" / "features" / "radiology_masks_medsam3").resolve()
    assert captured["keyword_map_path"] == (module.ROOT / "conf" / "data" / "tcga_medsam3_keywords.yaml").resolve()
    assert captured["checkpoint_path"] == (
        module.ROOT / "external" / "Medical-SAM3" / "checkpoints" / "checkpoint.pt"
    ).resolve()
    assert captured["medical_sam3_root"] == (module.ROOT / "external" / "Medical-SAM3").resolve()
    assert captured["sam3_root"] == (module.ROOT / "external" / "sam3").resolve()
