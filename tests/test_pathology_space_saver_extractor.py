from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys

import h5py
import pandas as pd


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "embeding_extraction" / "02_extract_pathology_features_space_saver.py"
    spec = importlib.util.spec_from_file_location("pathology_space_saver_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_patch_features(path: Path, rows: int, cols: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=[[0.0] * cols for _ in range(rows)])
        handle.create_dataset("coords", data=[[0, 0] for _ in range(rows)])


def test_build_remote_slide_candidates_only_counts_missing_raw_slides(tmp_path: Path) -> None:
    module = _load_script_module()

    features_root = tmp_path / "features"
    patch_features_dir = features_root / "features_conch_v15"
    slide_features_dir = features_root / "slide_features_titan"
    coords_root = features_root / "coords_20x_512px_0px_overlap"

    existing_raw = tmp_path / "TCGA-BB-0002-01Z-00-DX1.dx-uuid.svs"
    existing_raw.write_text("present", encoding="utf-8")

    precomputed_stem = "TCGA-CC-0003-01Z-00-DX1.dx-uuid"
    _write_patch_features(patch_features_dir / f"{precomputed_stem}.h5", rows=6)

    missing_stem = "TCGA-AA-0001-01Z-00-DX1.dx-uuid"

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "tcga-case-1",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "pathology_wsi_paths": [str(tmp_path / f"{missing_stem}.svs")],
                "pathology_file_ids": ["file-1"],
            },
            {
                "sample_id": "tcga-case-2",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "pathology_wsi_paths": [str(existing_raw)],
                "pathology_file_ids": ["file-2"],
            },
            {
                "sample_id": "tcga-case-3",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "pathology_wsi_paths": [str(tmp_path / f"{precomputed_stem}.svs")],
                "pathology_file_ids": ["file-3"],
            },
        ]
    )

    candidates, stats = module._build_remote_slide_candidates(
        registry_df,
        patch_features_dir=patch_features_dir,
        slide_features_dir=slide_features_dir,
        coords_root=coords_root,
        save_format="h5",
        extract_patch_only=True,
        overwrite_existing=False,
    )

    assert len(candidates) == 1
    assert candidates[0].sample_id == "tcga-case-1"
    assert candidates[0].slide_stem == missing_stem
    assert stats.selected_slide_refs == 3
    assert stats.local_raw_available == 1
    assert stats.already_extracted == 1
    assert stats.todo_candidates == 1


def test_build_remote_slide_jobs_matches_file_name_to_candidate_path() -> None:
    module = _load_script_module()

    candidate = module.PendingRemoteSlideCandidate(
        row_idx=0,
        sample_id="tcga-case-1",
        source="tcga",
        project_id="TCGA-KIRC",
        slide_path="/tmp/TCGA-AA-0001-01Z-00-DX1.dx-uuid.svs",
        slide_stem="TCGA-AA-0001-01Z-00-DX1.dx-uuid",
        pathology_file_ids=("wrong-id", "right-id"),
        needs_patch=True,
        needs_slide=False,
    )

    jobs, unresolved = module._build_remote_slide_jobs(
        [candidate],
        {
            "wrong-id": "other-file.svs",
            "right-id": "TCGA-AA-0001-01Z-00-DX1.dx-uuid.svs",
        },
    )

    assert unresolved == 0
    assert len(jobs) == 1
    assert jobs[0].file_id == "right-id"


def test_resolve_tcga_gdc_cfg_falls_back_to_repo_tcga_source_config() -> None:
    module = _load_script_module()

    cfg = module.load_cfg()
    gdc_cfg = module._resolve_tcga_gdc_cfg(cfg)

    assert str(gdc_cfg.base_url) == "https://api.gdc.cancer.gov"
    assert int(gdc_cfg.page_size) == 1000


def test_submit_prefetch_download_writes_unique_temp_file(tmp_path: Path) -> None:
    module = _load_script_module()

    class FakeGDCClient:
        def download_data_file(self, *, file_id: str, output_path: Path, skip_existing: bool = True, chunk_size: int = 1024 * 1024) -> Path:
            output_path.write_text(f"downloaded:{file_id}", encoding="utf-8")
            return output_path

    job = module.PendingRemoteSlideJob(
        row_idx=0,
        sample_id="tcga-case-1",
        source="tcga",
        project_id="TCGA-KIRC",
        slide_path="/tmp/TCGA-AA-0001-01Z-00-DX1.dx-uuid.svs",
        slide_stem="TCGA-AA-0001-01Z-00-DX1.dx-uuid",
        file_id="file-1",
        needs_patch=True,
        needs_slide=False,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        prefetched = module._submit_prefetch_download(
            executor=executor,
            gdc_client=FakeGDCClient(),
            job=job,
            temp_dir=tmp_path,
            job_index=7,
        )
        downloaded_path = prefetched.future.result(timeout=5)

    assert downloaded_path == prefetched.temp_slide_path
    assert prefetched.temp_slide_path.exists()
    assert prefetched.temp_slide_path.name.startswith("00007-file-1-")
    assert prefetched.temp_slide_path.read_text(encoding="utf-8") == "downloaded:file-1"
