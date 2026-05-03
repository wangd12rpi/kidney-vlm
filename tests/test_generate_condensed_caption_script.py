from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "10_mcq_from_caption_new" / "01_generate_condensed_caption.py"
    spec = importlib.util.spec_from_file_location("generate_condensed_caption_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _filter_cfg():
    return OmegaConf.create(
        {
            "filters": {
                "split": {"enabled": True, "values": ["test"]},
                "project_id": {"enabled": True, "values": ["TCGA-BRCA"]},
                "caption_contains": {"enabled": True, "values": ["ductal"]},
                "first_n": {"enabled": True, "value": 1},
            }
        }
    )


def _prompt_cfg():
    return OmegaConf.create(
        {
            "azure": {
                "max_completion_tokens": 200,
                "temperature": 0.0,
                "reasoning_effort": "low",
                "max_retries": 3,
                "retry_sleep_seconds": 0.0,
            },
            "prompt": {
                "max_words_per_section": 20,
                "contrast_examples": {"count": 3},
                "system_prompt": "Return strict JSON only.",
                "user_template": (
                    "Target {case_id} from {project_id}; {max_words} words.\n"
                    "Target:\n```text\n{target_caption}\n```\n"
                    "Others:\n{other_case_captions}"
                ),
            },
        }
    )


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason
        self.content_filter_results = {}


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.choices = [_FakeChoice(content, finish_reason=finish_reason)]


class _FakeCompletions:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self.content, finish_reason=self.finish_reason)


class _FakeChat:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.completions = _FakeCompletions(content, finish_reason=finish_reason)


class _FakeClient:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.chat = _FakeChat(content, finish_reason=finish_reason)


def test_apply_filters_supports_split_project_caption_and_first_n() -> None:
    module = _load_script_module()
    frame = pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "project_id": "TCGA-BRCA",
                "split": "test",
                "caption": "Pathology Findings\nInvasive ductal carcinoma.",
            },
            {
                "case_id": "case-2",
                "project_id": "TCGA-BRCA",
                "split": "test",
                "caption": "Pathology Findings\nLobular carcinoma.",
            },
            {
                "case_id": "case-3",
                "project_id": "TCGA-LUAD",
                "split": "test",
                "caption": "Pathology Findings\nDuctal mimic.",
            },
            {
                "case_id": "case-4",
                "project_id": "TCGA-BRCA",
                "split": "train",
                "caption": "Pathology Findings\nInvasive ductal carcinoma.",
            },
        ]
    )

    filtered = module._apply_filters(frame, _filter_cfg())

    assert filtered["case_id"].tolist() == ["case-1"]


def test_build_contrast_caption_map_uses_same_project_other_cases() -> None:
    module = _load_script_module()
    frame = pd.DataFrame(
        [
            {"case_id": "brca-1", "project_id": "TCGA-BRCA", "caption": "BRCA caption 1"},
            {"case_id": "brca-2", "project_id": "TCGA-BRCA", "caption": "BRCA caption 2"},
            {"case_id": "brca-3", "project_id": "TCGA-BRCA", "caption": "BRCA caption 3"},
            {"case_id": "luad-1", "project_id": "TCGA-LUAD", "caption": "LUAD caption 1"},
            {"case_id": "luad-2", "project_id": "TCGA-LUAD", "caption": "LUAD caption 2"},
            {"case_id": "luad-3", "project_id": "TCGA-LUAD", "caption": "LUAD caption 3"},
        ]
    )

    contrast_map = module._build_contrast_caption_map(frame, count=2)

    assert [item["case_id"] for item in contrast_map["brca-1"]] == ["brca-2", "brca-3"]
    assert [item["case_id"] for item in contrast_map["luad-2"]] == ["luad-3", "luad-1"]


def test_extract_caption_fingerprint_sends_full_caption_and_returns_required_keys() -> None:
    module = _load_script_module()
    payload = {
        "radiology_findings": "Irregular enhancing breast lesion.",
        "pathology_findings": "Invasive ductal carcinoma with desmoplastic stroma.",
        "genomic_findings": "TP53 mutation with basal-like expression.",
        "integrated_interpretation": "Findings support aggressive invasive breast carcinoma.",
    }
    client = _FakeClient(json.dumps(payload))
    caption = (
        "Radiology Findings\nMRI shows an irregular enhancing breast lesion.\n\n"
        "Pathology Findings\nSections show invasive ductal carcinoma with desmoplastic stroma.\n\n"
        "Limitations\nDefinitive mapping is limited."
    )

    condensed = module._extract_caption_fingerprint(
        client=client,
        deployment="gpt-test",
        cfg=_prompt_cfg(),
        case_id="case-target",
        project_id="TCGA-BRCA",
        caption=caption,
        contrast_examples=[
            {"case_id": "case-other-1", "caption": "Other breast case one."},
            {"case_id": "case-other-2", "caption": "Other breast case two."},
            {"case_id": "case-other-3", "caption": "Other breast case three."},
        ],
    )

    assert condensed == payload
    kwargs = client.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["temperature"] == 0.0
    assert kwargs["reasoning_effort"] == "low"
    user_message = kwargs["messages"][1]["content"]
    assert "Target case-target from TCGA-BRCA" in user_message
    assert "Radiology Findings" in user_message
    assert "Pathology Findings" in user_message
    assert "Limitations" in user_message
    assert "case-other-1" in user_message
    assert "```text" in user_message


def test_extract_caption_fingerprint_fails_loudly_when_json_key_is_missing() -> None:
    module = _load_script_module()
    client = _FakeClient(
        json.dumps(
            {
                "radiology_findings": "",
                "pathology_findings": "Invasive carcinoma.",
                "genomic_findings": "TP53 mutation.",
            }
        )
    )

    with pytest.raises(module.MissingFingerprintKeys, match="missing required keys"):
        module._extract_caption_fingerprint(
            client=client,
            deployment="gpt-test",
            cfg=_prompt_cfg(),
            case_id="case-target",
            project_id="TCGA-BRCA",
            caption="Pathology Findings\nInvasive carcinoma.",
            contrast_examples=[
                {"case_id": "case-other-1", "caption": "Other breast case one."},
                {"case_id": "case-other-2", "caption": "Other breast case two."},
                {"case_id": "case-other-3", "caption": "Other breast case three."},
            ],
        )


def test_extract_caption_fingerprint_distinguishes_invalid_json() -> None:
    module = _load_script_module()
    client = _FakeClient("not json")

    with pytest.raises(module.InvalidFingerprintJson, match="not valid JSON"):
        module._extract_caption_fingerprint(
            client=client,
            deployment="gpt-test",
            cfg=_prompt_cfg(),
            case_id="case-target",
            project_id="TCGA-BRCA",
            caption="Pathology Findings\nInvasive carcinoma.",
            contrast_examples=[
                {"case_id": "case-other-1", "caption": "Other breast case one."},
                {"case_id": "case-other-2", "caption": "Other breast case two."},
                {"case_id": "case-other-3", "caption": "Other breast case three."},
            ],
        )


def test_extract_caption_fingerprint_distinguishes_empty_whole_response() -> None:
    module = _load_script_module()
    client = _FakeClient("", finish_reason="content_filter")

    with pytest.raises(module.EmptyFingerprintResponse, match="completely empty response"):
        module._extract_caption_fingerprint(
            client=client,
            deployment="gpt-test",
            cfg=_prompt_cfg(),
            case_id="case-target",
            project_id="TCGA-BRCA",
            caption="Pathology Findings\nInvasive carcinoma.",
            contrast_examples=[
                {"case_id": "case-other-1", "caption": "Other breast case one."},
                {"case_id": "case-other-2", "caption": "Other breast case two."},
                {"case_id": "case-other-3", "caption": "Other breast case three."},
            ],
        )


def test_extract_caption_fingerprint_accepts_empty_section_values() -> None:
    module = _load_script_module()
    payload = {
        "radiology_findings": "",
        "pathology_findings": "Discohesive tumor cords.",
        "genomic_findings": "",
        "integrated_interpretation": "Lobular-pattern tumor.",
    }
    client = _FakeClient(json.dumps(payload))

    condensed = module._extract_caption_fingerprint(
        client=client,
        deployment="gpt-test",
        cfg=_prompt_cfg(),
        case_id="case-target",
        project_id="TCGA-BRCA",
        caption="Pathology Findings\nDiscohesive tumor cords.",
        contrast_examples=[
            {"case_id": "case-other-1", "caption": "Other breast case one."},
            {"case_id": "case-other-2", "caption": "Other breast case two."},
            {"case_id": "case-other-3", "caption": "Other breast case three."},
        ],
    )

    assert condensed == payload


def test_extract_caption_fingerprint_with_retries_returns_none_after_repeated_bad_outputs(capsys) -> None:
    module = _load_script_module()
    client = _FakeClient('{"pathology_findings": "Only one key."}')

    condensed = module._extract_caption_fingerprint_with_retries(
        client=client,
        deployment="gpt-test",
        cfg=_prompt_cfg(),
        caption="Pathology Findings\nInvasive carcinoma.",
        case_id="case-bad",
        project_id="TCGA-BRCA",
        contrast_examples=[
            {"case_id": "case-other-1", "caption": "Other breast case one."},
            {"case_id": "case-other-2", "caption": "Other breast case two."},
            {"case_id": "case-other-3", "caption": "Other breast case three."},
        ],
    )

    captured = capsys.readouterr()
    assert condensed is None
    assert "attempt 3/3" in captured.out
    assert "Skipping case_id=case-bad" in captured.out
