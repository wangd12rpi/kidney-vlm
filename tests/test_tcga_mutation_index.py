from __future__ import annotations

from typing import Any

from kidney_vlm.data.sources.tcga import GDCClient, index_ssm_hits_by_case_and_patient


def test_index_ssm_hits_by_case_and_patient() -> None:
    hits = [
        {
            "ssm_id": "ssm-a",
            "mutation_type": "Missense_Mutation",
            "occurrence": [
                {
                    "case": {
                        "case_id": "case-1",
                        "submitter_id": "TCGA-AA-0001",
                        "project": {"project_id": "TCGA-KIRC"},
                    }
                }
            ],
            "consequence": [
                {
                    "transcript": {
                        "gene": {"symbol": "vhl"},
                        "consequence_type": "missense_variant",
                    }
                }
            ],
        },
        {
            "ssm_id": "ssm-b",
            "mutation_type": "Nonsense_Mutation",
            "case": {
                "case_id": "case-2",
                "submitter_id": "TCGA-BB-0002",
            },
            "consequence": [
                {
                    "transcript": {
                        "gene": {"symbol": "TP53"},
                        "vep_consequence": "stop_gained",
                    }
                }
            ],
        },
    ]

    by_case, by_patient = index_ssm_hits_by_case_and_patient(hits)

    assert set(by_case.keys()) == {"case-1", "case-2"}
    assert set(by_patient.keys()) == {"TCGA-AA-0001", "TCGA-BB-0002"}
    assert by_case["case-1"][0]["ssm_id"] == "ssm-a"
    assert by_case["case-1"][0]["gene_symbols"] == ["VHL"]
    assert by_case["case-2"][0]["consequence_terms"] == ["stop_gained"]


def test_fetch_ssm_hits_uses_verified_gdc_fields(monkeypatch) -> None:
    client = GDCClient()
    calls: list[dict[str, Any]] = []
    expected_hit = {
        "ssm_id": "ssm-c",
        "mutation_subtype": "Single base substitution",
        "occurrence": [
            {
                "case": {
                    "case_id": "case-3",
                    "submitter_id": "TCGA-CC-0003",
                    "project": {"project_id": "TCGA-GBM"},
                }
            }
        ],
        "consequence": [
            {
                "transcript": {
                    "gene": {"symbol": "EGFR"},
                    "consequence_type": "missense_variant",
                }
            }
        ],
    }

    def fake_post_hits(endpoint: str, payload: dict[str, Any], max_records: int | None = None) -> list[dict[str, Any]]:
        calls.append(payload)
        assert endpoint == "ssms"
        assert payload["fields"]
        assert max_records == 50
        return [expected_hit]

    monkeypatch.setattr(client, "_post_hits", fake_post_hits)

    hits = client.fetch_ssm_hits(project_ids=["TCGA-GBM"], gene_symbols=["EGFR"], max_hits=50)

    assert hits == [expected_hit]
    assert len(calls) == 1
    assert calls[0]["filters"]["content"][0]["content"]["field"] == "occurrence.case.project.project_id"
    assert calls[0]["filters"]["content"][1]["content"]["field"] == "consequence.transcript.gene.symbol"
    assert "occurrence.case.submitter_id" in calls[0]["fields"]

    by_case, by_patient = index_ssm_hits_by_case_and_patient(hits)

    assert by_case["case-3"][0]["mutation_type"] == "Single base substitution"
    assert by_patient["TCGA-CC-0003"][0]["gene_symbols"] == ["EGFR"]
