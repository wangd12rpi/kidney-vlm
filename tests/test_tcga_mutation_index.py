from __future__ import annotations

from kidney_vlm.data.sources.tcga import index_ssm_hits_by_case_and_patient


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
