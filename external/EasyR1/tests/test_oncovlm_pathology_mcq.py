# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path

import pytest


REWARD_PATH = Path(__file__).parents[1] / "examples" / "reward_function" / "oncovlm_pathology_mcq.py"
SPEC = importlib.util.spec_from_file_location("oncovlm_pathology_mcq", REWARD_PATH)
assert SPEC is not None and SPEC.loader is not None
reward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward)

GROUND_TRUTH = "Clear cell renal cell carcinoma"


def score(response: str, ground_truth: str = GROUND_TRUTH) -> dict[str, float]:
    return reward.compute_score(
        {
            "response": response,
            "response_length": len(response),
            "ground_truth": ground_truth,
        }
    )


def test_correct_exact_option_receives_full_reward():
    result = score(
        "<think>Nested vessels and clear cytoplasm support this diagnosis.</think>"
        "<answer>Clear cell renal cell carcinoma</answer>"
    )
    assert result == {"overall": 1.0, "accuracy": 1.0, "format": 1.0}


def test_accuracy_normalizes_nfkc_case_and_whitespace_only():
    result = score(
        "<think>The morphology is concordant.</think>"
        "<answer>ＣＬＥＡＲ   CELL\nRENAL CELL CARCINOMA</answer>"
    )
    assert result == {"overall": 1.0, "accuracy": 1.0, "format": 1.0}


@pytest.mark.parametrize(
    "answer",
    [
        "Papillary renal cell carcinoma",
        "A",
        "Clear cell RCC",
    ],
)
def test_wrong_letter_and_paraphrased_answers_receive_only_format_reward(answer: str):
    result = score(f"<think>Visual evidence.</think><answer>{answer}</answer>")
    assert result == {"overall": 0.1, "accuracy": 0.0, "format": 1.0}


def test_bare_correct_answer_is_rejected():
    assert score(GROUND_TRUTH) == {"overall": 0.0, "accuracy": 0.0, "format": 0.0}


def test_missing_answer_tag_is_rejected():
    result = score(f"<think>Visual evidence.</think>{GROUND_TRUTH}")
    assert result == {"overall": 0.0, "accuracy": 0.0, "format": 0.0}


def test_correct_answer_without_think_tag_keeps_only_accuracy_reward():
    result = score(f"<answer>{GROUND_TRUTH}</answer>")
    assert result == {"overall": 0.9, "accuracy": 1.0, "format": 0.0}


def test_multiple_answer_tags_are_rejected():
    result = score(
        f"<think>Visual evidence.</think><answer>{GROUND_TRUTH}</answer>"
        f"<answer>{GROUND_TRUTH}</answer>"
    )
    assert result == {"overall": 0.0, "accuracy": 0.0, "format": 0.0}


def test_empty_think_loses_format_but_keeps_accuracy_reward():
    result = score(f"<think>  </think><answer>{GROUND_TRUTH}</answer>")
    assert result == {"overall": 0.9, "accuracy": 1.0, "format": 0.0}


def test_text_after_answer_is_rejected_for_accuracy_and_format():
    result = score(f"<think>Visual evidence.</think><answer>{GROUND_TRUTH}</answer> extra")
    assert result == {"overall": 0.0, "accuracy": 0.0, "format": 0.0}


def test_text_before_think_is_rejected_for_accuracy_and_format():
    result = score(f"extra <think>Visual evidence.</think><answer>{GROUND_TRUTH}</answer>")
    assert result == {"overall": 0.0, "accuracy": 0.0, "format": 0.0}


def test_correct_answer_and_format_weights_are_independent():
    accuracy_only = score(f"<answer>{GROUND_TRUTH}</answer>")
    format_only = score("<think>Visual evidence.</think><answer>Wrong diagnosis</answer>")
    assert accuracy_only["overall"] == pytest.approx(0.9)
    assert format_only["overall"] == pytest.approx(0.1)
