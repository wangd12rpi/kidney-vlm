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

import re
import unicodedata
from typing import Any, Optional


REWARD_NAME = "oncovlm_pathology_mcq"
REWARD_TYPE = "sequential"

ACCURACY_WEIGHT = 0.9
FORMAT_WEIGHT = 0.1

_STRICT_FORMAT = re.compile(
    r"\A<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\Z",
    re.DOTALL,
)
_ANSWER_ENVELOPE = re.compile(
    r"\A(?:<think>.*?</think>\s*)?<answer>(?P<answer>.*?)</answer>\Z",
    re.DOTALL,
)
_CHOICE_LETTERS = {"a", "b", "c", "d"}


def normalize_option(value: str) -> str:
    """Apply only the normalization allowed for exact raw-option matching."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _has_exactly_one_tag_pair(response: str, tag: str) -> bool:
    return response.count(f"<{tag}>") == 1 and response.count(f"</{tag}>") == 1


def _extract_single_answer(response: str) -> Optional[str]:
    if not _has_exactly_one_tag_pair(response, "answer"):
        return None

    match = _ANSWER_ENVELOPE.fullmatch(response)
    if match is None:
        return None

    answer = match.group("answer").strip()
    return answer or None


def format_reward(response: str) -> float:
    if not _has_exactly_one_tag_pair(response, "think") or not _has_exactly_one_tag_pair(response, "answer"):
        return 0.0

    match = _STRICT_FORMAT.fullmatch(response)
    if match is None or not match.group("think").strip() or not match.group("answer").strip():
        return 0.0

    return 1.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = _extract_single_answer(response)
    if answer is None:
        return 0.0

    normalized_answer = normalize_option(answer)
    if normalized_answer in _CHOICE_LETTERS:
        return 0.0

    return 1.0 if normalized_answer == normalize_option(ground_truth) else 0.0


def compute_score(reward_input: dict[str, Any]) -> dict[str, float]:
    """Score one EasyR1 response using exact-option accuracy and strict formatting."""
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    format_score = format_reward(reward_input["response"])
    return {
        "overall": ACCURACY_WEIGHT * accuracy_score + FORMAT_WEIGHT * format_score,
        "accuracy": accuracy_score,
        "format": format_score,
    }
