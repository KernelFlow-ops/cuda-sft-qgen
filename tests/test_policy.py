"""Shared pipeline policy helpers used by sequential routing and the coordinator."""

from __future__ import annotations

from types import SimpleNamespace

from cuda_sft_qgen.graph.policy import chair_flush_size, chair_review_trigger, pending_room


def test_pending_room_caps_new_briefs_under_hard_limit():
    retry_room, room = pending_room(pending_n=3, active_n=2, capacity=6, total_capacity=8)
    assert retry_room == 4
    assert room == 4


def test_pending_room_zero_when_over_total_capacity():
    retry_room, room = pending_room(pending_n=10, active_n=4, capacity=6, total_capacity=8)
    assert retry_room == 0
    assert room == 0


def test_chair_review_trigger_hands_off_first_item():
    settings = SimpleNamespace(review_batch_size=20, review={"chair_max_workers": 4})
    assert chair_review_trigger(settings) == 1
    settings.review["chair_trigger_count"] = 3
    assert chair_review_trigger(settings) == 3
    settings.review_batch_size = 2
    assert chair_review_trigger(settings) == 2


def test_chair_flush_size_caps_below_review_batch():
    settings = SimpleNamespace(review_batch_size=20, review={"chair_max_workers": 4})
    assert chair_flush_size(settings) == 8
    settings.review["chair_flush_max"] = 6
    assert chair_flush_size(settings) == 6
    settings.review_batch_size = 2
    assert chair_flush_size(settings) == 2
