"""Pure decision policies used by the AutoBot orchestration layer."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WatchdogDecision:
    is_stuck: bool
    watched_location: tuple[int, int]
    checked_at: float
    elapsed: float


@dataclass(frozen=True)
class AttackDirectionDecision:
    direction: str | None
    distance_left: float
    distance_right: float
    left_is_valid: bool
    right_is_valid: bool


@dataclass(frozen=True)
class ChannelDecision:
    should_change: bool
    center: tuple[int, int] | None
    movement: int | None = None
    is_supported_mode: bool = True


def evaluate_watchdog(
    player_location,
    watched_location,
    last_movement_at,
    movement_range,
    timeout,
    now,
):
    distance = sum(
        abs(current - previous)
        for current, previous in zip(player_location, watched_location)
    )
    elapsed = now - last_movement_at
    if distance > movement_range or elapsed > timeout:
        return WatchdogDecision(
            is_stuck=elapsed > timeout and distance <= movement_range,
            watched_location=player_location,
            checked_at=now,
            elapsed=elapsed,
        )
    return WatchdogDecision(False, watched_location, last_movement_at, elapsed)


def choose_attack_direction(
    monster_left,
    monster_right,
    player_location,
    decisive_margin=50,
):
    def metrics(monster, direction):
        if monster is None:
            return float("inf"), False
        x, y = monster["position"]
        width, height = monster["size"]
        center = (x + width // 2, y + height // 2)
        distance = abs(center[0] - player_location[0]) + abs(
            center[1] - player_location[1]
        )
        is_valid = (
            center[0] < player_location[0]
            if direction == "left"
            else center[0] > player_location[0]
        )
        return distance, is_valid

    distance_left, left_valid = metrics(monster_left, "left")
    distance_right, right_valid = metrics(monster_right, "right")
    direction = None
    if monster_left is not None and monster_right is None and left_valid:
        direction = "left"
    elif monster_right is not None and monster_left is None and right_valid:
        direction = "right"
    elif monster_left is not None and monster_right is not None:
        if left_valid and not right_valid:
            direction = "left"
        elif right_valid and not left_valid:
            direction = "right"
        elif left_valid and right_valid:
            if distance_left < distance_right - decisive_margin:
                direction = "left"
            elif distance_right < distance_left - decisive_margin:
                direction = "right"
    return AttackDirectionDecision(
        direction,
        distance_left,
        distance_right,
        left_valid,
        right_valid,
    )


def evaluate_other_players(locations, previous_center, mode, movement_threshold):
    if not locations:
        return ChannelDecision(False, previous_center)
    center_values = np.mean(locations, axis=0)
    if np.any(np.isnan(center_values)):
        return ChannelDecision(False, previous_center)
    center = tuple(map(int, center_values))
    if mode == "true":
        return ChannelDecision(True, previous_center)
    if mode != "pixel":
        return ChannelDecision(False, previous_center, is_supported_mode=False)
    if previous_center is None:
        return ChannelDecision(False, center)
    movement = sum(
        abs(current - previous)
        for current, previous in zip(center, previous_center)
    )
    return ChannelDecision(
        movement > movement_threshold,
        previous_center,
        movement,
    )


def evaluate_scheduled_switch(enabled, last_switch_at, interval, now):
    should_change = enabled and now - last_switch_at > interval
    return should_change, now if should_change else last_switch_at
