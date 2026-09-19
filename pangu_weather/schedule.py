"""Canonical greedy paths and a prefix-sharing execution graph."""
from collections import Counter
from dataclasses import dataclass

from .constants import STEPS


def greedy_steps(hours: int) -> tuple[int, ...]:
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 0:
        raise ValueError("lead time must be a nonnegative integer")
    result = []
    for step in STEPS:
        count, hours = divmod(hours, step)
        result.extend([step] * count)
    return tuple(result)


@dataclass(frozen=True)
class Node:
    lead: int
    parent: int
    step: int


def build_schedule(leads) -> list[Node]:
    nodes = {}
    for lead in leads:
        if lead == 0:
            raise ValueError("forecast outputs must have positive lead times")
        parent = 0
        for step in greedy_steps(lead):
            current = parent + step
            node = Node(current, parent, step)
            if current in nodes and nodes[current] != node:
                raise AssertionError("conflicting greedy prefixes")
            nodes[current] = node
            parent = current
    return [nodes[key] for key in sorted(nodes)]


def predict(initial_state, leads, runtime):
    """Yield requested states in time order, retaining only live dependencies."""
    nodes = build_schedule(leads)
    requested = set(leads)
    remaining = Counter(node.parent for node in nodes)
    states = {0: initial_state}
    del initial_state
    for node in nodes:
        state = runtime.run(node.step, states[node.parent])
        remaining[node.parent] -= 1
        if remaining[node.parent] == 0:
            del states[node.parent]
        if remaining[node.lead]:
            states[node.lead] = state
        if node.lead in requested:
            yield node.lead, state
        del state
