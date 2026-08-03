from data import EDGES, GROUPS, WEIGHTS
from simulation import simulate_flow, route_time, arrival_time


def calculate_capacity_penalty(sim):
    total = 0
    for (e, minute), entry in sim.items():
        over = entry["occ"] - EDGES[e]["kapasitas"]
        if over > 0:
            total += over * over
    return total


def calculate_conflict_penalty(sim):
    return sum(entry["fwd"] * entry["bwd"] for entry in sim.values())


def calculate_lateness_penalty(state):
    total = 0
    for group, (route, delay) in zip(GROUPS, state):
        late = arrival_time(route, delay) - group["deadline"]
        if late > 0:
            total += group["jumlah"] * late
    return total


def calculate_waiting_penalty(state):
    return sum(delay for _, delay in state)


def calculate_distance_penalty(state):
    return sum(route_time(route) for route, _ in state)


def calculate_fairness_penalty(state):
    delays = [delay for _, delay in state]
    if not delays:
        return 0
    mean = sum(delays) / len(delays)
    return sum((d - mean) ** 2 for d in delays) / len(delays)


def objective_function(state):
    sim = simulate_flow(state)
    return (
        WEIGHTS["capacity"] * calculate_capacity_penalty(sim)
        + WEIGHTS["conflict"] * calculate_conflict_penalty(sim)
        + WEIGHTS["late"] * calculate_lateness_penalty(state)
        + WEIGHTS["waiting"] * calculate_waiting_penalty(state)
        + WEIGHTS["distance"] * calculate_distance_penalty(state)
        + WEIGHTS["fairness"] * calculate_fairness_penalty(state)
    )
