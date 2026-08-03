from data import EDGES, GROUPS
from objective import (
    calculate_capacity_penalty,
    calculate_conflict_penalty,
    calculate_distance_penalty,
    calculate_fairness_penalty,
    calculate_lateness_penalty,
    calculate_waiting_penalty,
    objective_function,
)
from simulation import simulate_flow


def route_name(route):
    return " -> ".join(EDGES[e]["nama"] for e in route)


def print_state(state):
    for group, (route, delay) in zip(GROUPS, state):
        print(f"{group['nama']:<11} : {route_name(route)} (delay {delay})")


def print_penalties(state):
    sim = simulate_flow(state)
    print(f"Capacity penalty : {calculate_capacity_penalty(sim)}")
    print(f"Conflict penalty : {calculate_conflict_penalty(sim)}")
    print(f"Lateness penalty : {calculate_lateness_penalty(state)}")
    print(f"Waiting penalty  : {calculate_waiting_penalty(state)}")
    print(f"Distance penalty : {calculate_distance_penalty(state)}")
    print(f"Fairness penalty : {calculate_fairness_penalty(state):.2f}")
    print(f"Total objective cost: {objective_function(state):.2f}")


def print_simulation(state):
    from simulation import arrival_time

    horizon = max(arrival_time(r, d) for (r, d) in state) + 1
    sim = simulate_flow(state)
    for minute in range(horizon):
        print(f"MENIT {minute}")
        for e in EDGES:
            entry = sim.get((e, minute))
            occ = entry["occ"] if entry else 0
            cap = EDGES[e]["kapasitas"]
            filled = min(10, int(round(occ / cap * 10))) if cap else 0
            bar = "#" * filled + "." * (10 - filled)
            print(f"{EDGES[e]['nama']:<14} : [{bar}] {occ}/{cap}")


def print_summary(name, initial_cost, final_cost, iterations):
    improvement = (
        0 if initial_cost == 0 else (initial_cost - final_cost) / initial_cost * 100
    )
    print(f"Algorithm       : {name}")
    print(f"Initial cost    : {initial_cost:.2f}")
    print(f"Final cost      : {final_cost:.2f}")
    print(f"Improvement     : {improvement:.2f}%")
    print(f"Total iteration : {iterations}")


def print_search_visualization(name, state_history, cost_history, max_steps=25):
    print(f"\n=== VISUALISASI PENCARIAN: {name} ===")
    if not cost_history:
        print("Tidak ada iterasi.")
        return
    shown = min(len(cost_history), max_steps)
    ceiling = max(cost_history)
    floor = min(cost_history)
    span = max(ceiling - floor, 1e-12)
    for index in range(shown):
        cost = cost_history[index]
        width = int(round((cost - floor) / span * 36))
        changed = (
            "awal"
            if index == 0
            else _changed_groups(state_history[index - 1], state_history[index])
        )
        print(f"Iterasi {index:>3} | {cost:>8.2f} | {'#' * width:<36} | {changed}")
    if len(cost_history) > shown:
        print(f"... {len(cost_history) - shown} iterasi berikutnya disembunyikan")


def _changed_groups(previous, current):
    changed = [
        str(index + 1)
        for index, (before, after) in enumerate(zip(previous, current))
        if before != after
    ]
    return "kelompok " + ", ".join(changed) if changed else "tidak berubah"
