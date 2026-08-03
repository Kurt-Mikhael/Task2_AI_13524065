import random

from algorithms import (genetic_algorithm, hill_climbing_basic,
                        hill_climbing_sideways, hill_climbing_stochastic,
                        random_restart_hill_climbing, simulated_annealing)
from objective import objective_function
from state import generate_initial_state
from view import (print_penalties, print_search_visualization, print_simulation,
                  print_state, print_summary)

random.seed(42)


def main():
    print("=" * 64)
    print("CAMPUSFLOW - PROOF OF CONCEPT")
    print("Optimasi Pergerakan Kelompok Mahasiswa Menggunakan Local Search")
    print("=" * 64)

    initial = generate_initial_state()
    print("\n=== INITIAL STATE (random) ===")
    print_state(initial)
    print()
    print_penalties(initial)
    print("\n=== VISUALISASI SIMULASI INITIAL STATE ===")
    print_simulation(initial)

    print("\n" + "=" * 64)
    print("1. VARIAN HILL-CLIMBING")
    print("=" * 64)
    basic_state, basic_history, basic_states = hill_climbing_basic(
        initial, return_states=True
    )
    sideways_state, sideways_history, sideways_states = hill_climbing_sideways(
        initial, return_states=True
    )
    stochastic_state, stochastic_history, stochastic_states = hill_climbing_stochastic(
        initial, return_states=True
    )
    hc_state, hc_histories, hc_state_histories = random_restart_hill_climbing(
        restarts=6, return_states=True
    )
    hill_variants = [
        ("Basic (Steepest-Ascent)", basic_state, basic_history, basic_states),
        ("Sideways Move", sideways_state, sideways_history, sideways_states),
        ("Stochastic", stochastic_state, stochastic_history, stochastic_states),
    ]
    for name, state, history, states in hill_variants:
        print(f"\n{name}")
        print("Cost: " + " -> ".join(f"{cost:.2f}" for cost in history))
        print_search_visualization(name, states, history)
    print("\nRandom Restart")
    for i, h in enumerate(hc_histories):
        trace = " -> ".join(f"{c:.2f}" for c in h)
        print(f"Restart {i + 1}: {trace}")
    print(f"\nBest cost setelah 6 restart : {objective_function(hc_state):.2f}")
    best_restart_index = min(
        range(len(hc_histories)),
        key=lambda index: hc_histories[index][-1],
    )
    print_search_visualization(
        "Random Restart",
        hc_state_histories[best_restart_index],
        hc_histories[best_restart_index],
    )
    print("\n=== FINAL STATE (Hill Climbing) ===")
    print_state(hc_state)
    print()
    print_penalties(hc_state)

    print("\n" + "=" * 64)
    print("2. SIMULATED ANNEALING")
    print("=" * 64)
    sa_state, sa_history = simulated_annealing(initial)
    print("Iterasi 0..9      : " + " ".join(f"{c:.2f}" for c in sa_history[:10]))
    print("Tengah simulasi   : " + " ".join(f"{c:.2f}" for c in sa_history[len(sa_history) // 2:len(sa_history) // 2 + 5]))
    print("Akhir simulasi    : " + " ".join(f"{c:.2f}" for c in sa_history[-5:]))
    print(f"Total iterasi     : {len(sa_history)}")
    print("\n=== FINAL STATE (Simulated Annealing) ===")
    print_state(sa_state)
    print()
    print_penalties(sa_state)

    print("\n" + "=" * 64)
    print("3. GENETIC ALGORITHM")
    print("=" * 64)
    ga_state, ga_history = genetic_algorithm()
    print("Best fitness per generasi (setiap 10):")
    for i in range(0, len(ga_history), 10):
        print(f"Generasi {i:<4} : {ga_history[i]:.2f}")
    print(f"Generasi {len(ga_history) - 1:<4} : {ga_history[-1]:.2f}")
    print("\n=== FINAL STATE (Genetic Algorithm) ===")
    print_state(ga_state)
    print()
    print_penalties(ga_state)

    candidates = [("Basic Hill Climbing", basic_state),
                  ("Sideways Move Hill Climbing", sideways_state),
                  ("Stochastic Hill Climbing", stochastic_state),
                  ("Random-Restart Hill Climbing", hc_state),
                  ("Simulated Annealing", sa_state),
                  ("Genetic Algorithm", ga_state)]
    best_name, best_state = min(candidates, key=lambda t: objective_function(t[1]))

    print("\n" + "=" * 64)
    print("4. RINGKASAN PERBANDINGAN ALGORITMA")
    print("=" * 64)
    for name, state in candidates:
        print(f"{name:<30} : cost akhir {objective_function(state):.2f}")
    print(f"\nSolusi terbaik: {best_name}")

    print("\n=== FINAL STATE (solusi terbaik) ===")
    print_state(best_state)
    print()
    print_penalties(best_state)
    print("\n=== VISUALISASI SIMULASI FINAL STATE ===")
    print_simulation(best_state)

    print("\n=== RINGKASAN HASIL ===")
    hc_iters = len(min(hc_histories, key=len))
    summary_data = [
        ("Basic Hill Climbing", basic_state, len(basic_history)),
        ("Sideways Move Hill Climbing", sideways_state, len(sideways_history)),
        ("Stochastic Hill Climbing", stochastic_state, len(stochastic_history)),
        ("Random-Restart Hill Climbing", hc_state, hc_iters),
        ("Simulated Annealing", sa_state, len(sa_history)),
        ("Genetic Algorithm", ga_state, len(ga_history)),
    ]
    for name, state, iters in summary_data:
        print_summary(name, objective_function(initial), objective_function(state), iters)


if __name__ == "__main__":
    main()
