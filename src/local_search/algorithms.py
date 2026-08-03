import math
import random

from objective import objective_function
from state import generate_initial_state, generate_neighbors, is_valid_state, random_neighbor


def _search_result(current, costs, states, return_states):
    if return_states:
        return current, costs, states
    return current, costs


def hill_climbing_basic(initial, max_iterations=1000, return_states=False):
    current = initial
    current_cost = objective_function(current)
    costs = [current_cost]
    states = [current]

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current)
        if not neighbors:
            break
        best = min(neighbors, key=objective_function)
        best_cost = objective_function(best)
        if best_cost >= current_cost:
            break
        current = best
        current_cost = best_cost
        costs.append(current_cost)
        states.append(current)

    return _search_result(current, costs, states, return_states)


def hill_climbing_sideways(initial, max_iterations=1000, max_sideways=10,
                           return_states=False):
    current = initial
    current_cost = objective_function(current)
    costs = [current_cost]
    states = [current]
    sideways_used = 0

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current)
        if not neighbors:
            break
        best_cost = min(objective_function(nbr) for nbr in neighbors)
        candidates = [
            nbr for nbr in neighbors
            if objective_function(nbr) == best_cost
        ]
        if best_cost > current_cost:
            break
        if best_cost == current_cost:
            if sideways_used >= max_sideways:
                break
            sideways_used += 1
        else:
            sideways_used = 0
        current = random.choice(candidates)
        current_cost = best_cost
        costs.append(current_cost)
        states.append(current)

    return _search_result(current, costs, states, return_states)


def hill_climbing_stochastic(initial, max_iterations=1000, return_states=False):
    current = initial
    current_cost = objective_function(current)
    costs = [current_cost]
    states = [current]

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current)
        improving = [
            nbr for nbr in neighbors
            if objective_function(nbr) < current_cost
        ]
        if not improving:
            break
        current = random.choice(improving)
        current_cost = objective_function(current)
        costs.append(current_cost)
        states.append(current)

    return _search_result(current, costs, states, return_states)


def hill_climbing(initial, max_iterations=1000, return_states=False):
    return hill_climbing_basic(initial, max_iterations, return_states)


def random_restart_hill_climbing(restarts=6, max_iterations=1000,
                                 return_states=False):
    if restarts <= 0:
        raise ValueError("restarts harus lebih besar dari 0")
    best_state = None
    best_cost = float("inf")
    all_histories = []
    all_states = []
    for _ in range(restarts):
        start = generate_initial_state()
        final, history, states = hill_climbing_basic(
            start, max_iterations=max_iterations, return_states=True
        )
        all_histories.append(history)
        all_states.append(states)
        if objective_function(final) < best_cost:
            best_cost = objective_function(final)
            best_state = final
    if return_states:
        return best_state, all_histories, all_states
    return best_state, all_histories


def simulated_annealing(initial, t0=120.0, alpha=0.995, min_temp=0.5):
    current = initial
    temp = t0
    history = []
    while temp > min_temp:
        nbr = random_neighbor(current)
        delta = objective_function(nbr) - objective_function(current)
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = nbr
        history.append(objective_function(current))
        temp *= alpha
    return current, history


def initialize_population(size=40):
    return [generate_initial_state() for _ in range(size)]


def fitness(chromosome):
    return 1.0 / (1.0 + objective_function(chromosome))


def tournament_selection(population, k=3):
    best = None
    for _ in range(k):
        candidate = random.choice(population)
        if best is None or fitness(candidate) > fitness(best):
            best = candidate
    return best


def crossover(parent_1, parent_2):
    cut = random.randint(1, len(parent_1) - 1)
    child_1 = parent_1[:cut] + parent_2[cut:]
    child_2 = parent_2[:cut] + parent_1[cut:]
    if not is_valid_state(child_1):
        child_1 = parent_1
    if not is_valid_state(child_2):
        child_2 = parent_2
    return child_1, child_2


def mutate(chromosome):
    return random_neighbor(chromosome)


def genetic_algorithm(population_size=40, generations=60, mutation_rate=0.4, elite_size=4):
    population = initialize_population(population_size)
    history = []
    for _ in range(generations):
        population.sort(key=objective_function)
        history.append(objective_function(population[0]))
        new_population = population[:elite_size]
        while len(new_population) < population_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = crossover(p1, p2)
            child = c1 if fitness(c1) >= fitness(c2) else c2
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = new_population
    population.sort(key=objective_function)
    return population[0], history
