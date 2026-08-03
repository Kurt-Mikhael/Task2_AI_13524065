import random

from data import GROUPS, MAX_DELAY, HARD_TOLERANCE
from simulation import arrival_time


def is_valid_state(state):
    for group, (route, delay) in zip(GROUPS, state):
        if route not in group["alternatif"]:
            return False
        if not (0 <= delay <= MAX_DELAY):
            return False
        if arrival_time(route, delay) > group["deadline"] + HARD_TOLERANCE:
            return False
    return True


def generate_initial_state():
    while True:
        state = tuple(
            (random.choice(g["alternatif"]), random.randint(0, MAX_DELAY))
            for g in GROUPS
        )
        if is_valid_state(state):
            return state


def generate_neighbors(state):
    neighbors = []
    n = len(state)
    for i in range(n):
        route, delay = state[i]
        for delta in (-1, 1):
            nd = delay + delta
            if 0 <= nd <= MAX_DELAY:
                nbr = list(state)
                nbr[i] = (route, nd)
                nbr = tuple(nbr)
                if is_valid_state(nbr):
                    neighbors.append(nbr)
    for i in range(n):
        route, delay = state[i]
        for alt in GROUPS[i]["alternatif"]:
            if alt != route:
                nbr = list(state)
                nbr[i] = (alt, delay)
                nbr = tuple(nbr)
                if is_valid_state(nbr):
                    neighbors.append(nbr)
    for i in range(n):
        for j in range(i + 1, n):
            nbr = list(state)
            nbr[i] = (state[j][0], state[i][1])
            nbr[j] = (state[i][0], state[j][1])
            nbr = tuple(nbr)
            if is_valid_state(nbr):
                neighbors.append(nbr)
    for i in range(n):
        for j in range(i + 1, n):
            nbr = list(state)
            nbr[i] = (state[i][0], state[j][1])
            nbr[j] = (state[j][0], state[i][1])
            nbr = tuple(nbr)
            if is_valid_state(nbr):
                neighbors.append(nbr)
    return neighbors


def random_neighbor(state):
    n = len(state)
    if n < 2:
        return state
    for _ in range(60):
        nbr = list(state)
        move = random.randint(0, 3)
        if move == 0:
            i = random.randrange(n)
            route, delay = state[i]
            delta = random.choice((-1, 1))
            nd = delay + delta
            if 0 <= nd <= MAX_DELAY:
                nbr[i] = (route, nd)
        elif move == 1:
            i = random.randrange(n)
            route, delay = state[i]
            alts = [a for a in GROUPS[i]["alternatif"] if a != route]
            if alts:
                nbr[i] = (random.choice(alts), delay)
        elif move == 2:
            i, j = random.sample(range(n), 2)
            nbr[i] = (state[j][0], state[i][1])
            nbr[j] = (state[i][0], state[j][1])
        else:
            i, j = random.sample(range(n), 2)
            nbr[i] = (state[i][0], state[j][1])
            nbr[j] = (state[j][0], state[i][1])
        nbr = tuple(nbr)
        if nbr != state and is_valid_state(nbr):
            return nbr
    return state
