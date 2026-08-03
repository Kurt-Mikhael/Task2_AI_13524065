from data import EDGES, GROUPS


def route_time(route):
    return sum(EDGES[e]["waktu"] for e in route)


def arrival_time(route, delay):
    return delay + route_time(route)


def simulate_flow(state):
    sim = {}
    for idx, (route, delay) in enumerate(state):
        t = delay
        node = GROUPS[idx]["asal"]
        for e in route:
            info = EDGES[e]
            if node == info["a"]:
                fwd = True
                node = info["b"]
            else:
                fwd = False
                node = info["a"]
            for minute in range(t, t + info["waktu"]):
                entry = sim.setdefault((e, minute), {"occ": 0, "fwd": 0, "bwd": 0})
                entry["occ"] += GROUPS[idx]["jumlah"]
                if fwd:
                    entry["fwd"] += 1
                else:
                    entry["bwd"] += 1
            t += info["waktu"]
    return sim
