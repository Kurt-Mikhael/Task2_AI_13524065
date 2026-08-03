import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, "src/local_search")

import algorithms
from objective import objective_function
from state import generate_initial_state, is_valid_state, random_neighbor
from view import print_search_visualization


class LocalSearchMainCaseTests(unittest.TestCase):
    def setUp(self):
        algorithms.random.seed(42)
        self.initial = generate_initial_state()

    def test_all_hill_climbing_variants_return_valid_states(self):
        searches = [
            algorithms.hill_climbing_basic,
            algorithms.hill_climbing_sideways,
            algorithms.hill_climbing_stochastic,
        ]
        for search in searches:
            state, history, states = search(
                self.initial, max_iterations=50, return_states=True
            )
            self.assertTrue(is_valid_state(state))
            self.assertEqual(len(history), len(states))
            self.assertGreaterEqual(len(history), 1)
            self.assertLessEqual(history[-1], history[0])

    def test_basic_hill_climbing_never_increases_cost(self):
        _, history = algorithms.hill_climbing_basic(self.initial)
        self.assertEqual(history, sorted(history, reverse=True))

    def test_random_restart_returns_each_restart_history(self):
        state, histories, state_histories = algorithms.random_restart_hill_climbing(
            restarts=3, max_iterations=30, return_states=True
        )
        self.assertTrue(is_valid_state(state))
        self.assertEqual(len(histories), 3)
        self.assertEqual(len(state_histories), 3)
        self.assertTrue(all(history for history in histories))

    def test_visualization_prints_cost_and_state_changes(self):
        state, history, states = algorithms.hill_climbing_basic(
            self.initial, return_states=True
        )
        output = io.StringIO()
        with redirect_stdout(output):
            print_search_visualization("Basic", states, history)
        self.assertIn("VISUALISASI PENCARIAN", output.getvalue())
        self.assertIn("Iterasi", output.getvalue())
        self.assertIn(f"{objective_function(state):.2f}", output.getvalue())


class LocalSearchEdgeCaseTests(unittest.TestCase):
    def test_empty_state_is_supported(self):
        empty = tuple()
        self.assertEqual(objective_function(empty), 0)
        self.assertEqual(random_neighbor(empty), empty)
        for search in (
            algorithms.hill_climbing_basic,
            algorithms.hill_climbing_sideways,
            algorithms.hill_climbing_stochastic,
        ):
            state, history = search(empty)
            self.assertEqual(state, empty)
            self.assertEqual(history, [0])

    def test_no_neighbors_stops_without_error(self):
        state = self._state()
        with patch.object(algorithms, "generate_neighbors", return_value=[]):
            result, history = algorithms.hill_climbing_basic(state)
        self.assertEqual(len(history), 1)
        self.assertEqual(result, state)

    def test_invalid_restart_count_is_rejected(self):
        with self.assertRaises(ValueError):
            algorithms.random_restart_hill_climbing(restarts=0)

    def _state(self):
        return generate_initial_state()


if __name__ == "__main__":
    unittest.main()
