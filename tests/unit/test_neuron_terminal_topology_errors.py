import unittest

from emperor.neuron._terminal_topology import _connection_offsets


class TestNeuronTerminalTopologyErrors(unittest.TestCase):
    def test_unsupported_connection_shape_names_terminal_contract(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "^Unsupported terminal connection shape: 'invalid'$",
        ):
            _connection_offsets(None, "invalid")


if __name__ == "__main__":
    unittest.main()
