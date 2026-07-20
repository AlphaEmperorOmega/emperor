import re
import unittest

from unit.test_neuron import NeuronTestCase


class TestNeuronCompositionValidation(NeuronTestCase):
    def test_schema_validation_owns_child_config_type_errors(self) -> None:
        invalid_nucleus = self.neuron_config()
        invalid_nucleus.nucleus_config.model_config = object()
        invalid_terminal = self.terminal_config()
        invalid_terminal.sampler_config = object()

        cases = (
            (
                invalid_nucleus.build,
                "model_config must be ConfigBase for NucleusConfig, got object",
            ),
            (
                invalid_terminal.build,
                "sampler_config must be SamplerConfig for TerminalConfig, got object",
            ),
        )
        for build, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, re.escape(message)):
                    build()


if __name__ == "__main__":
    unittest.main()
