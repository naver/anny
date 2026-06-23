import unittest

import torch

import anny.torch_compat as torch_compat


class TorchCompatTest(unittest.TestCase):
    def test_make_buffer_registers_non_persistent_buffer(self):
        module = torch.nn.Module()
        tensor = torch.ones(2)

        module.weights = torch_compat.make_buffer(module, "weights", tensor, persistent=False)

        self.assertIn("weights", dict(module.named_buffers()))
        self.assertNotIn("weights", module.state_dict())

    def test_make_buffer_fallback_registers_non_persistent_buffer(self):
        original_torch_buffer = torch_compat._TORCH_BUFFER
        torch_compat._TORCH_BUFFER = None
        try:
            module = torch.nn.Module()
            tensor = torch.ones(2)

            module.weights = torch_compat.make_buffer(module, "weights", tensor, persistent=False)

            self.assertIn("weights", dict(module.named_buffers()))
            self.assertNotIn("weights", module.state_dict())
            self.assertIs(module.weights, tensor)
        finally:
            torch_compat._TORCH_BUFFER = original_torch_buffer


if __name__ == "__main__":
    unittest.main()
