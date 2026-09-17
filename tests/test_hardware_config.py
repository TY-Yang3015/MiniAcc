import unittest
from pathlib import Path

from miniacc_bootstrap import derive_allocator_ceiling
from miniacc_core.config import HardwareConfig, load_hardware_config


class HardwareConfigTests(unittest.TestCase):
    def test_profiles_are_explicit_and_valid(self):
        rtx = load_hardware_config(Path("exp_configs/rtx4090.yaml"))
        a100 = load_hardware_config(Path("exp_configs/wolf8-a100-4x.yaml"))
        self.assertEqual((rtx.gpu_count, rtx.allocator_gib, rtx.vram_mode), (1, 16.0, "novram"))
        self.assertEqual((a100.gpu_count, a100.world_size, a100.parallel_backend), (4, 4, "independent_jobs"))
        self.assertEqual((a100.vram_headroom_gib, a100.host_headroom_gib), (8.0, 16.0))

    def test_invalid_profile_cannot_exhaust_device_headroom(self):
        with self.assertRaises(ValueError):
            HardwareConfig("unsafe", 1, 24.0, 18.0, 8.0)
        with self.assertRaises(RuntimeError):
            derive_allocator_ceiling(70.0, 7 * 1024**3, 8 * 1024**3)
        self.assertEqual(
            derive_allocator_ceiling(71.0, int(79.25 * 1024**3), 8 * 1024**3), 71.0
        )

    def test_launcher_gates_failed_forward_and_uses_isolated_jobs(self):
        source = Path("scripts/wolf8_independent_reference.sh").read_text()
        self.assertIn("miniacc_dit_forward_error", source)
        self.assertIn("abort_owned", source)
        self.assertIn("--port $((8188 + $gpu))", source)
        self.assertIn("--hardware-config", source)
        self.assertIn("serialized_first_forward", Path("exp_configs/wolf8-a100-4x.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
