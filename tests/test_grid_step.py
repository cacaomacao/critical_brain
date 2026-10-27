"""Tiny arithmetic fixtures for the opt-in contract; no scientific runs."""

from dataclasses import replace
import unittest

from grid_step import GridConfig, GridState, SynapticWeights, step


def fixture():
    config = GridConfig(
        grid_shape=(2, 2),
        neuron_types=("I", "E", "I", "E"),
        connectivity=((False, True, True, False),
                      (True, False, False, True),
                      (False, False, False, True),
                      (False, False, True, False)),
        weights=SynapticWeights(
            excitatory_to_excitatory=1, excitatory_to_inhibitory=4,
            inhibitory_to_excitatory=-2, inhibitory_to_inhibitory=-3),
        dt_ms=0.5, tau_excitatory_ms=2, tau_inhibitory_ms=4,
        tau_synaptic_ms=2, threshold=1, reset_excitatory=-2, reset_inhibitory=-3)
    state = GridState((0, 0.5, -0.5, 1), (0, 2, -2, 0),
                      (True, True, False, False))
    return config, state, (0, 1, 0, 0)


class GridTransition(unittest.TestCase):
    def test_documented_transition_all_four_weights_and_interleaved_identities(self):
        config, state, external = fixture()
        self.assertEqual(step(config, state, external), GridState(
            (1, -2, -1.9375, 1), (2, 0.5, -3, 0.5),
            (False, True, False, False)))
        # The two values exactly at threshold do not spike. The E reset does.
        self.assertEqual(state.voltage, (0, 0.5, -0.5, 1))
        self.assertEqual(state.spikes, (True, True, False, False))

    def test_dt_scales_leak_drive_and_membrane_update(self):
        config, state, external = fixture()
        self.assertEqual(step(replace(config, dt_ms=0.25), state, external), GridState(
            (0.25, 1, -1.09375, 0.9375), (1, 1.25, -2.5, 0.25),
            (False, False, False, False)))

    def test_new_spikes_arrive_only_on_next_call(self):
        config, _, _ = fixture()
        config = replace(config, grid_shape=(1, 2), neuron_types=("E", "E"),
                         connectivity=((False, True), (False, False)),
                         weights=SynapticWeights(4, 0, 0, 0))
        state = GridState((0.9, 0), (0, 0), (False, False))
        first = step(config, state, (1, 0))
        self.assertEqual(first.spikes, (True, False))
        self.assertEqual(first.synaptic_current, (0, 0))
        self.assertEqual(first.voltage, (-2, 0))
        second = step(config, first, (0, 0))
        self.assertEqual(second.synaptic_current, (0, 2))
        self.assertEqual(second.voltage, (-1.5, 1))
        self.assertEqual(second.spikes, (False, False))

    def test_previous_mask_drives_even_after_reset_or_without_high_voltage(self):
        config, state, external = fixture()
        reset_state = replace(state, voltage=(-3, -2, -0.5, 1))
        self.assertEqual(step(config, reset_state, external).synaptic_current,
                         (2, 0.5, -3, 0.5))
        high_voltage = replace(state, voltage=(2, 2, 2, 2),
                               spikes=(False, False, False, False))
        self.assertEqual(step(config, high_voltage, external).synaptic_current,
                         (0, 1.5, -1.5, 0))

    def test_inhibitory_reset_uses_same_identity_as_membrane_decay(self):
        config, _, _ = fixture()
        state = GridState((2, 0, 0, 0), (0, 0, 0, 0),
                          (False, False, False, False))
        result = step(config, state, (0, 0, 0, 0))
        self.assertEqual(result.voltage, (-3, 0, 0, 0))
        self.assertEqual(result.spikes, (True, False, False, False))

    def test_relabeling_cells_does_not_change_transition(self):
        config, state, external = fixture()
        order = (3, 0, 1, 2)
        permuted_config = replace(config,
            neuron_types=tuple(config.neuron_types[i] for i in order),
            connectivity=tuple(tuple(config.connectivity[i][j] for j in order)
                               for i in order))
        permuted_state = GridState(tuple(state.voltage[i] for i in order),
                                   tuple(state.synaptic_current[i] for i in order),
                                   tuple(state.spikes[i] for i in order))
        expected = step(config, state, external)
        actual = step(permuted_config, permuted_state, tuple(external[i] for i in order))
        self.assertEqual(actual.voltage, tuple(expected.voltage[i] for i in order))
        self.assertEqual(actual.synaptic_current,
                         tuple(expected.synaptic_current[i] for i in order))
        self.assertEqual(actual.spikes, tuple(expected.spikes[i] for i in order))

    def test_explicit_self_edge_also_observes_one_bin_delay(self):
        config, _, _ = fixture()
        config = replace(config, grid_shape=(1, 1), neuron_types=("E",),
                         connectivity=((True,),))
        first = step(config, GridState((0.9,), (0,), (False,)), (1,))
        self.assertEqual(first.synaptic_current, (0,))
        self.assertEqual(first.spikes, (True,))
        self.assertEqual(step(config, first, (0,)).synaptic_current, (0.5,))

    def test_sequence_inputs_are_snapshots(self):
        config, _, _ = fixture()
        identities = list(config.neuron_types)
        edges = [list(row) for row in config.connectivity]
        copied = replace(config, neuron_types=identities, connectivity=edges)
        voltage = [0, 0, 0, 0]
        mask = [False, False, False, False]
        state = GridState(voltage, voltage, mask)
        identities[0] = "E"
        edges[0][1] = False
        voltage[0] = 10
        mask[0] = True
        self.assertEqual(copied, config)
        self.assertEqual(state.voltage, (0, 0, 0, 0))
        self.assertEqual(state.spikes, (False, False, False, False))

    def test_rejects_invalid_configurations(self):
        config, _, _ = fixture()
        invalid = (dict(grid_shape=(0, 4)), dict(grid_shape=(True, 4)),
                   dict(neuron_types=("E",) * 3), dict(neuron_types=("X",) * 4),
                   dict(connectivity=((False,),) * 4),
                   dict(connectivity=((0, 0, 0, 0),) * 4),
                   dict(dt_ms=0), dict(tau_synaptic_ms=-1),
                   dict(threshold=float("nan")), dict(reset_excitatory=float("inf")),
                   dict(weights=(1, 2, 3, 4)))
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(config, **change)
        with self.assertRaises(ValueError):
            SynapticWeights(1, 2, 3, float("inf"))

    def test_rejects_invalid_state_drive_and_mask(self):
        config, state, external = fixture()
        for change in (dict(voltage=(0,)), dict(spikes=(0, 0, 0, 0)),
                       dict(synaptic_current=(0, float("nan"), 0, 0))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(state, **change)
        for bad_drive in ((0,), (0, 0, float("inf"), 0), 0):
            with self.subTest(drive=bad_drive), self.assertRaises(ValueError):
                step(config, state, bad_drive)
        with self.assertRaises(ValueError):
            step(config, GridState((0,), (0,), (False,)), external)

    def test_rejects_overflow_before_reset_can_hide_it(self):
        config, state, _ = fixture()
        with self.assertRaises(ValueError):
            step(replace(config, dt_ms=1e308), state, (1e308, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
