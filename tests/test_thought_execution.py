"""Thought boundaries exercised through real survival rollout collectors."""

from contextlib import ExitStack
from dataclasses import fields, replace
import unittest
from unittest.mock import patch

import torch

from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import survival_loss_components
from self_genesis.world import Action


class ThoughtExecutionTests(unittest.TestCase):
    def collector(self, batched, steps=16, device='cpu'):
        torch.manual_seed(23)
        config = ExperimentConfig(
            num_agents=5, appearance_dim=2, encounter_count=2,
            initial_life=5, initial_points=1, survival_horizon=3,
            point_generation_probability_min=1,
            point_generation_probability_max=1, device=device)
        network = RecurrentPolicy(2, memory_dim=4, affect_dim=2,
                                  entity_memory_dim=3, think_steps=steps).double().to(device)
        # Positive, contractive dynamics avoid dead ReLUs or vanishing FP32
        # gradients obscuring a detach at an early think step.
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.fill_(0.02)
            network.thought.weight[:, -4:] = torch.eye(4, device=device) * 0.5
            network.thought.bias.fill_(0.1)
            network.action_head.weight[1].fill_(0.04)
            network.message_head.weight.copy_(torch.arange(
                1, 5, device=device, dtype=torch.double)[:, None].expand(4, 4) * 0.02)
        return (BatchedRolloutCollector(config, network, seeds=[17, 29]) if batched
                else RolloutCollector(config, network))

    def test_rollout_boundaries_and_complete_episode_gradients(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            for batched in (False, True):
                for steps in (16, 32, 64):
                    with self.subTest(device=device, batched=batched, steps=steps):
                        self.check_rollout(batched, steps, device)

    def check_rollout(self, batched, steps, device):
        collector = self.collector(batched, steps, device)
        network, world = collector.network, collector.world
        outputs, events, callbacks = [], [], []
        current = {}

        def snapshot():
            tensors = [getattr(world.state, f.name) for f in fields(world.state)]
            if batched:
                tensors += [world.steps, world._generation_rng.counters,
                            collector.protocol._random.counters]
            else:
                tensors += [world._generation_rng.get_state()]
            return [x.clone() for x in tensors]

        recur = network._recur

        def checked_recur(*args):
            current['world'] = snapshot()
            current['clock'] = None if batched else collector.elapsed_steps
            current['context'] = torch.cat(args, dim=-1).detach().clone()
            current['count'] = 0
            events.append('callback')
            result = recur(*args)
            self.assertEqual(current['count'], steps)
            return result

        def thought_hook(module, args, output):
            for actual, expected in zip(snapshot(), current['world']):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if not batched:
                self.assertEqual(collector.elapsed_steps, current['clock'])
            torch.testing.assert_close(args[0][..., :-4], current['context'], rtol=0, atol=0)
            if current['count'] == 0:
                self.assertEqual(args[0][..., -4:].count_nonzero(), 0)
            if batched:
                # One device-side matrix forward per phase, across every pair
                # and world; a scalar fallback must never execute.
                self.assertEqual(output.shape, (4, 4))
            self.assertEqual(output.device.type, device)
            outputs.append(output)
            current['count'] += 1
            events.append('thought')

        forward = network.forward_batch if batched else network.forward

        def checked_forward(observation, state, **kwargs):
            callbacks.append((observation, kwargs['communicating'], kwargs.get('active')))
            result = forward(observation, state, **kwargs)
            if batched:
                logits, values, updated = result
                inactive = ~kwargs['active']
                self.assertEqual(logits[inactive].count_nonzero(), 0)
                self.assertEqual(values[inactive].count_nonzero(), 0)
                for old, new in ((state.memory, updated.memory),
                                 (state.affect, updated.affect),
                                 (state.entities.keys, updated.entities.keys),
                                 (state.entities.values, updated.entities.values),
                                 (state.entities.occupied, updated.entities.occupied)):
                    torch.testing.assert_close(new[inactive], old[inactive], rtol=0, atol=0)
            return result

        def record(name):
            def hook(module, args, output):
                self.assertEqual(current['count'], steps)
                events.append(name)
            return hook

        original_step = world.step

        def resolve(*args):
            events.append('world')
            return original_step(*args)

        with ExitStack() as stack:
            stack.enter_context(patch.object(network, '_recur', side_effect=checked_recur))
            stack.enter_context(patch.object(network, 'forward_batch' if batched else 'forward',
                                             side_effect=checked_forward))
            if batched:
                stack.enter_context(patch.object(network, 'forward',
                                                 side_effect=AssertionError('scalar forward')))
            stack.enter_context(patch.object(world, 'step', side_effect=resolve))
            stack.callback(network.thought.register_forward_hook(thought_hook).remove)
            for name in ('memory_update', 'affect_update', 'entity_update', 'encounter_update'):
                stack.callback(getattr(network, name).register_forward_hook(record(name)).remove)
            rollout = collector.collect(10)

        callback = ['callback'] + ['thought'] * steps + [
            'memory_update', 'affect_update', 'entity_update']
        self.assertEqual(events, (callback * (4 if batched else 8) + ['world']
                                 + ['encounter_update'] * (1 if batched else 4)) * 3)
        self.assertEqual(len(outputs), steps * (12 if batched else 24))
        # The only observations admitted during thinking are local resources,
        # Appearance, role, past messages and the first mover's visible action.
        expected_fields = ({'resources', 'partner_appearance', 'first',
                            'received_message', 'partner_action'} if batched else
                           {'life', 'points', 'partner_life', 'partner_points',
                            'partner_appearance', 'first', 'received_message', 'partner_action'})
        for i, (observation, communicating, active) in enumerate(callbacks):
            phase = i % 4
            self.assertEqual({f.name for f in fields(observation)}, expected_fields)
            self.assertEqual(communicating, phase < 2)
            if phase < 3:
                if batched:
                    self.assertEqual(observation.partner_action.count_nonzero(), 0)
                else:
                    self.assertIsNone(observation.partner_action)
            elif batched:
                self.assertTrue(((observation.partner_action[active] >= 1)
                                 & (observation.partner_action[active] <= 2)).all())
            else:
                self.assertIsInstance(observation.partner_action, Action)

        components = survival_loss_components(rollout)
        # Holding sampled scores fixed, message content must not introduce a
        # direct communication reward or an auxiliary reconstruction objective.
        def change_message(decision):
            if batched:
                return (replace(decision, choice=(decision.choice + 1) % network.vocabulary_size)
                        if decision.communicating else decision)
            return (replace(decision, choice=tuple((token + 1) % network.vocabulary_size
                                                  for token in decision.choice))
                    if isinstance(decision.choice, tuple) else decision)

        def change_experience(item):
            return replace(item, decisions=tuple(change_message(d) for d in item.decisions))

        altered = replace(rollout, experiences=(
            tuple(change_experience(item) for item in rollout.experiences) if batched else
            tuple(tuple(change_experience(item) for item in items) for items in rollout.experiences)))
        torch.testing.assert_close(survival_loss_components(altered).loss, components.loss,
                                   rtol=0, atol=0)
        # Check actor and critic separately: a surviving value path must not
        # conceal a detached policy path (or vice versa), even at the first step.
        for loss in (components.actor_loss, components.value_loss):
            gradients = torch.autograd.grad(loss, outputs, retain_graph=True)
            for gradient in gradients:
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient.abs().sum(dim=-1) > 0).all())
        components.loss.backward()
        for name, parameter in network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        # Renewal and communication generate no reward: every initially living
        # agent earns precisely one survival unit per world step.
        rewards = ([item.reward for item in rollout.experiences] if batched else
                   [item.reward for items in rollout.experiences for item in items])
        for reward in rewards:
            torch.testing.assert_close(torch.as_tensor(reward), torch.ones_like(torch.as_tensor(reward)))
        truncated = (replace(rollout, horizon_completed=torch.zeros_like(rollout.horizon_completed))
                     if batched else replace(rollout, horizon_completed=False, truncated=True))
        with self.assertRaisesRegex(ValueError, 'complete episode'):
            survival_loss_components(truncated)

    def test_dead_agents_and_finished_worlds_do_not_think(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for batched in (False, True):
                with self.subTest(device=device, batched=batched):
                    collector = self.collector(batched, device=device)
                    network = collector.network
                    # Deterministic NOTHING leaves the scripted death schedule
                    # independent of sampled partners and renewable Points.
                    with torch.no_grad():
                        network.action_head.bias[1] = -1000
                    collector.world.state.life[..., 0] = 1
                    if batched:
                        collector.world.state.life[0] = 1
                    collector.collect(1)
                    if batched:
                        frozen = collector.state
                        self.assertTrue(collector.world.done[0])
                        forward = network.forward_batch
                        masks = []

                        def checked(observation, state, **kwargs):
                            active = kwargs['active']
                            masks.append(active.clone())
                            self.assertFalse(active[0].any())
                            self.assertFalse(active[:, 0].any())
                            return forward(observation, state, **kwargs)

                        with patch.object(network, 'forward_batch', side_effect=checked), \
                                patch.object(network, 'forward', side_effect=AssertionError('scalar forward')):
                            collector.collect(10)
                        self.assertEqual(len(masks), 8)  # Four phases, two live-world steps.
                        for before, after in ((frozen.memory, collector.state.memory),
                                              (frozen.affect, collector.state.affect),
                                              (frozen.entities.keys, collector.state.entities.keys),
                                              (frozen.entities.values, collector.state.entities.values),
                                              (frozen.entities.occupied, collector.state.entities.occupied)):
                            torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
                            torch.testing.assert_close(after[1, 0], before[1, 0], rtol=0, atol=0)
                        self.assertEqual(collector.world.steps.tolist(), [1, 3])
                    else:
                        dead = collector.agents[0]
                        self.assertEqual(dead.state.memory.count_nonzero(), 0)
                        self.assertEqual(dead.state.affect.count_nonzero(), 0)
                        self.assertEqual(dead.state.entities, ())
                        with patch.object(dead, 'communicate', side_effect=AssertionError('dead message')), \
                                patch.object(dead, 'act', side_effect=AssertionError('dead action')):
                            collector.collect(10)
                    # Completed collectors neither think nor advance the world.
                    with patch.object(network, '_recur', side_effect=AssertionError('completed thought')), \
                            patch.object(collector.world, 'step', side_effect=AssertionError('completed step')):
                        collector.collect(1)

    def test_private_state_and_episode_reset(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                collector = self.collector(batched)
                network = collector.network
                rollout = collector.collect(3)
                if batched:
                    state = collector.state
                    observation = rollout.experiences[-1].decisions[-1].observation
                    active = torch.ones_like(observation.first)
                    changed = replace(state, memory=state.memory.clone(), affect=state.affect.clone(),
                                      entities=replace(state.entities, values=state.entities.values.clone()))
                    changed.memory[0, 0] += 7
                    changed.affect[0, 0] += 7
                    changed.entities.values[0, 0] += 7
                    reference = network.forward_batch(observation, state, communicating=False, active=active)
                    perturbed = network.forward_batch(observation, changed, communicating=False, active=active)
                    untouched = active.clone()
                    untouched[0, 0] = False
                    for before, after in zip(reference[:2], perturbed[:2]):
                        torch.testing.assert_close(before[untouched], after[untouched], rtol=0, atol=0)
                        self.assertFalse(torch.equal(before[0, 0], after[0, 0]))
                    for before, after in ((reference[2].memory, perturbed[2].memory),
                                          (reference[2].affect, perturbed[2].affect),
                                          (reference[2].entities.keys, perturbed[2].entities.keys),
                                          (reference[2].entities.values, perturbed[2].entities.values),
                                          (reference[2].entities.occupied, perturbed[2].entities.occupied)):
                        torch.testing.assert_close(before[untouched], after[untouched], rtol=0, atol=0)
                    collector.reset(0, seed=17)
                    for before, after in ((state.memory, collector.state.memory),
                                          (state.affect, collector.state.affect),
                                          (state.entities.keys, collector.state.entities.keys),
                                          (state.entities.values, collector.state.entities.values),
                                          (state.entities.occupied, collector.state.entities.occupied)):
                        self.assertEqual(after[0].count_nonzero(), 0)
                        torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)
                    # Resetting one episode must retain the other world's graph.
                    gradient, = torch.autograd.grad(collector.state.memory[1].sum(), state.memory)
                    self.assertEqual(gradient[0].count_nonzero(), 0)
                    torch.testing.assert_close(gradient[1], torch.ones_like(gradient[1]))
                else:
                    states = [agent.state for agent in collector.agents]
                    observation = next(d.observation for items in rollout.experiences
                                       for item in items for d in item.decisions)
                    reference, _ = network(observation, states[1], communicating=False)
                    original, _ = network(observation, states[0], communicating=False)
                    collector.agents[0].state = replace(states[0], memory=states[0].memory + 7)
                    perturbed, _ = network(observation, collector.agents[0].state, communicating=False)
                    again, _ = network(observation, collector.agents[1].state, communicating=False)
                    self.assertFalse(torch.equal(original, perturbed))
                    torch.testing.assert_close(reference, again, rtol=0, atol=0)
                    collector.reset()
                    for agent in collector.agents:
                        self.assertEqual(agent.state.memory.count_nonzero(), 0)
                        self.assertEqual(agent.state.affect.count_nonzero(), 0)
                        self.assertEqual(agent.state.entities, ())
                        self.assertIsNone(agent.state.memory.grad_fn)
                        self.assertEqual(agent.log_probs, [])


if __name__ == '__main__':
    unittest.main()
