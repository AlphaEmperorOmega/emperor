import torch
from torch import Tensor

from emperor.neuron._cluster.state import NeuronClusterRouteState
from emperor.neuron._trace import NeuronClusterTrace, NeuronClusterTraceStep


class _NeuronClusterRecurrentRoutesMixin:
    def _propagate_signal_through_recurrent_routes(
        self,
        input: Tensor,
        input_shape: tuple[int, ...],
        return_trace: bool,
    ) -> tuple[Tensor, Tensor, NeuronClusterTrace | None]:
        if self.beam_width > 1:
            return self._propagate_signal_through_beam_routes(input)

        route_state = self.__run_entry_routes(input, input_shape, return_trace)

        for _ in range(self.max_steps):
            route_mask = self._current_route_mask(route_state)
            if not bool(route_mask.any().item()):
                break
            route_state = self.__run_recurrent_route_step(route_state, route_mask)

        route_state = self._maybe_finalize_cluster_halting(route_state)
        return route_state.hidden, route_state.loss, route_state.trace

    def __run_entry_routes(
        self,
        input: Tensor,
        input_shape: tuple[int, ...],
        return_trace: bool,
    ) -> NeuronClusterRouteState:
        probabilities, selected_coords, entry_loss = self._route_entry_input(input)
        entry_called_mask = torch.ones(
            input.shape[0],
            dtype=torch.bool,
            device=input.device,
        )
        branch_outputs, valid_mask, escape_mask = self._run_process_branches(
            input,
            selected_coords,
            entry_called_mask,
        )
        weighted_candidate = self._weighted_branch_candidate(
            branch_outputs,
            probabilities,
            valid_mask,
        )
        halting_state = self._maybe_update_halting_state(
            None,
            input,
            weighted_candidate,
            entry_called_mask,
        )

        chosen_branch_indices = probabilities.argmax(dim=1)
        chosen_valid_mask = self._gather_branch_mask(
            valid_mask,
            chosen_branch_indices,
        )
        batch_indices = torch.arange(input.shape[0], device=input.device)

        hidden = input.clone()
        positions = torch.zeros(
            input.shape[0],
            3,
            dtype=torch.long,
            device=input.device,
        )
        active_mask = chosen_valid_mask.clone()
        escaped_mask = ~chosen_valid_mask
        final_mask = escaped_mask.clone()

        if bool(chosen_valid_mask.any().item()):
            valid_batch_indices = batch_indices[chosen_valid_mask]
            valid_branch_indices = chosen_branch_indices[chosen_valid_mask]
            hidden[valid_batch_indices] = weighted_candidate[valid_batch_indices]
            positions[valid_batch_indices] = selected_coords[
                valid_batch_indices,
                valid_branch_indices,
            ]

        if bool(escaped_mask.any().item()):
            hidden[escaped_mask] = weighted_candidate[escaped_mask]

        trace = None
        if return_trace:
            trace = NeuronClusterTrace(
                input_shape=input_shape,
                entry_coordinates=self._detach_trace_tensor(
                    self.entry_coordinates.to(device=input.device)
                ),
                entry_probabilities=self._detach_trace_tensor(probabilities),
                entry_selected_coordinates=self._detach_trace_tensor(selected_coords),
                entry_valid_mask=self._detach_trace_tensor(valid_mask),
                entry_escape_mask=self._detach_trace_tensor(escape_mask),
                entry_chosen_branch_indices=self._detach_trace_tensor(
                    chosen_branch_indices
                ),
                entry_halt_mask=self._detach_trace_tensor(
                    self._halt_mask_tensor(
                        halting_state,
                        input.shape[0],
                        input.device,
                    )
                ),
                entry_active_mask=self._detach_trace_tensor(active_mask),
            )

        return NeuronClusterRouteState(
            hidden=hidden,
            positions=positions,
            active_mask=active_mask,
            escaped_mask=escaped_mask,
            final_mask=final_mask,
            halting_state=halting_state,
            loss=input.new_zeros(()) + entry_loss,
            trace=trace,
        )

    def _route_entry_input(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probabilities, indices, _, auxiliary_loss = (
            self.entry_sampler.sample_probabilities_and_indices(input)
        )
        probabilities = self._ensure_probability_matrix(probabilities)
        indices = self._resolve_selected_indices(
            input,
            indices,
            int(self.entry_coordinates.shape[0]),
        )
        indices = self._ensure_index_matrix(indices)
        entry_coordinates = self.entry_coordinates.to(indices.device)[indices]
        return probabilities, entry_coordinates, auxiliary_loss

    def _current_route_mask(self, route_state: NeuronClusterRouteState) -> Tensor:
        halt_mask = self._get_halt_mask(route_state.halting_state)
        active_mask = route_state.active_mask & ~route_state.final_mask
        if halt_mask is None:
            return active_mask
        return active_mask & ~halt_mask

    def __run_recurrent_route_step(
        self,
        route_state: NeuronClusterRouteState,
        route_mask: Tensor,
    ) -> NeuronClusterRouteState:
        next_hidden = route_state.hidden.clone()
        next_positions = route_state.positions.clone()
        next_active_mask = route_state.active_mask.clone()
        next_escaped_mask = route_state.escaped_mask.clone()
        next_final_mask = route_state.final_mask.clone()
        current_called_mask = torch.zeros_like(route_mask)
        probabilities = None
        selected_coords = None
        loss = route_state.loss

        for neuron_name, batch_indices in self._group_indices_by_position(
            route_state.positions,
            route_mask,
        ).items():
            index_tensor = self._index_tensor(
                batch_indices,
                route_state.hidden.device,
            )
            if neuron_name not in self.cluster:
                next_active_mask[index_tensor] = False
                next_final_mask[index_tensor] = True
                continue

            route_probabilities, route_coords, neuron_loss = self._route_neuron(
                self.cluster[neuron_name],
                route_state.hidden.index_select(0, index_tensor),
            )
            probabilities, selected_coords = self._ensure_route_buffers(
                probabilities,
                selected_coords,
                route_probabilities,
                route_coords,
                route_state.hidden,
            )
            probabilities[index_tensor] = route_probabilities
            selected_coords[index_tensor] = route_coords.to(
                device=route_state.hidden.device,
                dtype=torch.long,
            )
            current_called_mask[index_tensor] = True
            loss = self._accumulate_auxiliary_loss(loss, neuron_loss)

        missing_current_mask = route_mask & ~current_called_mask
        if bool(missing_current_mask.any().item()):
            next_active_mask[missing_current_mask] = False
            next_final_mask[missing_current_mask] = True

        if probabilities is None or selected_coords is None:
            return NeuronClusterRouteState(
                hidden=next_hidden,
                positions=next_positions,
                active_mask=next_active_mask,
                escaped_mask=next_escaped_mask,
                final_mask=next_final_mask,
                halting_state=route_state.halting_state,
                loss=loss,
                trace=route_state.trace,
            )

        branch_outputs, valid_target_mask, escape_mask = self._run_process_branches(
            route_state.hidden,
            selected_coords,
            current_called_mask,
        )
        weighted_candidate = self._weighted_branch_candidate(
            branch_outputs,
            probabilities,
            valid_target_mask,
        )
        chosen_branch_indices = torch.zeros(
            route_state.hidden.shape[0],
            dtype=torch.long,
            device=route_state.hidden.device,
        )
        if bool(current_called_mask.any().item()):
            chosen_branch_indices[current_called_mask] = probabilities[
                current_called_mask
            ].argmax(dim=1)
        chosen_valid_mask = self._gather_branch_mask(
            valid_target_mask,
            chosen_branch_indices,
        )
        continuing_mask = current_called_mask & chosen_valid_mask
        escaped_final_mask = current_called_mask & ~chosen_valid_mask

        batch_indices = torch.arange(
            route_state.hidden.shape[0],
            device=route_state.hidden.device,
        )
        if bool(continuing_mask.any().item()):
            continuing_batch_indices = batch_indices[continuing_mask]
            continuing_branch_indices = chosen_branch_indices[continuing_mask]
            next_hidden[continuing_batch_indices] = weighted_candidate[
                continuing_batch_indices
            ]
            next_positions[continuing_batch_indices] = selected_coords[
                continuing_batch_indices,
                continuing_branch_indices,
            ]

        if bool(escaped_final_mask.any().item()):
            next_hidden[escaped_final_mask] = weighted_candidate[escaped_final_mask]
            next_escaped_mask[escaped_final_mask] = True
            next_final_mask[escaped_final_mask] = True

        next_active_mask[current_called_mask] = continuing_mask[current_called_mask]

        halting_state = self._maybe_update_halting_state(
            route_state.halting_state,
            route_state.hidden,
            weighted_candidate,
            current_called_mask,
        )

        if route_state.trace is not None:
            route_state.trace.steps.append(
                NeuronClusterTraceStep(
                    probabilities=self._detach_trace_tensor(probabilities),
                    selected_coordinates=self._detach_trace_tensor(selected_coords),
                    valid_mask=self._detach_trace_tensor(valid_target_mask),
                    escape_mask=self._detach_trace_tensor(escape_mask),
                    chosen_branch_indices=self._detach_trace_tensor(
                        chosen_branch_indices
                    ),
                    halt_mask=self._detach_trace_tensor(
                        self._halt_mask_tensor(
                            halting_state,
                            route_state.hidden.shape[0],
                            route_state.hidden.device,
                        )
                    ),
                    active_mask=self._detach_trace_tensor(next_active_mask),
                )
            )

        return NeuronClusterRouteState(
            hidden=next_hidden,
            positions=next_positions,
            active_mask=next_active_mask,
            escaped_mask=next_escaped_mask,
            final_mask=next_final_mask,
            halting_state=halting_state,
            loss=loss,
            trace=route_state.trace,
        )

    def _route_neuron(
        self,
        neuron,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if hasattr(neuron, "route_signal"):
            return neuron.route_signal(hidden)

        _, probabilities, selected_coords, auxiliary_loss = neuron(hidden)
        return probabilities, selected_coords, auxiliary_loss

    def _process_neuron(self, neuron, hidden: Tensor) -> Tensor:
        if hasattr(neuron, "process_signal"):
            output = neuron.process_signal(hidden)
        else:
            output, _, _, _ = neuron(hidden)
        return self.__blend_warming_up_neuron_output(neuron, hidden, output)

    def __blend_warming_up_neuron_output(
        self,
        neuron,
        input: Tensor,
        output: Tensor,
    ) -> Tensor:
        if self.growth_warmup_steps is None:
            return output
        warmup_remaining = getattr(neuron, "warmup_remaining_steps", None)
        if warmup_remaining is None:
            return output
        remaining_steps = int(warmup_remaining.item())
        if remaining_steps <= 0:
            return output
        total_steps = self.growth_warmup_steps
        output_weight = (total_steps - remaining_steps + 1) / total_steps
        return output_weight * output + (1.0 - output_weight) * input

    def _run_process_branches(
        self,
        source_hidden: Tensor,
        selected_coords: Tensor,
        call_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        branch_outputs = (
            source_hidden.unsqueeze(1).expand(-1, selected_coords.shape[1], -1).clone()
        )
        target_groups, valid_target_mask, escape_mask = (
            self.__group_valid_process_calls(selected_coords, call_mask)
        )

        for neuron_name, branch_indices in target_groups.items():
            batch_indices = [batch_index for batch_index, _ in branch_indices]
            topk_indices = [topk_index for _, topk_index in branch_indices]
            batch_index_tensor = self._index_tensor(
                batch_indices,
                source_hidden.device,
            )
            topk_index_tensor = self._index_tensor(
                topk_indices,
                source_hidden.device,
            )
            output = self._process_neuron(
                self.cluster[neuron_name],
                source_hidden.index_select(0, batch_index_tensor),
            )
            branch_outputs[batch_index_tensor, topk_index_tensor] = output
            if self.training:
                self._neurons_called_this_forward.add(neuron_name)

        return branch_outputs, valid_target_mask, escape_mask

    def __group_valid_process_calls(
        self,
        selected_coords: Tensor,
        call_mask: Tensor,
    ) -> tuple[dict[str, list[tuple[int, int]]], Tensor, Tensor]:
        target_groups: dict[str, list[tuple[int, int]]] = {}
        valid_target_mask = torch.zeros(
            selected_coords.shape[:2],
            dtype=torch.bool,
            device=selected_coords.device,
        )
        escape_mask = torch.zeros_like(valid_target_mask)
        escaped_missing_positions: list[tuple[int, int, int]] = []
        coordinate_rows = selected_coords.detach().cpu().tolist()
        for batch_index in self._mask_indices(call_mask):
            for topk_index in range(selected_coords.shape[1]):
                coordinate = self._coordinate_from_row(
                    coordinate_rows[batch_index][topk_index]
                )
                if not self._is_valid_coordinate(coordinate):
                    escape_mask[batch_index, topk_index] = True
                    if self._is_within_grid_capacity(coordinate):
                        escaped_missing_positions.append(coordinate)
                    continue
                neuron_name = self._neuron_name(*coordinate)
                target_groups.setdefault(neuron_name, []).append(
                    (batch_index, topk_index)
                )
                valid_target_mask[batch_index, topk_index] = True
        self._record_escaped_missing_positions(escaped_missing_positions)
        return target_groups, valid_target_mask, escape_mask
