import torch


class _NeuronClusterCheckpointingMixin:
    def _reconcile_cluster_with_state_dict(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        cluster_prefix = f"{prefix}cluster."
        incoming_neuron_names = self.__incoming_neuron_names(
            state_dict,
            cluster_prefix,
        )
        if not incoming_neuron_names:
            return
        if not self.__validate_incoming_topology(
            incoming_neuron_names,
            error_msgs,
        ):
            return
        self.__reconcile_neurons(incoming_neuron_names)
        self.__seed_missing_atrophy_counters(state_dict, cluster_prefix)
        self.__seed_missing_growth_budget_buffers(state_dict, prefix)
        self.__register_warmup_buffers_for_incoming_keys(state_dict, cluster_prefix)
        self.__seed_missing_warmup_buffers(state_dict, cluster_prefix)

    def __incoming_neuron_names(
        self,
        state_dict,
        cluster_prefix: str,
    ) -> tuple[str, ...]:
        incoming_neuron_names: dict[str, None] = {}
        for key in state_dict:
            if not key.startswith(cluster_prefix):
                continue
            neuron_name = key[len(cluster_prefix) :].split(".", 1)[0]
            if self._is_neuron_name(neuron_name):
                incoming_neuron_names.setdefault(neuron_name, None)
        return tuple(incoming_neuron_names)

    def __validate_incoming_topology(
        self,
        incoming_neuron_names: tuple[str, ...],
        error_msgs: list[str],
    ) -> bool:
        noncanonical_names = sorted(
            neuron_name
            for neuron_name in incoming_neuron_names
            if neuron_name != self._neuron_name(*self._parse_neuron_name(neuron_name))
        )
        if noncanonical_names:
            error_msgs.append(
                "NeuronCluster checkpoint topology contains non-canonical neuron "
                f"names: {noncanonical_names}."
            )
            return False

        invalid_names = sorted(
            neuron_name
            for neuron_name in incoming_neuron_names
            if not self._is_within_grid_capacity(self._parse_neuron_name(neuron_name))
        )
        if invalid_names:
            error_msgs.append(
                "NeuronCluster checkpoint topology contains neurons outside the "
                f"configured cluster capacity: {invalid_names}."
            )
            return False

        entry_neuron_names = {
            self._neuron_name(*self._coordinate_from_row(coordinate_row))
            for coordinate_row in self.entry_coordinates.detach().cpu().tolist()
        }
        missing_entry_neuron_names = sorted(
            entry_neuron_names - set(incoming_neuron_names)
        )
        if missing_entry_neuron_names:
            error_msgs.append(
                "NeuronCluster checkpoint topology is missing configured "
                f"entry-plane neurons: {missing_entry_neuron_names}."
            )
            return False
        return True

    def __reconcile_neurons(self, incoming_neuron_names: tuple[str, ...]) -> None:
        missing_neuron_names = tuple(
            neuron_name
            for neuron_name in incoming_neuron_names
            if neuron_name not in self.cluster
        )
        reconstructed_neurons = {
            neuron_name: self._initialize_neuron(
                *self._parse_neuron_name(neuron_name)
            )
            for neuron_name in missing_neuron_names
        }
        checkpoint_ordered_neurons = [
            (
                neuron_name,
                self.cluster[neuron_name]
                if neuron_name in self.cluster
                else reconstructed_neurons[neuron_name],
            )
            for neuron_name in incoming_neuron_names
        ]
        self.cluster.clear()
        for neuron_name, neuron in checkpoint_ordered_neurons:
            self._add_neuron(self.cluster, neuron_name, neuron)

    def __seed_missing_atrophy_counters(
        self,
        state_dict,
        cluster_prefix: str,
    ) -> None:
        """Zero-fill missing atrophy counters from legacy checkpoints."""
        for neuron_name in self.cluster.keys():
            counter_key = f"{cluster_prefix}{neuron_name}.atrophy_counter"
            if counter_key not in state_dict:
                state_dict[counter_key] = torch.zeros((), dtype=torch.int64)

    def __seed_missing_growth_budget_buffers(
        self,
        state_dict,
        prefix: str,
    ) -> None:
        for buffer_name in ("forwards_since_last_growth", "total_growth_count"):
            if getattr(self, buffer_name) is None:
                continue
            buffer_key = f"{prefix}{buffer_name}"
            if buffer_key not in state_dict:
                state_dict[buffer_key] = torch.zeros((), dtype=torch.int64)

    def __register_warmup_buffers_for_incoming_keys(
        self,
        state_dict,
        cluster_prefix: str,
    ) -> None:
        buffer_suffix = ".warmup_remaining_steps"
        for key in list(state_dict.keys()):
            if not key.startswith(cluster_prefix) or not key.endswith(buffer_suffix):
                continue
            neuron_name = key[len(cluster_prefix) :].split(".", 1)[0]
            if neuron_name not in self.cluster:
                continue
            neuron = self.cluster[neuron_name]
            if getattr(neuron, "warmup_remaining_steps", None) is None:
                neuron.register_buffer(
                    "warmup_remaining_steps",
                    torch.zeros(
                        (),
                        dtype=torch.int64,
                        device=neuron.batch_counter.device,
                    ),
                    persistent=True,
                )

    def __seed_missing_warmup_buffers(
        self,
        state_dict,
        cluster_prefix: str,
    ) -> None:
        for neuron_name, neuron in self.cluster.items():
            if getattr(neuron, "warmup_remaining_steps", None) is None:
                continue
            buffer_key = f"{cluster_prefix}{neuron_name}.warmup_remaining_steps"
            if buffer_key not in state_dict:
                state_dict[buffer_key] = torch.zeros((), dtype=torch.int64)
