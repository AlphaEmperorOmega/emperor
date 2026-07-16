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
        self.__rebuild_grown_neurons(incoming_neuron_names)
        self.__remove_pruned_neurons(incoming_neuron_names)
        self.__seed_missing_atrophy_counters(state_dict, cluster_prefix)
        self.__seed_missing_growth_budget_buffers(state_dict, prefix)
        self.__register_warmup_buffers_for_incoming_keys(state_dict, cluster_prefix)
        self.__seed_missing_warmup_buffers(state_dict, cluster_prefix)

    def __incoming_neuron_names(
        self,
        state_dict,
        cluster_prefix: str,
    ) -> set[str]:
        incoming_neuron_names = set()
        for key in state_dict:
            if not key.startswith(cluster_prefix):
                continue
            neuron_name = key[len(cluster_prefix) :].split(".", 1)[0]
            if self._is_neuron_name(neuron_name):
                incoming_neuron_names.add(neuron_name)
        return incoming_neuron_names

    def __rebuild_grown_neurons(self, incoming_neuron_names: set[str]) -> None:
        for neuron_name in sorted(incoming_neuron_names):
            if neuron_name in self.cluster:
                continue
            self._add_neuron(
                self.cluster,
                neuron_name,
                self._initialize_neuron(*self._parse_neuron_name(neuron_name)),
            )

    def __remove_pruned_neurons(self, incoming_neuron_names: set[str]) -> None:
        for neuron_name in list(self.cluster.keys()):
            if neuron_name not in incoming_neuron_names:
                del self.cluster[neuron_name]

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
