from collections.abc import Sequence

import torch
from torch import Tensor

from emperor.datasets.text._bert_vocabulary import BertSpecialTokenIds


class MaskedLanguageModelingCollator:
    def __init__(
        self,
        special_token_ids: BertSpecialTokenIds,
        vocab_size: int,
        mlm_probability: float = 0.15,
        mask_replace_probability: float = 0.8,
        random_replace_probability: float = 0.1,
        ignore_index: int = -100,
        generator: torch.Generator | None = None,
    ) -> None:
        self._validate_probabilities(
            mlm_probability,
            mask_replace_probability,
            random_replace_probability,
        )
        self.special_token_ids = special_token_ids
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.mask_replace_probability = mask_replace_probability
        self.random_replace_probability = random_replace_probability
        self.ignore_index = ignore_index
        self.generator = generator

        special_ids = set(special_token_ids.values())
        max_special_id = max(special_ids)
        if vocab_size <= max_special_id:
            raise ValueError(
                "vocab_size must be greater than every BERT special token id."
            )
        self.random_token_ids = torch.tensor(
            [index for index in range(vocab_size) if index not in special_ids],
            dtype=torch.long,
        )

    def __call__(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        token_ids = self._stack_batch(batch)
        input_ids = token_ids.clone()
        labels = token_ids.clone()
        attention_mask = (token_ids != self.special_token_ids.pad).long()

        masked_indices = self._sample_masked_indices(token_ids)
        labels[~masked_indices] = self.ignore_index

        replacement_sample = torch.rand(
            input_ids.shape,
            generator=self.generator,
            device=input_ids.device,
        )
        mask_replacement = masked_indices & (
            replacement_sample < self.mask_replace_probability
        )
        random_replacement = masked_indices & (
            replacement_sample >= self.mask_replace_probability
        )
        random_replacement &= (
            replacement_sample
            < self.mask_replace_probability + self.random_replace_probability
        )

        input_ids[mask_replacement] = self.special_token_ids.mask
        if self.random_token_ids.numel() > 0:
            random_token_ids = self.random_token_ids.to(input_ids.device)
            random_indices = torch.randint(
                random_token_ids.numel(),
                input_ids.shape,
                generator=self.generator,
                device=input_ids.device,
            )
            random_tokens = random_token_ids[random_indices]
            input_ids[random_replacement] = random_tokens[random_replacement]

        return input_ids, labels, attention_mask

    def _stack_batch(self, batch) -> Tensor:
        if isinstance(batch, Tensor):
            token_ids = batch
        else:
            if len(batch) == 0:
                raise ValueError("MaskedLanguageModelingCollator received no samples.")
            samples = []
            for sample in batch:
                if isinstance(sample, Tensor):
                    samples.append(sample)
                elif (
                    isinstance(sample, Sequence)
                    and len(sample) == 1
                    and isinstance(sample[0], Tensor)
                ):
                    samples.append(sample[0])
                else:
                    raise TypeError(
                        "MaskedLanguageModelingCollator expects tensors or "
                        "single-tensor samples."
                    )
            token_ids = torch.stack(samples)

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2:
            raise ValueError(
                "MaskedLanguageModelingCollator expects a 2D token tensor."
            )
        return token_ids.long()

    def _sample_masked_indices(self, token_ids: Tensor) -> Tensor:
        probability_matrix = torch.full(
            token_ids.shape,
            self.mlm_probability,
            device=token_ids.device,
        )
        masked_indices = torch.bernoulli(
            probability_matrix,
            generator=self.generator,
        ).bool()
        masked_indices &= self._eligible_mask(token_ids)

        if not masked_indices.any():
            candidate_positions = self._eligible_mask(token_ids).nonzero(as_tuple=False)
            if candidate_positions.numel() > 0:
                selected_position = torch.randint(
                    candidate_positions.size(0),
                    (1,),
                    generator=self.generator,
                    device=token_ids.device,
                )
                row, column = candidate_positions[selected_position.item()]
                masked_indices[row, column] = True

        return masked_indices

    def _eligible_mask(self, token_ids: Tensor) -> Tensor:
        mask = torch.ones_like(token_ids, dtype=torch.bool)
        for special_token_id in self.special_token_ids.values():
            mask &= token_ids != special_token_id
        return mask

    def _validate_probabilities(
        self,
        mlm_probability: float,
        mask_replace_probability: float,
        random_replace_probability: float,
    ) -> None:
        probabilities = {
            "mlm_probability": mlm_probability,
            "mask_replace_probability": mask_replace_probability,
            "random_replace_probability": random_replace_probability,
        }
        for name, probability in probabilities.items():
            if probability < 0.0 or probability > 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0.")
        if mask_replace_probability + random_replace_probability > 1.0:
            raise ValueError(
                "mask_replace_probability + random_replace_probability must be <= 1.0."
            )
