import torch
import matplotlib.pyplot as plt

from Emperor.config import ParameterGeneratorConfig


class Frequency:
    def __init__(self, cfg: ParameterGeneratorConfig):
        self.d_input = cfg.inputDim
        self.d_depth = cfg.outputDim
        self.batch_size = cfg.batchSize

    # Normalize tensor values to the range [0, 1]
    def normalize_tensor(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def resetFrequency(self):
        self.frequency.zero_()

    def plot(self, min_width=2):
        frequency_data = self.normalize_tensor(self.frequency_data)

        # Check if the tensor is 1D or 2D and convert if necessary
        if frequency_data.dim() == 1:
            frequency_data = frequency_data.unsqueeze(0)  # Convert 1D tensor to 2D

        # Convert the tensor to a NumPy array
        numpy_array = frequency_data.numpy()

        # Create the plot with the specified minimum width
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot differently based on the tensor dimensions
        if numpy_array.shape[0] == 1:
            ax.bar(range(numpy_array.shape[1]), numpy_array[0])
            ax.set_title("Expert frequency")
            ax.set_xlabel("Depth")
            ax.set_ylabel("Expert usage in percentage")
        else:
            cax = ax.imshow(numpy_array, cmap="viridis", aspect="auto")
            fig.colorbar(cax, ax=ax)
            ax.set_title("2D Tensor as Image")
            ax.set_xlabel("Depth")
            ax.set_ylabel("Input size")

        plt.show()


class VectorChoiceSparseFrequency(Frequency):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.frequency_data = torch.zeros(
            (self.d_input, self.d_depth), dtype=torch.float
        )

    def update(self, idxs):
        """
        Inputs:
            - idxs: indices where the column repersents the number of vectors that need to be chosen
            and the batch_size. Shape: [d_input, batch]
        """
        # Tenor used to store the location of which vector is chosen for each tensor
        frequency = torch.full(
            (self.d_input, self.batch_size, self.d_depth), -1, dtype=torch.float
        )
        idxs = idxs.unsqueeze(2)

        # Store the indices at their assigned location
        frequency = frequency.scatter(2, idxs, idxs.float())
        frequency = (frequency > -1).long()
        frequency = frequency.permute(1, 0, 2).sum(dim=0)
        self.frequency_data += frequency


class MatrixChoiceFrequency(Frequency):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.row_indices = torch.arange(self.batch_size)
        self.frequency_data = torch.zeros((self.d_depth), dtype=torch.long)

    def update(self, idxs):
        """
        Inputs:
            - idxs: indices where the column repersents the number of vectors that need to be chosen
            and the batch_size. Shape: [d_input, batch]
        """
        zeros = torch.zeros(self.batch_size, self.d_depth)
        zeros[self.row_indices, idxs.reshape(-1)] = 1
        zeros = zeros.sum(dim=0)
        self.frequency_data += zeros.long()


class VectorChoiceMixtureFrequency(Frequency):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.frequency_data = torch.zeros(
            (self.d_input, self.d_depth), dtype=torch.float
        )

    def update(self, idxs):
        """
        Inputs:
            idxs: indices of shape [d_input, batch_size, topk]
        """
        # Tenor used to store the location of which vector is chosen for each tensor
        frequency = torch.full(
            (self.d_input, self.batch_size, self.d_depth), -1, dtype=torch.float
        )

        # Store the indices at their assigned location
        frequency = frequency.scatter(2, idxs, idxs.float())
        frequency = (frequency > -1).long()
        frequency = frequency.permute(1, 0, 2).sum(dim=0)
        self.frequency_data += frequency


class MatrixChoiceMixtureFrequency(Frequency):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.frequency_data = torch.zeros((self.d_depth), dtype=torch.long)

    def update(self, idxs):
        """
        Inputs:
            - idxs: indices of shape [batch_size, topk]
        """
        frequency = torch.full((self.batch_size, self.d_depth), -1, dtype=torch.float)
        frequency = frequency.scatter(1, idxs, idxs.float())
        self.frequency_data = (frequency > -1).long().sum(dim=0)


class MatrixGeneratorMixtureFrequency(Frequency):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.frequency_data = torch.zeros((self.d_depth))

    def update(self, idxs):
        """
        Inputs:
            - idxs: indices of shape [batch_size, topk]
        """
        frequency = torch.full((self.batch_size, self.d_depth), -1, dtype=torch.float)
        frequency = frequency.scatter(1, idxs, idxs.float())
        self.frequency_data = (frequency > -1).long().sum(dim=0)
