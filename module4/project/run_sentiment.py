from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import random

import embeddings

import minitorch
from datasets import load_dataset

BACKEND = minitorch.TensorBackend(minitorch.FastOps)


def RParam(*shape: int) -> minitorch.Parameter:
    """Create a random parameter tensor with given shape for network initialization.

    Args:
        shape (tuple): shape of the tensor.

    Returns:
        minitorch.Parameter: initialized parameter.

    """
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    """Linear module for creating a linear transformation with weights and biases."""

    def __init__(self, in_size: int, out_size: int) -> None:
        """Initialize weights and biases.

        Args:
            in_size (int): size of each input sample.
            out_size (int): size of each output sample.

        """
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Forward pass for the linear transformation.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor after applying linear transformation.

        """
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(minitorch.Module):
    """1D Convolution module for creating a convolutional layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_width: int) -> None:
        """Initialize weights and biases for 1D convolution.

        Args:
            in_channels (int): number of channels in the input signal.
            out_channels (int): number of channels produced by the convolution.
            kernel_width (int): size of the convolving kernel.

        """
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input: minitorch.Tensor) -> minitorch.Tensor:
        """Forward pass for 1D convolution.

        Args:
            input (Tensor): input tensor.

        Returns:
            Tensor: output tensor after applying 1D convolution.

        """
        # TODO: Implement for Task 4.5.
        out = minitorch.conv1d(input, self.weights.value) + self.bias.value
        return out


class CNNSentimentKim(minitorch.Module):
    """Implement a CNN for Sentiment classification based on Y. Kim 2014.

    This model should implement the following procedure:

    1. Apply a 1d convolution with input_channels=embedding_dim
        feature_map_size=100 output channels and [3, 4, 5]-sized kernels
        followed by a non-linear activation function (the paper uses tanh, we apply a ReLu)
    2. Apply max-over-time across each feature map
    3. Apply a Linear to size C (number of classes) followed by a ReLU and Dropout with rate 25%
    4. Apply a sigmoid over the class dimension.
    """

    def __init__(
        self,
        feature_map_size: int = 100,
        embedding_size: int = 50,
        filter_sizes: list[int] = [3, 4, 5],
        dropout: float = 0.25
    ):
        """Initializes the CNN with specified sizes and dropout rate.

        Args:
            feature_map_size (int): number of output channels in each convolution.
            embedding_size (int): dimensionality of input embeddings.
            filter_sizes (list of int): sizes of the convolutional filters.
            dropout (float): dropout rate for regularization.

        """
        super().__init__()
        self.feature_map_size = feature_map_size
        # TODO: Implement for Task 4.5.
        self.conv1 = Conv1d(embedding_size, feature_map_size, filter_sizes[0])
        self.conv2 = Conv1d(embedding_size, feature_map_size, filter_sizes[1])
        self.conv3 = Conv1d(embedding_size, feature_map_size, filter_sizes[2])
        self.linear = Linear(feature_map_size, 1)
        self.dropout = dropout

    def forward(self, embeddings: list[list[float]]) -> list[float]:
        """Embeddings tensor: [batch x sentence length x embedding dim]"""
        # TODO: Implement for Task 4.5.
        # permute embedding dim to input channels dim for conv layer
        x = embeddings.permute(0, 2, 1)
        x1 = self.conv1(x).relu()
        x2 = self.conv2(x).relu()
        x3 = self.conv3(x).relu()
        # Max over each feature map
        x = minitorch.max(x1, 2) + minitorch.max(x2, 2) + minitorch.max(x3, 2)
        x = self.linear(x.view(x.shape[0], self.feature_map_size))
        x = minitorch.dropout(x, self.dropout, self.mode == "eval") # when in evaluation mode, dropout is typically not applied
        # Apply sigmoid and view as batch size
        return x.sigmoid().view(x.shape[0])


# Evaluation helper methods
def get_predictions_array(y_true: list[int], model_output: list[int]) -> list[tuple[int, int, float]]:
    """Generate an array of predictions comparing the model output with true labels.

    Args:
        y_true (Tensor): true labels.
        model_output (Tensor): output from the model.

    Returns:
        list: a list of tuples containing the true label, predicted label, and model logit for each sample.

    """
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array: list[tuple[int, int, float]]) -> float:
    """Calculate the accuracy based on predictions.

    Args:
        predictions_array (list): list of tuples containing true and predicted labels.

    Returns:
        float: accuracy of predictions.

    """
    correct = 0
    for y_true, y_pred, logit in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)


best_val = 0.0


def default_log_fn(
    epoch: int,
    train_loss: float,
    losses: list[float],
    train_predictions: list[tuple[int, int, float]],
    train_accuracy: list[float],
    validation_predictions: list[tuple[int, int, float]],
    validation_accuracy: list[float]
) -> None:
    """Default logging function to print training and validation results.

    Args:
        epoch (int): current epoch number.
        train_loss (float): accumulated loss of the current epoch.
        losses (list): list of accumulated losses over epochs.
        train_predictions (list): list of training predictions for the current epoch.
        train_accuracy (list): list of training accuracies over epochs.
        validation_predictions (list): list of validation predictions for the current epoch.
        validation_accuracy (list): list of validation accuracies over epochs.

    """
    global best_val
    best_val = (
        best_val if best_val > validation_accuracy[-1] else validation_accuracy[-1]
    )
    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")


class SentenceSentimentTrain:
    """Training class for sentence sentiment analysis using a defined model."""

    def __init__(self, model: minitorch.Module) -> None:
        """Initialize with a model.

        Args:
            model (minitorch.Module): the model to train.

        """
        self.model = model

    def train(
        self,
        data_train: tuple[list[list[list[float]]], list[int]],
        learning_rate: float,
        batch_size: int = 10,
        max_epochs: int = 500,
        data_val: tuple[list[list[list[float]]], list[int]] = None,
        log_fn: callable = default_log_fn
) -> None:
        """Train the model using the provided training data.

        Args:
            data_train (tuple): tuple containing training features and labels.
            learning_rate (float): learning rate for the optimizer.
            batch_size (int): size of batches for training.
            max_epochs (int): maximum number of epochs to train.
            data_val (tuple, optional): tuple containing validation features and labels.
            log_fn (callable, optional): function to log training progress.

        """
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()

                # Save train predictions
                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]

                # Update
                optim.step()

            # Evaluate on validation set at the end of the epoch
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(
                    y_val,
                    backend=BACKEND,
                )
                x = minitorch.tensor(
                    X_val,
                    backend=BACKEND,
                )
                out = model.forward(x)
                validation_predictions += get_predictions_array(y, out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
            total_loss = 0.0


def encode_sentences(
    dataset: dict,
    N: int,
    max_sentence_len: int,
    embeddings_lookup: embeddings.GloveEmbedding,
    unk_embedding: list[float],
    unks: set[str]
) -> tuple[list[list[list[float]]], list[int]]:
    """Encode sentences into embeddings with padding.

    Args:
        dataset (Dataset): dataset containing sentences and labels.
        N (int): number of samples to encode.
        max_sentence_len (int): maximum length for padding sentences.
        embeddings_lookup (Embeddings): lookup object to get embeddings.
        unk_embedding (list): embedding for unknown words.
        unks (set): set to collect unknown words.

    Returns:
        tuple: tuple containing encoded sentences and labels.

    """
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0s to max sentence length in order to enable batching
        # TODO: move padding to training code
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                # use random embedding for unks
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    # load labels
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(
    dataset: dict,
    pretrained_embeddings: embeddings.GloveEmbedding,
    N_train: int,
    N_val: int = 0
) -> tuple[tuple[list[list[list[float]]], list[int]], tuple[list[list[list[float]]], list[int]]]:
    """Prepare and encode data for sentiment analysis training and validation.

    Args:
        dataset (dict): dataset dictionary containing training and validation splits.
        pretrained_embeddings (Embeddings): pre-trained embeddings for encoding.
        N_train (int): number of training samples to encode.
        N_val (int): number of validation samples to encode (default is 0).

    Returns:
        tuple: tuple containing encoded training data and validation data.

    """
    #  Determine max sentence length for padding
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    train_size = 450
    validation_size = 100
    learning_rate = 0.05 #originally 0.01
    max_epochs = 250

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )
