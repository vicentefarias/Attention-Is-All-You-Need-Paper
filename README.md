# Attention is All You Need: Implementation of the Transformer Model

This repository contains an implementation of the **Transformer model** for sequence-to-sequence tasks, specifically trained and evaluated on a **bilingual translation dataset**. The implementation is based on the seminal paper **"Attention is All You Need"** by Vaswani et al., which introduced the attention mechanism to improve the efficiency and performance of neural networks in natural language processing tasks. https://arxiv.org/pdf/1706.03762

## Introduction

The Transformer model revolutionized the field of natural language processing by utilizing self-attention mechanisms, allowing the model to weigh the significance of different parts of the input sequence. This implementation focuses on training a Transformer for machine translation tasks, facilitating the translation between two languages with high accuracy and efficiency.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [References](#references)

## Requirements

To run this project, ensure you have the following packages installed:

- Python 3.x
- PyTorch
- TorchMetrics
- Tokenizers
- tqdm
- TensorBoard
- Datasets

You can install the required packages via pip:

```bash
pip install torch torchmetrics tokenizers tqdm tensorboard datasets
```

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/vicentefarias/Attention-Is-All-You-Need-Paper.git
    ```

## Dataset

This project utilizes a custom dataset class, `BilingualDataset`, designed to handle bilingual translation tasks efficiently. The class is built to work seamlessly with PyTorch's data loading utilities, allowing easy integration for training and evaluation.

## Dataset Class: `BilingualDataset`

The `BilingualDataset` class is designed to process pairs of source and target language texts, tokenize them, and prepare them for input into the Transformer model.

### Parameters
- `ds`: A dataset containing translation pairs, typically structured as a list of dictionaries with a 'translation' key.
- `tokenizer_src`: Tokenizer for the source language, responsible for encoding text into token IDs.
- `tokenizer_tgt`: Tokenizer for the target language, similarly responsible for encoding text.
- `src_lang`: Language code for the source language (e.g., "en" for English).
- `tgt_lang`: Language code for the target language (e.g., "fr" for French).
- `seq_len`: Maximum sequence length for both input and output sequences.

#### Methods
- `__len__()`: Returns the total number of samples in the dataset.
- `__getitem__(index)`: Retrieves and processes the translation pair at the specified index, returning a dictionary containing:
- `encoder_input`: Tensor for the encoder's input sequence.
- `decoder_input`: Tensor for the decoder's input sequence.
- `encoder_mask`: Mask tensor indicating non-padding tokens for the encoder.
- `decoder_mask`: Mask tensor indicating non-padding and causal tokens for the decoder.
- `label`: Tensor representing the target output for the decoder.
- `src_text`: The original source text string.
- `tgt_text`: The original target text string.

#### Causal Mask Function
The `causal_mask(size)` function generates a mask that prevents the decoder from attending to future tokens, ensuring predictions are made only based on past and present tokens

## Model Architecture

The model architecture implemented in this project follows the Transformer architecture, as introduced in the paper "Attention is All You Need" by Vaswani et al. This architecture leverages self-attention mechanisms and feed-forward networks to effectively handle sequence-to-sequence tasks, such as machine translation.

### Components
-  `Input Embeddings`: The InputEmbeddings class maps token indices to dense vectors of size d_model. The embeddings are scaled by the square root of d_model to stabilize training.
- `Positional Encoding`: The PositionalEncoding class adds positional information to the input embeddings, allowing the model to learn the order of the tokens. It employs sine and cosine functions to encode positions.
- `Multi-Head Attention`: The MultiHeadAttention class implements the multi-head self-attention mechanism, allowing the model to attend to different parts of the input sequence simultaneously. It includes linear transformations for queries, keys, and values.
- `Feed Forward Network`: The FeedForward class applies a two-layer feed-forward network to each position, with a ReLU activation function in between.
- `Residual Connections & Layer Normalization`: The ResidualConnection and LayerNormalization classes help stabilize the training process by applying residual connections and normalizing outputs after each sub-layer.
- `Encoder & Decoder Blocks`: The EncoderBlock and DecoderBlock classes consist of multi-head attention layers and feed-forward networks, with residual connections. The encoder processes the input sequence, while the decoder generates the output sequence based on the encoder's output.
- `Encoder & Decoder`: The Encoder and Decoder classes stack multiple encoder and decoder blocks, respectively. They implement the forward pass through each layer and apply final layer normalization.
- `Projection Layer`: The ProjectionLayer maps the decoder output to the target vocabulary size using a linear layer followed by a log-softmax activation.
- `Transformer Model`: The Transformer class combines the encoder, decoder, embedding layers, positional encodings, and the projection layer into a single model. It manages the encoding and decoding process during training and inference.

    
### Model Initialization

A helper function, build_transformer, initializes the transformer model with specified parameters such as vocabulary sizes, sequence lengths, model dimensions, and number of layers. This function constructs all components of the transformer and initializes their parameters using Xavier uniform initialization to improve convergence during training.

```python
def build_transformer(
    src_vocab_size: int,   # Size of the source vocabulary
    tgt_vocab_size: int,   # Size of the target vocabulary
    src_seq_len: int,      # Maximum sequence length for the source input
    tgt_seq_len: int,      # Maximum sequence length for the target input
    d_model: int = 512,    # Dimensionality of the embeddings and the hidden states
    N: int = 6,            # Number of encoder and decoder layers
    h: int = 8,            # Number of attention heads in multi-head attention
    dropout: float = 0.1,  # Dropout rate for regularization
    d_ff: int = 2048       # Dimensionality of the feed-forward network
) -> Transformer:
```

## Training

The training process involves setting up the model, optimizer,  data loaders, and running the training loop for a specified number of epochs. The process is outlined below:

### Setup

1. **Device Configuration**: Determine whether to use a GPU or CPU for training.
2. **Model and Dataset Initialization**: Load the dataset and initialize the transformer model.
3. **TensorBoard Setup**: Create a `SummaryWriter` for logging training metrics.
4. **Optimizer**: Use Adam optimizer for model training with a specified learning rate.

### Training Loop

The training loop consists of the following steps:

1. **Forward Pass**:
   - Input the encoder and decoder data into the model.
   - Compute the output probabilities for the next tokens in the sequence.

2. **Loss Calculation**:
   - Calculate the cross-entropy loss using the output probabilities and the target labels.
   - Apply label smoothing to enhance generalization.

3. **Backward Pass**:
   - Compute gradients and update the model parameters.

4. **Logging**:
   - Log the training loss to TensorBoard for visualization.

5. **Validation**:
   - At the end of each epoch, evaluate the model's performance on the validation dataset.

6. **Model Saving**:
   - Save the model state after each epoch for potential future use.

## Evaluation

During the evaluation phase, the model's performance is assessed using a validation dataset. The evaluation process involves generating predictions for a set of source sentences and comparing them to the corresponding target sentences.

### Evaluation Procedure

1. **Model Evaluation Mode**:
   - Set the model to evaluation mode to disable dropout and ensure deterministic outputs.

2. **Prediction Generation**:
   - Use the `greedy_decode` function to generate predictions for the validation inputs. This function involves:
     - Encoding the source input to obtain the encoder output.
     - Iteratively predicting the next token in the target sequence until the end-of-sequence token is generated or a maximum length is reached.

3. **Text Decoding**:
   - Convert the predicted token IDs back to text using the target tokenizer.

4. **Metrics Calculation**:
   - After generating predictions for a batch of examples, compute the following metrics:
     - **Character Error Rate (CER)**
     - **Word Error Rate (WER)**
     - **BLEU Score**
   - These metrics are calculated to evaluate the quality of the translations produced by the model.

5. **Logging**:
   - Log the metrics to TensorBoard for visualization. This helps track the model's performance over time.

## References

This implementation is inspired by the following works:

1. **Attention is All You Need**  
   Vaswani, A., Shard, A., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is All You Need*. In Advances in Neural Information Processing Systems (NIPS), 30.

