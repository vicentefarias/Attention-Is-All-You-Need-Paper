import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # Get the start of sequence and end of sequence token indices
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output to reuse for each decoding step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the start of sequence token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        # Break if the maximum length is reached
        if decoder_input.size(1) == max_len:
            break

        # Build a causal mask for the decoder input
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Perform decoding step
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the predicted probabilities for the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)  # Select the token with the highest probability

        # Append the predicted token to the decoder input
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # Break if the end of sequence token is generated
        if next_word == eos_idx:
            break

    # Return the generated sequence without the batch dimension
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()  # Set the model to evaluation mode
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80  # Width for console output formatting

    with torch.no_grad():  # Disable gradient calculation
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)  # Move encoder input to device
            encoder_mask = batch['encoder_mask'].to(device)  # Move encoder mask to device

            # Ensure the batch size is 1 for validation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Decode the input to get the model output
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Extract the source, target, and predicted texts
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Store the texts for metric calculation
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target, and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            # Stop after processing the specified number of examples
            if count == num_examples:
                print_msg('-'*console_width)
                break

        if writer:
            # Evaluate various metrics
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()

def get_all_sentences(ds, lang):
    # Generator function to yield sentences from the dataset
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # Path for the tokenizer file
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    # If tokenizer doesn't exist, build one
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))  # Create a new WordLevel tokenizer
        tokenizer.pre_tokenizer = Whitespace()  # Set the pre-tokenizer to split by whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)  # Train the tokenizer
        tokenizer.save(str(tokenizer_path))  # Save the tokenizer to file
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))  # Load the existing tokenizer
    
    return tokenizer

def get_ds(config):
    # Load the dataset and split into training and validation sets
    print(config)
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers for both source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split the dataset: 90% training, 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create bilingual datasets for training and validation
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    # Determine maximum lengths of source and target sentences
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Create DataLoader for training and validation
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    # Build and return the transformer model
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Set the device to use for training (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Create directory for saving model weights
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    # Load the dataset and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Initialize TensorBoard writer for logging
    writer = SummaryWriter(config['experiment_name'])

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    # If preloading is required, load the model weights
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # Define the loss function with label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()  # Set model to training mode
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:
            # Move batch data to the specified device
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Forward pass through the model
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)  # Get labels for loss calculation

            # Compute loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})  # Update progress bar with loss

            # Log the loss to TensorBoard
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Clear gradients for the next iteration

            global_step += 1  # Increment global step

        # Run validation at the end of each epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, 
                       lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Save the model state after each epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # Save model parameters
            'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer parameters
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output
    config = get_config()  # Load configuration settings
    train_model(config)  # Start training the model
