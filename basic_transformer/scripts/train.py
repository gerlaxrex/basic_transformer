import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from basic_transformer import DATA_DIR
from basic_transformer.model import Transformer
from basic_transformer.scripts.dataset_preparation import dataloader, dataset

# Define the transformer model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(
    src_vocab_size=10_000,
    tgt_vocab_size=10_000,
    n_decoder_blocks=5,
    n_encoder_blocks=5,
    max_len=512,
    n_heads=8,
    embedding_dim=1024,
    pad_idx=0,
)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Set the device to GPU if available, otherwise use CPU

model.to(device)
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    print(f"Start epoch {epoch}")
    model.train()
    running_loss = 0.0
    for batch_idx, (input_ids, target_ids) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        # Move tensors to the device
        input_ids = input_ids.to(device)
        output_ids = target_ids[:, 1:].to(device)
        target_ids = target_ids[:, :-1].to(device)


        # Forward pass
        logits = model(input_ids, target_ids)

        # Compute loss
        loss = criterion(logits.permute(0, 2, 1), output_ids)
        running_loss += loss.item()

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

    # Print the average loss for the epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')

# Save the model state
torch.save(model.state_dict(), DATA_DIR / "saved_models" / "Riccah.pth")

input = "Ciao, come stai? "
input_tensor = torch.tensor(dataset.src_tokenizer.encode_as_ids(input)).unsqueeze(0)

output_sentence = f""

fn = model.greedy_decode(input_tensor,
                         bos_idx=1,
                         max_len=80)

for _ in range(20):
    next_token = next(fn)
    id = dataset.tgt_tokenizer.IdToPiece(int(next_token.detach()))
    output_sentence += id.replace("_", "")
    print(output_sentence)