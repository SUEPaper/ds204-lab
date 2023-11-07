import torch
import torch.nn as nn
import torch.nn.functional as F
import lab1
import util
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm


songs = lab1.load_training_data()
example_song = songs[0]
# print("\nExample song: ")
# print(example_song)


# lab1.play_song(example_song)


# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs) 
# Find all unique characters in the joined string


vocab = sorted(set(songs_joined))
# print("There are", len(vocab), "unique characters in the dataset")
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)



# print('{')
# for char,_ in zip(char2idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
# print('  ...\n}')


def vectorize_string(string):
  vectorized_songs = np.array([char2idx[song] for song in string ])
  return vectorized_songs
vectorized_songs = vectorize_string(songs_joined)


# print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
# check that vectorized_songs is a numpy array
assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"


def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)
  input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
  output_batch = [vectorized_songs[i+1:i+seq_length+1] for i in idx]
  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = torch.tensor(np.reshape(input_batch, [batch_size, seq_length]))
  y_batch = torch.tensor(np.reshape(output_batch, [batch_size, seq_length]))
  return x_batch, y_batch

test_args = (vectorized_songs, 10, 2)
# if not lab1.test_batch_func_types(get_batch, test_args) or \
#    not lab1.test_batch_func_shapes(get_batch, test_args) or \
#    not lab1.test_batch_func_next_step(get_batch, test_args): 
#    print("======\n[FAIL] could not pass tests")
# else: 
#    print("======\n[PASS] passed all tests!")

x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

# for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
#     print("Step {:3d}".format(i))
#     print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
#     print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    def init_weights(m):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = nn.Sequential(
        nn.Embedding(vocab_size, embedding_dim),
        LSTM(embedding_dim, rnn_units),
        nn.Linear(rnn_units, vocab_size)
    )
    return model
def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()
    if isinstance(m, LSTM):
        m.init_weights()
model = build_model(vocab_size=len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
# print(model)

x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
# print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
# print(y.shape)
# print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = torch.multinomial(F.softmax(pred[0], dim=-1), num_samples=1)
sampled_indices = sampled_indices.squeeze().numpy()
# print(sampled_indices)
# print("Input: \n", repr("".join(idx2char[x[0]])))
# print()
# print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
def compute_loss(labels, logits):
    logits = logits.permute(0,2,1)
    labels=labels.to(torch.int64)
    loss = F.cross_entropy(logits, labels)
    return loss
example_batch_loss = compute_loss(y, pred) 

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.item())
num_training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1
# Model parameters: 
vocab_size = len(vocab)
embedding_dim = 256 
rnn_units = 1024  # Experiment between 1 and 2048
# Checkpoint location: 
checkpoint_dir = './training_checkpoints.pt'
# checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
model.train()
optimizer = torch.optim.Adam(model.parameters(),learning_rate)
def train_step(x, y):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = compute_loss(y, y_hat)
    loss.backward()
    optimizer.step()
    return loss
history = []
plotter = util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
# for iter in tqdm(range(num_training_iterations)):
#   x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
#   loss = train_step(x_batch, y_batch)
#   history.append(loss.item())
#   plotter.plot(history)
# #   if iter % 100 == 0:     
# #     torch.save({
# #                     'model_state_dict': model.state_dict(),
# #                     'optimizer_state_dict': optimizer.state_dict(),
# #                     }, checkpoint_dir)
# torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     }, checkpoint_dir)
# plt.show()

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
checkpoint = torch.load(checkpoint_dir)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string, generation_length=10000):
    # Evaluation step (generating text using the learned RNN model)

    # Convert the start string to numbers (vectorize)
    input_eval = [char2idx[num] for num in start_string]
    input_eval = torch.tensor(input_eval).unsqueeze(0)
    # Empty string to store the generated text
    text_generated = []
    # Here batch size == 1
    # model.apply(weight_reset)
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = predictions.squeeze(0)
        predicted_id = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1)[-1, 0].item()
        input_eval = torch.tensor([predicted_id]).unsqueeze(0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)
generated_text = generate_text(model, start_string="X", generation_length=3000)
generated_songs = lab1.extract_song_snippet(generated_text)
for i, song in enumerate(generated_songs): 
  # Synthesize the waveform from a song
#   mdl.lab1.play_song(song)
 
   print(song, end="\n\n\n\n")