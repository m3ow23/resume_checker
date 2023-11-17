# import tensorflow as tf
import numpy as np

# loss_function = tf.keras.losses.CategoricalCrossentropy()

# # Corrected one-hot encoded labels
# tokenized_labels = tf.constant([[1,    0,  0], [0, 0, 0]], tf.float32)
# predictions = tf.constant([[.9, .05, .05], [0, 0, 0]], tf.float32)

# epsilon = 1e-8
# # print(predictions + epsilon)

# predictions = tf.where(predictions == 0, epsilon, predictions)

# loss = loss_function(tokenized_labels, predictions)

# print("tensorflow:", loss.numpy())

# # tokenized_labels = np.array([[1,    0,  0], [0, 0, 0]])
# # predictions = np.array([[.90,.05,.05], [0, 0, 0,]])

# # Calculate Categorical Cross-Entropy loss without applying softmax
# total_loss = 0
# for tokenized_label, prediction in zip(tokenized_labels, predictions):
#     total_loss += np.sum(tokenized_label * -np.log(prediction))
# print("manual:", total_loss / len(predictions))

# # print(tf.nn.softmax(np.array([-1.0000000e+09, -1.0000000e+09, -1.0000000e+09, -1.0000000e+09])))

batch = [
    [
        0, 
        1, 
        0
    ], 
    [
        1, 
        0, 
        1
    ]
]

token_ids = [5, 7, 8]
input_vocab_size = 10

index = 0
for sequence in batch:
    for token_index, token in enumerate(sequence):
        if (token == 0):
            token_label = [0] * input_vocab_size
            token_label[token_ids[index]] = 1
            sequence[token_index] = token_label
            index += 1
        elif(token == 1):
            sequence[token_index] = [0] * input_vocab_size

print(batch) 