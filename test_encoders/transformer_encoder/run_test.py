import tensorflow as tf

from transformer_encoder import TransformerEncoder

# Usage example with original Transformer hyperparameters
num_layers = 6
d_model = 512
num_heads = 8
dff = 2048
input_vocab_size = 10000
maximum_position_encoding = 10000

model = TransformerEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding)

# Define an optimizer (e.g., Adam)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define a loss function (e.g., categorical cross-entropy for classification)
loss_function = tf.keras.losses.CategoricalCrossentropy()

# Define a training loop
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_function(targets, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# Example of usage in the training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in training_data:  # Provide training data
        loss = train_step(inputs, labels)
        # Log or print the loss for monitoring

# Testing the model
input = tf.random.uniform((64, 40), dtype=tf.int64, minval=0, maxval=10000)
output = model(input, training=False)
print(output.shape)
