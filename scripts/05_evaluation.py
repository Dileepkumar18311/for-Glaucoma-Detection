import tensorflow as tf
from model_selection import create_model
from preprocessing import test_generator

model = create_model()
model.load_weights('../models/glaucoma_model.h5')  # Load the model weights

results = model.evaluate(
    test_generator,
    steps=test_generator.samples // test_generator.batch_size
)

print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")
