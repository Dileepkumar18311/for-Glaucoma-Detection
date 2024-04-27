import tensorflow as tf
from model_selection import create_model
from preprocessing import train_generator, val_generator

model = create_model()

for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

model.save('../models/glaucoma_model_finetuned.h5')
