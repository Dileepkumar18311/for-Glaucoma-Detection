from model_selection import create_model
from preprocessing import train_generator, val_generator

model = create_model()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

model.save('../models/glaucoma_model.h5')
