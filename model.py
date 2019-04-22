import os
import argparse


from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D, Activation, BatchNormalization
from keras.optimizers import Adam

import cloning.image_generator as ig

print('Loading input and ouput parameters..')
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument(
    'model_export_path',
    type=str,
    help='Path to directory to export the trained model in h5 format.'
)

MODEL_EXPORT_PATH = parser.parse_args().model_export_path

IMAGE_SHAPE = (160, 320, 3)

IMAGE_DIRECTORIES = [
    'first_drives',
    '3_forward_rounds',
    '3_backward_rounds',
    'edges_forward',
    'edges_backward',
    'edges_smooth',
    '2_rounds_for_testing',
    '3_more_fw_rounds',
    '3_more_bw_rounds',
    # 'driving_on_the_edge_fw_left',
    # 'driving_on_the_edge_fw_right',
    # '6_fw',
    # '6_bw',
    # '3_bw_hard',
    # '3_fw_hard',
    # '1_round_fw_hard_level'
]

LOAD_ALL_IMAGES_IN_MEMORY = False
BATCH_SIZE = 32
TRAIN_SET_SIZE = ig.load_image_registries(PROJECT_DIR, IMAGE_DIRECTORIES).shape[0]

print('Loading train and validation data sets...')
if LOAD_ALL_IMAGES_IN_MEMORY:
    all_data = ig.load_all_images(PROJECT_DIR, IMAGE_DIRECTORIES)
    train_generator = ig.image_generator_from_dataset(*all_data, batch_size=BATCH_SIZE, augment=True)
else:
    train_generator = ig.image_generator_from_files(PROJECT_DIR, IMAGE_DIRECTORIES, batch_size=BATCH_SIZE, augment=True)

print(f'Number of train samples: {TRAIN_SET_SIZE}')

valid_images, valid_meas = ig.load_all_images(PROJECT_DIR, ['validation_rounds', 'validation_rounds_bw'])
print(f'Number of valid samples: {valid_meas.shape[0]}')

print('Defining and compiling the model...')
model = Sequential([
    Cropping2D(cropping=((70, 25), (0, 0)), input_shape=IMAGE_SHAPE, name='crop'),
    Lambda(lambda image: image / 255.0 - 0.5, name='scale'),

    Conv2D(24, (5, 5), strides=(2, 2)),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(36, (5, 5), strides=(2, 2)),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(48, (5, 5), strides=(2, 2)),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(64, (3, 3)),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(64, (3, 3)),
    BatchNormalization(),
    Activation('relu'),

    Flatten(),

    Dense(100),
    BatchNormalization(),
    Activation('relu'),

    Dense(50),
    BatchNormalization(),
    Activation('relu'),

    Dense(10),
    BatchNormalization(),
    Activation('relu'),

    Dense(1, activation='linear')
])

model.compile(loss='mse', optimizer=Adam(lr=0.0005))

print('Training the model...')
steps_per_epoch = int(TRAIN_SET_SIZE / BATCH_SIZE)
validation_steps = int(valid_meas.shape[0] / BATCH_SIZE)

model.fit_generator(
    train_generator,
    validation_data=[valid_images, valid_meas],
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    max_queue_size=40,
    use_multiprocessing=True,
    workers=8
)

print(f'Exporting the model to {MODEL_EXPORT_PATH}')
model.save(MODEL_EXPORT_PATH)
