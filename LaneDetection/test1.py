from load_tfrecord import *

import tensorflow as tf

import numpy as np
from PIL import Image
import timeit


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if physical_devices:
    try:
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
    except RuntimeError as e:
        print(e)

test_ds = get_dataset_from_tfrecords(tfrecords_dir='/home/car-sable/LaneDetection/tf_records', batch_size=1,split='test')

model = tf.keras.models.load_model('checkpoints/model_475testest.hd5')
count = 0
running_sum = 0
for record in test_ds:
    start_time = timeit.default_timer()
    IN = record[0].numpy()
    label = record[1].numpy().reshape(256,512)
    out = (model.predict(IN).reshape(256,512))*255
    IN = IN.reshape(256,512,3)
    #INIM = Image.fromarray(np.uint8(IN),'RGB')
    #LABELIM = Image.fromarray(np.uint8(label))
    #OUTIM = Image.fromarray(np.uint8(out))
    #INIM.save(f'output_test/gt/{count}.png')
    #LABELIM.save(f'output_test/labels/{count}.png')
    #OUTIM.save(f'output_test/outputs/{count}.png')
    count = count + 1
    elapsed = timeit.default_timer() - start_time
    running_sum = running_sum + elapsed
    break

print(count)
print(f'FPS={1/(running_sum/count)}')
