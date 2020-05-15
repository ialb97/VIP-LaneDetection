from load_tfrecord import *

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch%25==0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("checkpoints/model_{}testest.hd5".format(epoch))


def train(n_epochs,lr,train_data,val_data):
    #encoding block
    inputs = tf.keras.Input(shape=(256,512,3))
    e_conv_1 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',use_bias=False)(inputs)
    e_bn_1 = tf.keras.layers.BatchNormalization()(e_conv_1)
    e_relu_1 = tf.keras.layers.ReLU()(e_bn_1)
    e_conv_2 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',use_bias=False)(e_relu_1)
    e_bn_2 = tf.keras.layers.BatchNormalization()(e_conv_2)
    e_relu_2 = tf.keras.layers.ReLU()(e_bn_2)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(e_relu_2)
    e_conv_3 = tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same',use_bias=False)(pool_1)
    e_bn_3 = tf.keras.layers.BatchNormalization()(e_conv_3)
    e_relu_3 = tf.keras.layers.ReLU()(e_bn_3)
    e_conv_4 = tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same',use_bias=False)(e_relu_3)
    e_bn_4 = tf.keras.layers.BatchNormalization()(e_conv_4)
    e_relu_4 = tf.keras.layers.ReLU()(e_bn_4)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(e_relu_4)
    e_conv_5 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same',use_bias=False)(pool_2)
    e_bn_5 = tf.keras.layers.BatchNormalization()(e_conv_5)
    e_relu_5 = tf.keras.layers.ReLU()(e_bn_5)
    e_conv_6 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same',use_bias=False)(e_relu_5)
    e_bn_6 = tf.keras.layers.BatchNormalization()(e_conv_6)
    e_relu_6 = tf.keras.layers.ReLU()(e_bn_6)
    e_conv_7 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same',use_bias=False)(e_relu_6)
    e_bn_7 = tf.keras.layers.BatchNormalization()(e_conv_7)
    e_relu_7 = tf.keras.layers.ReLU()(e_bn_7)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(e_relu_7)
    e_conv_8 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',use_bias=False)(pool_3)
    e_bn_8 = tf.keras.layers.BatchNormalization()(e_conv_8)
    e_relu_8 = tf.keras.layers.ReLU()(e_bn_8)
    e_conv_9 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',use_bias=False)(e_relu_8)
    e_bn_9 = tf.keras.layers.BatchNormalization()(e_conv_9)
    e_relu_9 = tf.keras.layers.ReLU()(e_bn_9)
    e_conv_10 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',use_bias=False)(e_relu_9)
    e_bn_10 = tf.keras.layers.BatchNormalization()(e_conv_10)
    e_relu_10 = tf.keras.layers.ReLU()(e_bn_10)
    pool_4 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(e_relu_10)
    e_conv_11 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',use_bias=False)(pool_4)
    e_bn_11 = tf.keras.layers.BatchNormalization()(e_conv_11)
    e_relu_11 = tf.keras.layers.ReLU()(e_bn_11)
    e_conv_12 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',use_bias=False)(e_relu_11)
    e_bn_12 = tf.keras.layers.BatchNormalization()(e_conv_12)
    e_relu_12 = tf.keras.layers.ReLU()(e_bn_12)
    e_conv_13 = tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,padding='same',use_bias=False)(e_relu_12)
    e_bn_13 = tf.keras.layers.BatchNormalization()(e_conv_13)
    e_relu_13 = tf.keras.layers.ReLU()(e_bn_13)



    #decoding block
    d_deconv_1 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=4,strides=2,use_bias=False,padding='same')(e_relu_13)
    d_bn_1 = tf.keras.layers.BatchNormalization()(d_deconv_1)
    d_relu_1 = tf.keras.layers.ReLU()(d_bn_1)
    add_1 = tf.keras.layers.Add()([d_relu_1,e_relu_10])
    d_bn_2 = tf.keras.layers.BatchNormalization()(add_1)
    d_relu_2 = tf.keras.layers.ReLU()(d_bn_2)
    d_deconv_2 = tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=4,strides=2,use_bias=False,padding='same')(d_relu_2)
    d_bn_3 = tf.keras.layers.BatchNormalization()(d_deconv_2)
    d_relu_3 = tf.keras.layers.ReLU()(d_bn_3)
    add_2 = tf.keras.layers.Add()([d_relu_3,e_relu_7])
    d_bn_4 = tf.keras.layers.BatchNormalization()(add_2)
    d_relu_4 = tf.keras.layers.ReLU()(d_bn_4)
    d_deconv_3 = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=4,strides=2,use_bias=False,padding='same')(d_relu_4)
    d_bn_5 = tf.keras.layers.BatchNormalization()(d_deconv_3)
    d_relu_5 = tf.keras.layers.ReLU()(d_bn_5)
    add_3 = tf.keras.layers.Add()([d_relu_5,e_relu_4])
    d_bn_6 = tf.keras.layers.BatchNormalization()(add_3)
    d_relu_6 = tf.keras.layers.ReLU()(d_bn_6)
    d_deconv_4 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(d_relu_6)
    d_bn_7 = tf.keras.layers.BatchNormalization()(d_deconv_4)
    d_relu_7 = tf.keras.layers.ReLU()(d_bn_7)
    add_4 = tf.keras.layers.Add()([d_relu_7,e_relu_2])
    d_bn_8 = tf.keras.layers.BatchNormalization()(add_4)
    d_relu_8 = tf.keras.layers.ReLU()(d_bn_7)

    d_conv1 = tf.keras.layers.Conv2D(filters=1,kernel_size=1,use_bias=False,strides=1,padding='same')(d_relu_8)
    d_bn_9 = tf.keras.layers.BatchNormalization()(d_conv1)
    d_relu_9 = tf.keras.layers.ReLU(max_value=1)(d_bn_9)

    model = tf.keras.Model(inputs=inputs, outputs=d_relu_9)
    #progress = tf.keras.utils.Progbar(target, width=30, verbose=1, interval=0.05, stateful_metrics=None,unit_name='step')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad=True)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])#fill in
    saver = CustomSaver()
    #model.summary()

    model.fit(x=train_data,epochs=n_epochs,validation_data=val_data,verbose=1,callbacks=[saver])

    model.save(f'model_{n_epochs}epochsfinal.h5')



def main():
    train_ds = get_dataset_from_tfrecords(tfrecords_dir='/home/car-sable/LaneDetection/tf_records',batch_size=4)
    val_ds = get_dataset_from_tfrecords(tfrecords_dir='/home/car-sable/LaneDetection/tf_records',split='validate',batch_size=4)

    train(500,.001,train_ds,val_ds)





if __name__ == '__main__':
    main()
