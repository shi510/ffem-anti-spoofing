import os

import train.utils as utils
import train.input_pipeline as input_pipeline
import train.config
from train.callbacks import LossTensorBoard
import net_arch.models
import train.blocks

import tensorflow as tf


def build_dataset(config):
    train_ds = input_pipeline.make_tfdataset(
        config['train_file'],
        config['img_root_path'],
        2,
        config['batch_size'],
        config['shape'][:2])
    test_ds = input_pipeline.make_tfdataset(
        config['test_file'],
        config['img_root_path'],
        2,
        config['batch_size'],
        config['shape'][:2],
        testset=True)
    return train_ds, test_ds


def build_backbone_model(config):
    """
        The saved model should contain the layers below:
            1. Input Layer
            2. Backbone Net layer
            3. Embeeding Layer
    """

    if os.path.exists(config['saved_model']):
        net = tf.keras.models.load_model(config['saved_model'])
        if os.path.isdir(config['saved_model']):
            net.load_weights(config['saved_model']+'/variables/variables')
        if len(net.layers) is not 2 and '.h5' not in config['saved_model']:
            y = x = net.layers[0].input
            y = net.layers[1](y)
            net = tf.keras.Model(x, y, name = net.name)
        print('')
        print('******************** Loaded saved weights ********************')
        print('')
    elif len(config['saved_model']) != 0:
        print(config['saved_model'] + ' can not open.')
        exit(1)
    else :
        net = net_arch.models.get_model(config['model'], config['shape'])

    return net


def build_basic_softmax_model(config, num_id):
    y = x = tf.keras.Input(config['shape'])
    y = build_backbone_model(config)(y)

    def _embedding_layer(feature):
        y = x = tf.keras.Input(feature.shape[1:])
        y = tf.keras.layers.Dropout(rate=0.5)(y)
        # y = tf.keras.layers.BatchNormalization()(y)
        # y = tf.keras.layers.Flatten()(y)
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.BatchNormalization()(y)
        # y = train.blocks.attach_embedding_projection(y, config['embedding_dim'])
        # y = train.blocks.attach_l2_norm_features(y, scale=2)
        return tf.keras.Model(x, y, name='GNAP_l2norm_embedding')(feature)


    def _classification_layer(feature):
        y = x = tf.keras.Input(feature.shape[1:])
        y = tf.keras.layers.Dense(num_id)(y)
        y = tf.keras.layers.Softmax()(y)
        return tf.keras.Model(x, y, name='softmax_classifier')(feature)


    y = _embedding_layer(y)
    y = _classification_layer(y)
    return tf.keras.Model(x, y, name=config['model_name'])


def build_callbacks(config):
    callback_list = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=3, min_lr=1e-5)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint'+os.path.sep+config['model_name'],
        save_weights_only=False,
        mode='min',
        save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=5)
    tensorboard_log = LossTensorBoard(
        100, os.path.join('logs', config['model_name']))

    if not config['lr_decay']:
        callback_list.append(reduce_lr)
    callback_list.append(checkpoint)
    callback_list.append(early_stop)
    callback_list.append(tensorboard_log)
    return callback_list


def build_optimizer(config):
    # In tf-v2.3.0, Do not use tf.keras.optimizers.schedules with ReduceLR callback.
    if config['lr_decay']:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            config['lr'],
            decay_steps=config['lr_decay_steps'],
            decay_rate=config['lr_decay_rate'],
            staircase=True)
    else:
        lr = config['lr']

    opt_list = {
        'adam': 
            tf.keras.optimizers.Adam(learning_rate=lr),
        'sgd':
            tf.keras.optimizers.SGD(learning_rate=lr,
                momentum=0.9, nesterov=True)
    }
    if config['optimizer'] not in opt_list:
        print(config['optimizer'], 'is not support.')
        print('please select one of below.')
        print(opt_list.keys())
        exit(1)
    return opt_list[config['optimizer']]


def convert_tflite_int8(model, ds):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset_gen():
        for n, (x, _ )in enumerate(ds.take(10000)):
            if n % 100 == 0:
                print(n)
            # Get sample input data as a numpy array in a method of your choosing.
            # The batch size should be 1.
            yield [x[0]]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8  # or tf.uint8
    tflite_quant_model = converter.convert()
    with open(model.name + '.tflite', 'wb') as f:
        f.write(tflite_quant_model)


if __name__ == '__main__':
    config = train.config.config
    train_ds, test_ds = build_dataset(config)
    net = build_basic_softmax_model(config, 2)
    opt = build_optimizer(config)
    net.summary()

    net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    net.fit(train_ds, validation_data=test_ds, epochs=config['epoch'], verbose=1,
        workers=input_pipeline.TF_AUTOTUNE,
        callbacks=build_callbacks(config))
    net.save('{}.h5'.format(net.name), include_optimizer=False)
