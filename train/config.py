config = {
    #
    # Save trained model named with 'model_name'.h5.
    # The best model at each epoch is saved to the folder ./checkpoint/'model_name'.
    #
    'model_name': 'MobileNetV3_face_spoof_gavg',

    #
    # Restore trained weights.
    # The architecture must be same with 'model' option.
    # checkpoint folder or keras saved file including extension.
    #
    'saved_model': '',
    'batch_size' : 128,
    'shape' : [128, 128, 3],
    'embedding_dim' : 512,

    #
    # If 'saved_model' not exsits, then it will be built with this architecture.
    # Choose one of below: 
    # 1. MobileNetV2
    # 2. MobileNetV3
    # 3. EfficientNetB3
    #
    'model' : 'MobileNetV3',

    #
    # There are two options.
    #  1. adam
    #  2. sgd with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'sgd',
    'epoch' : 15,

    #
    # initial learning rate.
    #
    'lr' : 1e-4,

    #
    # lr * decay_rate ^ (steps / decay_steps)
    #
    'lr_decay': False,
    'lr_decay_steps' : 10000,
    'lr_decay_rate' : 0.9,

    #
    # It should be absolute path that indicates face image file location in 'train_file' contents.
    #
    'img_root_path': 'path/CelebA_Spoof',

    #
    # See README.md file how to save this file.
    #
    'train_file': 'celeba_spoof_train.json',
    'test_file': 'celeba_spoof_test.json',
}
