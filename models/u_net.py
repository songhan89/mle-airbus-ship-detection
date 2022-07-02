from tensorflow.keras import models, layers

# Build U-Net model
def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)

def UNet_keras(gaussian_noise=0.1, upsample_mode='SIMPLE', net_scaling=None, img_shape=(768, 768), edge_crop=16):
    
    input_img = layers.Input((img_shape[0], img_shape[1], 3), name = 'RGB_Input')
    
    pp_in_layer = input_img
    
    if upsample_mode=='DECONV':
        upsample=upsample_conv
    else:
        upsample=upsample_simple
    
    if net_scaling is not None:
        pp_in_layer = layers.AvgPool2D(net_scaling)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(gaussian_noise)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = layers.MaxPooling2D((2, 2)) (c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = layers.MaxPooling2D((2, 2)) (c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = layers.MaxPooling2D((2, 2)) (c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    d = layers.Cropping2D((edge_crop, edge_crop))(d)
    d = layers.ZeroPadding2D((edge_crop, edge_crop))(d)
    
    if net_scaling is not None:
        d = layers.UpSampling2D(net_scaling)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    
    return seg_model