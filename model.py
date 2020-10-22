from keras.models import Model
from keras.layers import Conv2D, Input, Concatenate
from keras.layers import MaxPool2D, BatchNormalization, AvgPool2D


weight_path = './weights/weights.hdf5'

def HazDesNet():
    input_1 = Input(shape=(None, None, 3))

    conv_1 = Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1))(input_1)

    conv_1 = Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1))(conv_1)

    max_pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)

    max_pool_1 = BatchNormalization()(max_pool_1)

    conv_2 = Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1))(max_pool_1)

    max_pool_2 = MaxPool2D(pool_size=(5, 5), strides=(1, 1))(conv_2)
    avg_pool_2 = AvgPool2D(pool_size=(5, 5), strides=(1, 1))(conv_2)

    max_avg_pool = Concatenate()([max_pool_2, avg_pool_2])

    conv_3 = Conv2D(filters=1, kernel_size=(6, 6), strides=(1, 1), activation='sigmoid')(max_avg_pool)

    model = Model(inputs=input_1, outputs=conv_3)
    
    return model


def load_HazDesNet():
    model = HazDesNet()
    model.load_weights(weight_path)

    return model

if __name__ == "__main__":
    model = load_HazDesNet()
    model.summary()