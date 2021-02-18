import Config


def create_model(input_shape):
        from keras import Input, models
        from keras.layers import Flatten, Dense

        y = Input(shape=input_shape)
        z = Flatten()(y)
        z = Dense(Config.DENSE1_SIZE, activation=Config.DENSE1_ACTIVATION)(z)
        z = Dense(Config.DENSE2_SIZE, activation=Config.DENSE2_ACTIVATION)(z)

        # our model will accept the inputs of the two branches and
        # then output a single value
        model = models.Model(y, z)

        model.compile(optimizer=Config.OPTIMIZER, loss='mse')
        model.summary()
        return model