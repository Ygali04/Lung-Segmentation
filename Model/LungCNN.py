from Data import Dataset, Helpers

imgs_train, imgs_test, masks_train, masks_test = load()
print(imgs_train.shape, imgs_test.shape, masks_train.shape, masks_test.shape)

def SimpleConvModel():
    model = tf.keras.models.Sequential(
        [
            Conv2D(filters = 16, kernel_size = 3, strides = 1, padding="same", activation="relu"),
            Conv2D(filters = 32, kernel_size = 3, strides = 1, padding="same", activation="relu"),
            Conv2D(filters = 16, kernel_size = 3, strides = 1, padding="same", activation="relu"),
            Conv2D(filters = 1, kernel_size = 3, strides = 1, padding="same", activation="sigmoid"),
        ]
    ) 
    return model

model = SimpleConvModel()

def train(model, imgs, masks, loss, name, epochs):
    optimizer = tf.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.fit(imgs, masks, epochs=epochs, callbacks=[ShowLearning(imgs, masks, name)],)

train(model, imgs_train, masks_train, tf.losses.BinaryCrossentropy(), 'simple_conv', EPOCHS)

model.summary() #visualize model architecture

interact(lambda epoch: show_training_image('simple_conv', epoch), epoch=(0, EPOCHS-1));
print(masks_train.sum()/(masks_train.size)* 100) #analyze why, despite accuracy, model performs poorly
