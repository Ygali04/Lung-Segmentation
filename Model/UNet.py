from Data import Dataset, Helpers

def section(conv1_filters, conv2_filters): #Represents one horizontal "section" of the U
    return tf.keras.Sequential(
        [
            Conv2D(conv1_filters, 3, padding="same", activation="relu"),
            Conv2D(conv2_filters, 3, padding="same", activation="relu"),
        ]
    )

class U_Net(tf.keras.Model):
    def __init__(self):
        super(U_Net, self).__init__() #What do the numbers represent? 
        self.section1 = section(16, 16)
        self.section2 = section(32, 32)
        self.section3 = section(32, 64) #Bottom of the U!
        self.section4 = section(32, 32)
        self.section5 = section(16, 16)
        self.final_conv = Conv2D(1, 3, padding="same", activation="sigmoid")
        self.maxpool1, self.maxpool2 = MaxPool2D(2), MaxPool2D(2) #Why are there two of these?
        self.upsample1, self.upsample2 = UpSampling2D(2), UpSampling2D(2)

    def call(self, inputs):
        input1 = self.section1(inputs)
        input2 = self.section2(self.maxpool1(input1))
        input3 = self.section3(self.maxpool2(input2))
        input4 = self.section4(concatenate([input2, self.upsample1(input3)]))
        input5 = self.section5(concatenate([input1, self.upsample2(input4)]))
        output = self.final_conv(input5)
        return output
      
unet = U_Net()

train(unet, imgs_train, masks_train, dice_loss, 'unet_with_dice', epochs=EPOCHS)
interact(lambda epoch: show_training_image('unet_with_dice', epoch), epoch=(0, EPOCHS-1));

thresh = 0.65
preds = unet(imgs_test).numpy()
preds = preds >= thresh

def show_preds_helper(imgs_test, masks_test, preds, ind):
    show_lung_mask_pred_sbs(imgs_test[ind], masks_test[ind], preds[ind])
    plt.show()

interact(lambda i: show_preds_helper(imgs_test, masks_test, preds, i), i=(0, len(preds)-1));
