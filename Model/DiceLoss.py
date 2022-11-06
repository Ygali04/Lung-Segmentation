from Data import Dataset, Helpers
import LungCNN

def dice(true_mask, predicted_mask):
    true_mask, predicted_mask = true_mask.astype(bool).flatten(), predicted_mask.astype(bool).flatten()
    return  2 * sum(true_mask & predicted_mask) / (sum(true_mask) + sum(predicted_mask))

img_num = 43

img = np.expand_dims(imgs_train[img_num],0)
true_mask = masks_train[img_num]
pred_mask = model.predict(img) > 0.5

show_lung_mask_pred_sbs(img, true_mask, pred_mask)
print("Dice score:", dice(true_mask, pred_mask))

model_with_dice = SimpleConvModel()
train(model_with_dice, imgs_train, masks_train, dice_loss, 'simple_conv_with_dice', EPOCHS)

interact(lambda epoch: show_training_image('simple_conv_with_dice', epoch), epoch=(0, EPOCHS-1));
