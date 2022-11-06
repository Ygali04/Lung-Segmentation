import Dataset

np.random.seed(12)
tf.random.set_seed(12)

def load():
    imgs_paths = sorted([Path(IMG_PATH) / p for p in os.listdir(IMG_PATH)])
    masks_paths = sorted([Path(MASK_PATH) / p for p in os.listdir(MASK_PATH)])
    imgs = [plt.imread(p) for p in imgs_paths]
    masks = [plt.imread(p)[:, :, 0][:, :, None] > 0.5 for p in masks_paths]
    imgs, masks = np.array(imgs)[:,:,:,0:1], np.array(masks).astype(float)
    imgs_train, imgs_test, masks_train, masks_test = train_test_split(
        imgs, masks, test_size=0.2, shuffle=False
    )
    return imgs_train, imgs_test, masks_train, masks_test

def show_lung_mask_sbs(lung, mask):
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.imshow(lung.squeeze(), cmap = 'gray')
    a2.imshow(mask.squeeze(), cmap = 'gray')
    a1.set_title("lung")
    a2.set_title("mask")


def show_lung_mask_pred_sbs(lung, mask, pred):
    fig, (a1, a2, a3) = plt.subplots(1, 3)
    a1.imshow(lung.squeeze(), cmap = 'gray')
    a2.imshow(mask.squeeze(), cmap = 'gray')
    a3.imshow(pred.squeeze(), cmap = 'gray')
    a1.set_title("lung")
    a2.set_title("mask")
    a3.set_title("predicted mask")

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator

class ShowLearning(tf.keras.callbacks.Callback):
    def __init__(self, data, masks, name):
        self.data = data
        self.masks = masks
        self.fig_path = f"lung/figs/{name}"
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        rand_index = np.random.randint(0, len(self.data))
        rand_img = self.data[rand_index][None, :, :, :]
        mask = self.masks[rand_index][None, :, :, :]
        preds = self.model(rand_img).numpy() > 0.5
        show_lung_mask_pred_sbs(rand_img, mask, preds)
        plt.savefig(f"{self.fig_path}/epoch{epoch}.png")
        plt.close()

def show_training_image(name, epoch):
  im = plt.imread(f'lung/figs/{name}/epoch{epoch}.png')
  plt.imshow(im)

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator
