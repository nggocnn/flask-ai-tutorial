import tensorflow as tf
import numpy as np

decode_predictions = tf.keras.applications.vgg16.decode_predictions
pretrained_model = None


class PretrainedModel:
    pretrained_model = None

    @staticmethod
    def get_pretrained_model():
        if PretrainedModel.pretrained_model is None:
            pretrained_model = tf.keras.applications.vgg16.VGG16(
                include_top=True, weights='imagenet')

        return pretrained_model


def deprocess_img(processed_img):
    x = np.array(processed_img)
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Helper function to extract labels from probability vector


def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]


loss_object = tf.keras.losses.CategoricalCrossentropy()


def generate_untargeted_adversary(model, image, label, epsilon=0.01, epochs=100):

    image = tf.cast(image, tf.float32)

    adv_results = []
    adv_preds = []
    adv_img = image

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    model = PretrainedModel().get_pretrained_model()

    for i in range(epochs):

        with tf.GradientTape() as tape:
            tape.watch(adv_img)
            prediction = model(adv_img)
            loss = loss_object(label, prediction)
            gradient = tape.gradient(loss, adv_img)
            gradient_sign = tf.sign(gradient).numpy()

        adv_img = adv_img + gradient_sign * epsilon
        adv_pred = model.predict(adv_img)

        adv_preds.append(adv_pred)

        print(i, decode_predictions(preds=adv_pred, top=3))

    return adv_img, adv_preds
