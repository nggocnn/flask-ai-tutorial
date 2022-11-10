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

def deprocess_img(img):
    x = np.array(img)
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