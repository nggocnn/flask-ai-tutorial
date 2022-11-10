import tensorflow as tf
from flaskblog.adv_attack.utils import PretrainedModel

def generate_untargeted_fgsm(model, image, label, epsilon=0.01):

    adv_img = tf.cast(image, tf.float32)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    model = PretrainedModel().get_pretrained_model()


    with tf.GradientTape() as tape:
        tape.watch(adv_img)
        prediction = model(adv_img)
        loss = loss_object(label, prediction)
        gradient = tape.gradient(loss, adv_img)
        gradient_sign = tf.sign(gradient).numpy()

    adv_img = adv_img + gradient_sign * epsilon
    adv_pred = model.predict(adv_img)

    return adv_img, adv_pred
