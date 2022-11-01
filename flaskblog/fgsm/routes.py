from flask import Blueprint
import os
import secrets
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import render_template, url_for, request, current_app
from flaskblog.fgsm.forms import FGSMModelForm
from flask_login import login_required
from flaskblog.fgsm.fgsm import deprocess_img, get_imagenet_label, generate_untargeted_adversary, PretrainedModel
from flaskblog.users.utils import save_picture


fgsm = Blueprint('fgsm', __name__)


@fgsm.route("/attack", methods=['GET', 'POST'])
@login_required
def attack():

    pretrained_model = PretrainedModel().get_pretrained_model()

    form = FGSMModelForm()

    image = None
    image_class = None
    class_confidence = None
    input_image_file = None
    input_label = None
    epsilon = None
    epochs = None

    output_image_file = None
    adv_class = None
    adv_confidence = None

    if request.method == 'POST' and form.validate_on_submit():
        image_fn = save_picture(form.image.data, output_size=(
            224, 224), save_path='static/input_images')

        i = Image.open(form.image.data)
        image = np.array(i)
        print(np.max(image))
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        preprocessed_image = tf.keras.applications.vgg16.preprocess_input(
            image)
        preprocessed_image = preprocessed_image[None, ...]
        print(np.max(preprocessed_image))
        image_probs = pretrained_model.predict(preprocessed_image)
        _, image_class, class_confidence = get_imagenet_label(image_probs)

        input_label = int(form.label.data)
        epsilon = float(form.epsilon.data)
        epochs = int(form.epochs.data)

        label = tf.one_hot(input_label, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))

        adv, adv_preds = generate_untargeted_adversary(
            pretrained_model, preprocessed_image, label, epsilon, epochs)

        _, adv__class, adv__confidence = get_imagenet_label(adv_preds[-1])

        decoded_adv = deprocess_img(adv)
        print(np.max(decoded_adv))
        image = tf.cast(decoded_adv, tf.float32)
        image = tf.image.resize(image, (224, 224))
        preprocessed_image = tf.keras.applications.vgg16.preprocess_input(
            image)
        preprocessed_image = preprocessed_image[None, ...]
        image_probs = pretrained_model.predict(preprocessed_image)
        _, adv_class, adv_confidence = get_imagenet_label(image_probs)
        print(adv_class, adv_confidence)

        random_hex = secrets.token_hex(8)
        _, f_ext = os.path.splitext(form.image.data.filename)
        adv_fn = random_hex + f_ext
        adv_path = os.path.join(current_app.root_path, 'static/output_images', adv_fn)
        adv_im = Image.fromarray(decoded_adv)
        adv_im.save(adv_path)

        input_image_file = url_for(
            'static', filename='input_images/' + image_fn
        )

        output_image_file = url_for(
            'static', filename='output_images/' + adv_fn
        )

    return render_template(
        'fgsm.html', title='FGSM',
        form=form,
        input_image_file=input_image_file, label=input_label, epsilon=epsilon, epochs=epochs,
        origin_class=image_class, origin_conf=class_confidence,
        adv_file=output_image_file, adv_class=adv_class, adv_conf=adv_confidence,
    )
