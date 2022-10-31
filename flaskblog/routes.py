import os
import secrets
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import render_template, url_for, flash, redirect, request
from flaskblog import app, db, bcrypt
from flaskblog.forms import RegistrationForm, LoginForm, UpdateAccountForm, FGSMModelForm
from flaskblog.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
from flaskblog.fgsm import deprocess_img, get_imagenet_label, generate_untargeted_adversary, PretrainedModel

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

posts = []


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = RegistrationForm()
    
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(
            form.password.data).decode('utf-8')
        user = User(username=form.username.data,
                    email=form.email.data, password=hashed_password)

        db.session.add(user)
        db.session.commit()

        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = LoginForm()
    
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))

        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')

    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture, output_size=(224, 224), save_path='static/profile_pics'):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(
        app.root_path, save_path, picture_fn)

    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        
        current_user.username = form.username.data
        current_user.email = form.email.data
        
        db.session.commit()
        
        flash('Your account has been updated!', 'success')
        
        return redirect(url_for('account'))

    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email

    image_file = url_for(
        'static', filename='profile_pics/' + current_user.image_file
    )

    return render_template(
        'account.html', title='Account',
        image_file=image_file, form=form
    )


@app.route("/attack", methods=['GET', 'POST'])
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
        image_fn = save_picture(form.image.data, output_size=(224, 224), save_path='static/input_images')

        i = Image.open(form.image.data)
        image = np.array(i)
        print(np.max(image))
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        preprocessed_image = tf.keras.applications.vgg16.preprocess_input(image)
        preprocessed_image = preprocessed_image[None, ...]
        print(np.max(preprocessed_image))
        image_probs = pretrained_model.predict(preprocessed_image)
        _, image_class, class_confidence = get_imagenet_label(image_probs)

        input_label = int(form.label.data)
        epsilon = float(form.epsilon.data)
        epochs = int(form.epochs.data)

        label = tf.one_hot(input_label, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))

        adv, adv_preds = generate_untargeted_adversary(pretrained_model, preprocessed_image, label, epsilon, epochs)

        _, adv__class, adv__confidence = get_imagenet_label(adv_preds[-1])

        decoded_adv = deprocess_img(adv)
        print(np.max(decoded_adv))
        image = tf.cast(decoded_adv, tf.float32)
        image = tf.image.resize(image, (224, 224))
        preprocessed_image = tf.keras.applications.vgg16.preprocess_input(image)
        preprocessed_image = preprocessed_image[None, ...]
        image_probs = pretrained_model.predict(preprocessed_image)
        _, adv_class, adv_confidence = get_imagenet_label(image_probs)
        print(adv_class, adv_confidence)

        random_hex = secrets.token_hex(8)
        _, f_ext = os.path.splitext(form.image.data.filename)
        adv_fn = random_hex + f_ext
        adv_path = os.path.join(app.root_path, 'static/output_images', adv_fn)
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