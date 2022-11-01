import os
import secrets
from PIL import Image
from flask import current_app

def save_picture(form_picture, output_size=(224, 224), save_path='static/profile_pics'):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(
        current_app.root_path, save_path, picture_fn)

    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn