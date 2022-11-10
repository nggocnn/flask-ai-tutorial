from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class IFGSMModelForm(FlaskForm):
    image = FileField(
        'Input image to attack',
        validators=[DataRequired(), FileAllowed(['jpg', 'jpeg', 'png'])]
    )

    label = StringField(
        'Input label to attack',
        validators=[]
    )

    epsilon = StringField(
        'Input modification epsilon to attack',
        validators=[DataRequired()]
    )

    epochs = StringField(
        'Input epochs to attack',
        validators=[DataRequired()]
    )

    submit = SubmitField('Attack')


class FGSMModelForm(FlaskForm):
    image = FileField(
        'Input image to attack',
        validators=[DataRequired(), FileAllowed(['jpg', 'jpeg', 'png'])]
    )

    label = StringField(
        'Input label to attack',
        validators=[]
    )

    epsilon = StringField(
        'Input modification epsilon to attack',
        validators=[DataRequired()]
    )

    submit = SubmitField('Attack')