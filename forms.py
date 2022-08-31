from flask_wtf import FlaskForm
from wtforms import TextAreaField, StringField
from wtforms.validators import DataRequired, Email

class ContactForm(FlaskForm):
	name = StringField('Nom')
	email = StringField('Email', validators=[DataRequired(), Email()])
	message = TextAreaField('Message', validators=[DataRequired()])
