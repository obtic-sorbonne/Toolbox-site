from flask_wtf import FlaskForm
from wtforms import TextAreaField, StringField, SubmitField
from wtforms.validators import DataRequired, Email

class ContactForm(FlaskForm):
	name = StringField('Nom')
	email = StringField('Email', validators=[DataRequired(), Email()])
	message = TextAreaField('Message', validators=[DataRequired()])

class SearchForm(FlaskForm):
    searchbox = StringField("Recherche",validators=[DataRequired()])
    #submit = SubmitField("OK")
