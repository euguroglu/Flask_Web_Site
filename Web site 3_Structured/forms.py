from flask_wtf import FlaskForm
from wtforms import StringField,IntegerField,SubmitField

class AddForm(FlaskForm):

    name = StringField('Name of Puppy:')
    submit = SubmitField('Add Puppy')

class DelForm(FlaskForm):

    id = IntegerField("Id Number of Puppy to Remove:")
    submit = SubmitField("Remove Puppy")

class Add2Form(FlaskForm):

    name = StringField('Name of Owner')
    pup_id = IntegerField("Id number of puppy to adopt:")
    submit = SubmitField('Add Owner')
