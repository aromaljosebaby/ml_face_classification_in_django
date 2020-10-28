from django.forms import ModelForm
from .models import *

class UserImageForm(ModelForm):
    class Meta:
        model=UserImage
        fields='__all__'