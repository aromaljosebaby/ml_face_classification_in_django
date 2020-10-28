from django.db import models

# Create your models here.


class UserImage(models.Model):
    img=models.ImageField(null=False,blank=False)
