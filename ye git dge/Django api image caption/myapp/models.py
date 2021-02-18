from django.db import models


class Document(models.Model):
    image = models.FileField(upload_to='%Y_%m_%d')
