# Generated by Django 3.1.6 on 2021-02-01 15:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='image',
            field=models.FileField(upload_to='%Y_%m_%d'),
        ),
    ]