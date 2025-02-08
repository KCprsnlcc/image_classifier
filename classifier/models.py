# classifier/models.py
from django.db import models

def user_directory_path(instance, filename):
    # Files will be uploaded to MEDIA_ROOT/images/<label>/<filename>
    return f'images/{instance.label}/{filename}'

class ImageUpload(models.Model):
    image = models.ImageField(upload_to=user_directory_path)
    label = models.CharField(max_length=100, blank=True)  # Person name

    def __str__(self):
        return self.label
