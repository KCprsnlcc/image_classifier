# classifier/forms.py
from django import forms
from .models import ImageUpload
from multiupload.fields import MultiFileField  # We still use the MultiFileField from django-multiupload
from django.forms.widgets import ClearableFileInput

# Define a custom widget that supports multiple file selection.
class CustomMultiFileInput(ClearableFileInput):
    allow_multiple_selected = True

class MultiImageUploadForm(forms.Form):
    images = MultiFileField(
        min_num=1,
        max_num=100000,
        max_file_size=1000 * 1024 * 1024,  # 1 GB per file
        widget=CustomMultiFileInput(attrs={
            'accept': 'image/jpeg,image/png'
        })
    )
    label = forms.CharField(max_length=100)

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = ['image', 'label']
        widgets = {
            'image': forms.ClearableFileInput(attrs={
                'accept': 'image/jpeg, image/png'
            })
        }
