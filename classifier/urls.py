from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),
    path('train/', views.train_model, name='train_model'),
    path('classify/', views.classify_image, name='classify_image'),
    path('results/', views.results_view, name='results_view'),
    path('reset_uploads/', views.reset_uploads, name='reset_uploads'),  # New URL pattern
]
