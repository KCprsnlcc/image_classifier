# classifier/views.py
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from django.shortcuts import render, redirect
from django.conf import settings
from .models import ImageUpload
from .forms import ImageUploadForm, MultiImageUploadForm
from mtcnn.mtcnn import MTCNN
from django.db.models import Count
from PIL import Image as PILImage  # For image validation

def load_data():
    """
    Load and preprocess images stored under MEDIA_ROOT/images/<label>/.
    """
    image_data = []
    labels = []
    dataset_path = os.path.join(settings.MEDIA_ROOT, 'images')
    
    if not os.path.exists(dataset_path):
        return np.array(image_data), np.array(labels), []
    
    class_names = sorted(os.listdir(dataset_path))
    
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (128, 128))
                    image_data.append(img)
                    labels.append(idx)
                except Exception:
                    continue

    if image_data:
        image_data = np.array(image_data) / 255.0
    else:
        image_data = np.array(image_data)
    labels = np.array(labels)
    return image_data, labels, class_names

def create_model(num_classes):
    """
    Create a simple CNN model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def upload_image(request):
    """
    Handle image uploads using django-multiupload.
    This view supports multiple file uploads and validates that each file is a proper image.
    """
    if request.method == 'POST':
        form = MultiImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            images = form.cleaned_data['images']
            label = form.cleaned_data['label']
            if not images:
                return render(request, 'classifier/error.html', {
                    'message': 'No files were selected. Please choose at least one image.'
                })
            for img in images:
                try:
                    # Validate the image using Pillow.
                    pil_image = PILImage.open(img)
                    pil_image.verify()  # Raises an exception if the file is not a valid image.
                except Exception:
                    return render(request, 'classifier/error.html', {
                        'message': f'The file {img.name} is not a valid image.'
                    })
                # Reset file pointer after verify() because it moves to the end of the file.
                img.seek(0)
                ImageUpload.objects.create(image=img, label=label)
            return redirect('train_model')
        else:
            return render(request, 'classifier/upload.html', {'form': form})
    else:
        form = MultiImageUploadForm()
    return render(request, 'classifier/upload.html', {'form': form})

def train_model(request):
    """
    Train the CNN model. Users can reset, continue, or delete the trained model.
    Training is allowed if at least 1 class is available.
    """
    image_data, labels, class_names = load_data()
    if len(class_names) < 1:
        return render(request, 'classifier/error.html', {'message': 'Need at least 1 class to train.'})
    
    model_path = os.path.join(settings.BASE_DIR, 'classifier_model.h5')
    if request.method == 'POST':
        action = request.POST.get('action')
        # New branch for deleting the trained model.
        if action == 'delete':
            if os.path.exists(model_path):
                os.remove(model_path)
                message = "Trained model deleted successfully."
            else:
                message = "No trained model found to delete."
            return render(request, 'classifier/train.html', {'message': message, 'model_exists': os.path.exists(model_path)})
        try:
            if action == 'reset':
                if os.path.exists(model_path):
                    os.remove(model_path)
                model = create_model(len(class_names))
            elif action == 'continue':
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                else:
                    model = create_model(len(class_names))
            else:
                model = create_model(len(class_names))
            
            model.fit(image_data, labels, epochs=10, batch_size=32, validation_split=0.2)
            model.save(model_path)
            message = 'Model trained successfully!'
        except Exception as e:
            return render(request, 'classifier/error.html', {'message': f'Error during training: {str(e)}'})
        return render(request, 'classifier/train.html', {'message': message, 'model_exists': os.path.exists(model_path)})
    else:
        return render(request, 'classifier/train.html', {'model_exists': os.path.exists(model_path)})

def classify_image(request):
    """
    Classify an uploaded image by detecting a face with MTCNN and then
    predicting the class using the trained model. The cropped face is saved as
    a temporary file so it can be displayed on the results page.
    """
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Save the uploaded image as a temporary file.
            img_file = request.FILES['image']
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp.jpg')
            with open(temp_path, 'wb+') as destination:
                for chunk in img_file.chunks():
                    destination.write(chunk)
            
            # Load the trained model.
            model_path = os.path.join(settings.BASE_DIR, 'classifier_model.h5')
            if not os.path.exists(model_path):
                return render(request, 'classifier/error.html', {
                    'message': 'Model not found. Please train the model first.'
                })
            model = tf.keras.models.load_model(model_path)
            
            # Read the saved temporary image.
            img_cv = cv2.imread(temp_path)
            if img_cv is None:
                raise Exception("Could not read the uploaded image.")
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Detect faces using MTCNN.
            detector = MTCNN()
            faces = detector.detect_faces(img_rgb)
            if not faces:
                return render(request, 'classifier/error.html', {
                    'message': 'No face detected in the image.'
                })
            
            # Crop the first detected face.
            face = faces[0]
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cropped_face = img_rgb[y:y+h, x:x+w]
            cropped_face = cv2.resize(cropped_face, (128, 128))
            
            # Prepare the face for prediction.
            face_array = cropped_face.astype('float32') / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            prediction = model.predict(face_array)
            percentages = prediction[0] * 100
            
            # Determine the class names.
            images_dir = os.path.join(settings.MEDIA_ROOT, 'images')
            if os.path.exists(images_dir):
                class_names = sorted(os.listdir(images_dir))
            else:
                class_names = []
            results = []
            for i, perc in enumerate(percentages):
                cls_name = class_names[i] if i < len(class_names) else f'Class {i}'
                results.append({'class': cls_name, 'percentage': round(perc, 2)})
            results = sorted(results, key=lambda x: x['percentage'], reverse=True)
            
            # Save the cropped face as a temporary file (convert back to BGR for saving).
            cropped_face_path = os.path.join(settings.MEDIA_ROOT, 'temp_face.jpg')
            cv2.imwrite(cropped_face_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
            
            # Build the URL for the temporary cropped face image.
            image_url = settings.MEDIA_URL + 'temp_face.jpg'
            
            return render(request, 'classifier/classify.html', {
                'results': results,
                'image_url': image_url
            })
        except Exception as e:
            return render(request, 'classifier/error.html', {
                'message': f'Error during classification: {str(e)}'
            })
    return render(request, 'classifier/classify_form.html')


def results_view(request):
    """
    Show statistics about uploaded images and the model's existence.
    """
    stats = ImageUpload.objects.values('label').annotate(total=Count('id'))
    model_path = os.path.join(settings.BASE_DIR, 'classifier_model.h5')
    model_exists = os.path.exists(model_path)
    return render(request, 'classifier/results.html', {'stats': stats, 'model_exists': model_exists})

def reset_uploads(request):
    """
    Deletes all uploaded images (both the files on disk, if they exist, and the database records).
    If a record does not have an associated file, it will simply be deleted from the database.
    """
    if request.method == 'POST':
        uploads = ImageUpload.objects.all()
        for upload in uploads:
            # Check if the 'image' field has a file associated with it.
            if upload.image and upload.image.name:
                try:
                    file_path = upload.image.path  # This can raise ValueError if no file is set.
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    # Log the error if necessary and continue.
                    print(f"Error deleting file for {upload.image.name}: {e}")
        # Delete all ImageUpload records from the database.
        ImageUpload.objects.all().delete()
    return redirect('results_view')
