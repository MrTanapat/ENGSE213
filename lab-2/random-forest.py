import random
import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# Configuration
# ==========================================
IMG_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = None  # None = Load all images

# ==========================================
# 1. Dataset Preparation
# ==========================================
print("--- Downloading Dataset ---")
try:
    dataset_path = kagglehub.dataset_download(
        "muhammadirvanarfirza/decorative-plant-image-dataset")
    print(f"Dataset Path: {dataset_path}")
except Exception as e:
    print(f"Error downloading: {e}")
    exit()


def find_image_folder(start_path):
    """Recursively find the directory containing image files."""
    for root, dirs, files in os.walk(start_path):
        if len(files) > 0:
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                return os.path.dirname(root)
    return start_path


def load_data(root_path):
    """Load images, resize, flatten, and assign labels."""
    images = []
    labels = []
    class_names = []

    target_path = find_image_folder(root_path)

    if os.path.exists(target_path):
        folder_list = sorted(os.listdir(target_path))

        for folder in folder_list:
            folder_path = os.path.join(target_path, folder)

            if os.path.isdir(folder_path) and not folder.startswith('.'):
                files_inside = os.listdir(folder_path)
                image_files = [f for f in files_inside if f.lower().endswith(
                    ('.jpg', '.png', '.jpeg'))]

                if len(image_files) > 0:
                    class_names.append(folder)
                    label_index = len(class_names) - 1
                    count = 0

                    for img_file in image_files:
                        if MAX_IMAGES_PER_CLASS is not None and count >= MAX_IMAGES_PER_CLASS:
                            break

                        try:
                            img_path = os.path.join(folder_path, img_file)
                            with Image.open(img_path) as img:
                                # Convert to Grayscale
                                img = img.convert('L')
                                img = img.resize(IMG_SIZE)   # Resize
                                # Convert to NumPy Array
                                img_array = np.array(img)

                                images.append(img_array.flatten())
                                labels.append(label_index)
                                count += 1
                        except Exception:
                            pass

                    print(f"Loaded class '{folder}': {count} images")

    return np.array(images), np.array(labels), class_names


# Load and process data
print("\n--- Processing Images ---")
X, y, class_names = load_data(dataset_path)

if len(X) == 0:
    print("Error: No images found.")
    exit()

print(f"Total images: {len(X)}")
print(f"Feature shape: {X.shape}")

# ==========================================
# 2. Model Training (Random Forest)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("\n--- Training Random Forest Model ---")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("Training Complete!")

# ==========================================
# 3. Evaluation & Visualization
# ==========================================
print("\n--- Generating Report ---")

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

# Save report to text file
result_text = f"""
========================================
Training Report: Random Forest Classifier
========================================
Image Size: {IMG_SIZE}
Total Images: {len(X)}
Training Set: {len(X_train)}
Testing Set: {len(X_test)}
Accuracy: {acc * 100:.2f}%
----------------------------------------
Classification Report:
{report}
"""

with open("rf_report_results.txt", "w", encoding="utf-8") as f:
    f.write(result_text)
print("Report saved to 'rf_report_results.txt'")

# Visualize predictions
try:
    num_samples = 5
    sample_count = min(num_samples, len(X_test))
    indices = random.sample(range(len(X_test)), sample_count)

    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices):
        ax = plt.subplot(1, num_samples, i + 1)

        sample_img = X_test[idx].reshape(IMG_SIZE)
        pred_cls = class_names[clf.predict([X_test[idx]])[0]]
        true_cls = class_names[y_test[idx]]

        color = 'green' if pred_cls == true_cls else 'red'

        plt.imshow(sample_img, cmap='gray')
        plt.title(f"Actual: {true_cls}\nPred: {pred_cls}",
                  color=color, fontsize=10)
        plt.axis('off')

    plt.suptitle(
        f"Random Forest Predictions (Acc: {acc*100:.1f}%)", fontsize=14)
    plt.tight_layout()
    plt.savefig("rf_prediction_samples.png", dpi=300)
    print("Prediction samples saved to 'rf_prediction_samples.png'")
    plt.show()

except Exception as e:
    print(f"Error plotting samples: {e}")
