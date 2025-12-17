import random
import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. Download and Prepare Data
# ==========================================

print("--- 1. Downloading Dataset from Kaggle ---")
try:
    # โหลดข้อมูล
    dataset_path = kagglehub.dataset_download(
        "muhammadirvanarfirza/decorative-plant-image-dataset")
    print(f"Original Download Path: {dataset_path}")
except Exception as e:
    print(f"Error downloading: {e}")
    exit()

IMG_SIZE = (64, 64)


def find_image_folder(start_path):
    for root, dirs, files in os.walk(start_path):
        if len(files) > 0:
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                return os.path.dirname(root)
    return start_path


def load_data(root_path):
    images = []
    labels = []
    class_names = []

    target_path = find_image_folder(root_path)
    print(f"Corrected Dataset Path: {target_path}")

    if os.path.exists(target_path):
        folder_list = sorted(os.listdir(target_path))

        for folder in folder_list:
            folder_path = os.path.join(target_path, folder)

            if os.path.isdir(folder_path) and not folder.startswith('.'):

                files_inside = os.listdir(folder_path)
                has_images = any(f.lower().endswith(('.jpg', '.png'))
                                 for f in files_inside)

                if has_images:
                    class_names.append(folder)
                    label_index = len(class_names) - 1

                    count = 0
                    for img_file in files_inside:
                        try:
                            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                                img_path = os.path.join(folder_path, img_file)

                                # Use PIL
                                with Image.open(img_path) as img:
                                    img = img.convert('L')       # Grayscale
                                    img = img.resize(IMG_SIZE)   # Resize
                                    img_array = np.array(img)    # Numpy

                                    images.append(img_array.flatten())
                                    labels.append(label_index)
                                    count += 1
                        except Exception as e:
                            pass
                    print(f" - Loaded class '{folder}': {count} images")

    return np.array(images), np.array(labels), class_names


print("\n--- 2. Processing Images ---")
X, y, class_names = load_data(dataset_path)

if len(X) == 0:
    print("Error: Still no images found.")
    print("Check if the folder structure contains .jpg or .png files.")
    exit()

print(f"Done: {len(X)} images loaded. Feature shape: {X.shape}")

# ==========================================
# 2. Split Data and Train Model
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("\n--- 3. Training Decision Tree Model ---")
clf = DecisionTreeClassifier(
    criterion='entropy', max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# ==========================================
# 3. Evaluate and Visualize
# ==========================================

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Plot Tree
print("Plotting Decision Tree...")
plt.figure(figsize=(20, 10))
plot_tree(clf,
          feature_names=[f"px_{i}" for i in range(X.shape[1])],
          class_names=class_names,
          filled=True,
          rounded=True,
          max_depth=3,)

plt.title("Decision Tree ( Ornamental Plant )")
plt.show()

# Show Sample Prediction
try:
    idx = random.randint(0, len(X_test)-1)
    sample_img = X_test[idx].reshape(IMG_SIZE)

    plt.figure(figsize=(5, 5))
    plt.imshow(sample_img, cmap='gray')
    pred_cls = class_names[clf.predict([X_test[idx]])[0]]
    true_cls = class_names[y_test[idx]]
    plt.title(f"Actual: {true_cls} | Predicted: {pred_cls}")
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Error showing sample image: {e}")
