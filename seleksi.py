import os
import shutil
from deepface import DeepFace

def analisis_emo(img_path):
    try:
        analisis = DeepFace.analyze(img_path, actions=["emotion"], enforce_detection=False)
        result = analisis[0]["dominant_emotion"]
        return result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Path dataset asli dan path untuk dataset yang diseleksi
dataset_path = "datasetFIX"
output_path = f"hasil_seleksi_deepface_{dataset_path}"

# Loop melalui setiap folder label
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)

    # Pastikan hanya membaca folder
    if os.path.isdir(label_path):
        print(f"Memproses label: {label}")

        # Loop melalui setiap gambar dalam subfolder
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)

            # Pastikan hanya membaca file gambar
            if os.path.isfile(image_path):
                # Analisis emosi gambar
                detected_emotion = analisis_emo(image_path)
                
                # Jika emosi yang terdeteksi sama dengan label foldernya, simpan gambar
                if detected_emotion and detected_emotion.lower() == label.lower():
                    selected_label_path = os.path.join(output_path, label)

                    # Buat folder jika belum ada
                    os.makedirs(selected_label_path, exist_ok=True)

                    # Salin gambar ke folder seleksi
                    shutil.copy(image_path, os.path.join(selected_label_path, image_name))
                    print(f"✅ {image_name} disimpan ke {selected_label_path}")
