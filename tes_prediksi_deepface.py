import os
import pandas as pd
from deepface import DeepFace

def analisis_emo_deepface(img_path):
    try:
        analisis = DeepFace.analyze(img_path, actions=["emotion"], enforce_detection=False)
        result = analisis[0]["dominant_emotion"]
        
        # Membuat dictionary dengan key untuk dominant_emotion dan masing-masing emosi
        emotion_dict = {
            'angry': round(analisis[0]['emotion']['angry'], 4),
            'disgust': round(analisis[0]['emotion']['disgust'], 4),
            'fear': round(analisis[0]['emotion']['fear'], 4),
            'happy': round(analisis[0]['emotion']['happy'], 4),
            'sadness': round(analisis[0]['emotion']['sad'], 4),
            'surprise': round(analisis[0]['emotion']['surprise'], 4),
            'neutral': round(analisis[0]['emotion']['neutral'], 4),
            'dominant_emotion': result,
        }
        return emotion_dict
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def merge_with_suffix(dict1, dict2):
    merged = {}
    for k, v in dict1.items():
        if k in dict2:
            merged[f"{k}_before"] = v
        else:
            merged[k] = v
    for k, v in dict2.items():
        if k in dict1:
            merged[f"{k}_after"] = v
        else:
            merged[k] = v
    return merged

def analysis(file_img, label):
    try:
    
        #image_path_before = f'UJI/2_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis/{file_img}'
        #image_path_after = f'UJI/4_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis_frontal/{file_img}'
       
        image_path_before = f'UJI/3_dataset_affectnet_rafdb_seleksi_wajah_miring/{file_img}'
        image_path_after = f'UJI/5_dataset_affectnet_rafdb_seleksi_wajah_miring_frontal/{file_img}'

        file = {"file" : file_img, "gt": label}
        analysis_before = analisis_emo_deepface(image_path_before)
        analysis_after = analisis_emo_deepface(image_path_after)
        analysis_merged = merge_with_suffix(analysis_before, analysis_after)
        full_analysis = file | analysis_merged
        return full_analysis
    except Exception as e:
        return None


#dataset_path = "UJI/4_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis_frontal/"
dataset_path = "UJI/5_dataset_affectnet_rafdb_seleksi_wajah_miring_frontal/"

label_results = []

#max_images_per_label = 222 #tangan sintesis
max_images_per_label = 72 #miring

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        print(f"Memproses label: {label}")
        
        image_count = 0 
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            if os.path.isfile(image_path):
                if image_count < max_images_per_label:
                    image_ = os.path.join(label, image_name)
                    result = analysis(image_, label)
                    if result is not None:
                        #print(result)
                        image_count += 1
                        label_results.append(result)
                    else:
                        print("Gagal proses gambar")
                else:
                    break 
        

df = pd.DataFrame(label_results)

#df.to_csv('analisis_fix/analisis_frontal_hand_deepface.csv', index=False)
df.to_csv('analisis_fix/analisis_frontal_miring_deepface.csv', index=False)
print(df)