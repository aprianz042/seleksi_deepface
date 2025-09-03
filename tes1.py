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


path = "FINAL/10_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis_frontal/angry/angry_0026.jpg"
x = analisis_emo_deepface(path)
print(x)