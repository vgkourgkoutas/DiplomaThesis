"""
Demo αναγνώρισης συναισθήματος προσώπου με το υβριδικό μοντέλο (CNN-Transformer).
 
Επέκταση του inference_demo: δέχεται είτε φάκελο εικόνων DFEW (ήδη cropped) είτε
ακατέργαστο αρχείο βίντεο. Στην περίπτωση του βίντεο εντοπίζει και αποκόπτει αυτόματα
το πρόσωπο (Haar cascade) πριν την πρόβλεψη. Σε κάθε περίπτωση εξάγει 16 ομοιόμορφα
κατανεμημένα καρέ, τα προεπεξεργάζεται όπως στην εκπαίδευση (Resize 224, ImageNet
normalization) και εμφανίζει τα top-3 συναισθήματα μαζί με οπτικοποίηση των καρέ.

"""

import os
import glob
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

from libs import my_model 

def extract_uniform_frames(video_filepath, required_frames=16):
    """
    Διαβάζει ένα αρχείο βίντεο και εξάγει ομοιόμορφα κατανεμημένα καρέ.
    """
    capture_obj = cv2.VideoCapture(video_filepath)
    if not capture_obj.isOpened():
        raise ValueError(f"Αδυναμία φόρτωσης του βίντεο: {video_filepath}")

    total_video_frames = int(capture_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_video_frames >= required_frames:
        frame_indices = np.linspace(0, total_video_frames - 1, required_frames, dtype=int)
    else:
        frame_indices = np.pad(np.arange(total_video_frames), 
                               (0, required_frames - total_video_frames), 
                               'edge')

    extracted_frames = []
    current_idx = 0
    
    while True:
        success, frame = capture_obj.read()
        if not success:
            break
            
        if current_idx in frame_indices:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            extracted_frames.append(pil_image)
            
        current_idx += 1

    capture_obj.release()
    
    while len(extracted_frames) < required_frames:
        extracted_frames.append(extracted_frames[-1])
        
    return extracted_frames


def extract_frames_from_folder(folder_path, required_frames=16):
    """
    Διαβάζει εικόνες μέσα από έναν φάκελο (όπως η μορφή του DFEW dataset)
    και εξάγει/δειγματοληπτεί 16 καρέ.
    """
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.[jp][pn]g')) + 
                         glob.glob(os.path.join(folder_path, '*.[jJ][pP][eE][gG]')))
    
    if not image_files:
        raise ValueError(f"Δεν βρέθηκαν εικόνες στον φάκελο: {folder_path}")

    total_frames = len(image_files)
    
    if total_frames >= required_frames:
        frame_indices = np.linspace(0, total_frames - 1, required_frames, dtype=int)
    else:
        frame_indices = np.pad(np.arange(total_frames), 
                               (0, required_frames - total_frames), 
                               'edge')

    extracted_frames = []
    for idx in frame_indices:
        img_path = image_files[idx]
        pil_image = Image.open(img_path).convert('RGB')
        extracted_frames.append(pil_image)
        
    return extracted_frames


def transform_frames_for_model(frame_list):
    """
    Εφαρμόζει το απαραίτητο preprocessing ώστε να ταιριάζει με τα δεδομένα εκπαίδευσης.
    """
    preprocessing_pipeline = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_tensors = []
    for f in frame_list:
        processed_tensors.append(preprocessing_pipeline(f))

    video_tensor = torch.stack(processed_tensors, dim=1)
    return video_tensor.unsqueeze(0)


def normalize_image_size(image, target_width=400):
    """
    Προσαρμόζει το μέγεθος του crop για ομοιόμορφη εμφάνιση στο animation παράθυρο.
    """
    h, w = image.shape[:2]
    ratio = target_width / float(w)
    new_h = int(h * ratio)
    interp = cv2.INTER_AREA if w > target_width else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, new_h), interpolation=interp)


def predict_emotion(input_path, model_weights_path):
    """
    Κύρια συνάρτηση. Ανιχνεύει, αποκόπτει το πρόσωπο (αν δεν είναι ήδη DFEW) 
    και εκτελεί πρόβλεψη στα 16 καρέ.
    """
    emotion_dict = {
        0: "Ευτυχία (Happy)", 1: "Λύπη (Sad)", 2: "Ουδέτερο (Neutral)", 
        3: "Θυμός (Angry)", 4: "Έκπληξη (Surprise)", 5: "Αηδία (Disgust)", 6: "Φόβος (Fear)"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ΣΥΣΤΗΜΑ] Εκτέλεση σε υπολογιστικό περιβάλλον: {device.type.upper()}")

    print("[ΣΥΣΤΗΜΑ] Ανάλυση Εισόδου και Εξαγωγή Καρέ...")
    if os.path.isdir(input_path):
        frames = extract_frames_from_folder(input_path, required_frames=16)
    elif os.path.isfile(input_path):
        frames = extract_uniform_frames(input_path, required_frames=16)
    else:
        raise ValueError(f"Η διαδρομή {input_path} δεν υπάρχει!")

    # --- ΝΕΟ ΕΠΙΠΕΔΟ ΠΡΟΕΠΕΞΕΡΓΑΣΙΑΣ: FACE CROPPING ---
    cropped_frames_for_model = []
    
    if os.path.isdir(input_path):
        # Αν είναι φάκελος (DFEW), τα δεδομένα είναι ΗΔΗ cropped. 
        # Τα περνάμε αυτούσια χωρίς να τα πειράξουμε.
        print("[ΣΥΣΤΗΜΑ] Εντοπίστηκε dataset DFEW. Τα καρέ είναι ήδη cropped. Παράκαμψη ανίχνευσης.")
        cropped_frames_for_model = frames
    else:
        # Αν είναι βίντεο, γίνεται κανονικά ανίχνευση και αποκοπή.
        print("[ΣΥΣΤΗΜΑ] Εκτέλεση Τοπικού Εντοπισμού και Αποκοπής Προσώπου (Face Cropping)...")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        last_known_face = None

        for f in frames:
            img_bgr = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                crop_bgr = img_bgr[y:y+h, x:x+w]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                last_known_face = pil_crop
                cropped_frames_for_model.append(pil_crop)
            else:
                if last_known_face is not None:
                    cropped_frames_for_model.append(last_known_face)
                else:
                    cropped_frames_for_model.append(f)

    
    model_input = transform_frames_for_model(cropped_frames_for_model).to(device)

    print("[ΣΥΣΤΗΜΑ] Φόρτωση του Υβριδικού Μοντέλου...")
    model = my_model.CNN_Transformer(num_classes=7, num_frames=16)
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("[ΣΥΣΤΗΜΑ] Εκτέλεση Πρόβλεψης...")
    with torch.no_grad():
        logits = model(model_input)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        top3_probs, top3_classes = torch.topk(probabilities, 3)

    print("\n" + "="*45)
    print("           ΑΠΟΤΕΛΕΣΜΑ ΑΝΑΛΥΣΗΣ (TOP-3)")
    print("="*45)
    top3_texts = [] 
    for i in range(3):
        emo_full = emotion_dict[top3_classes[i].item()]
        emo_english = emo_full.split('(')[1].replace(')', '') 
        emo_prob = top3_probs[i].item() * 100
        
        print(f"{i+1}. {emo_full}: {emo_prob:.1f}%")
        top3_texts.append(f"{i+1}. {emo_english}: {emo_prob:.1f}%")
    print("="*45 + "\n")

    # ==========================================
    # ΟΠΤΙΚΗ ΑΠΕΙΚΟΝΙΣΗΣ ΤΩΝ ΔΕΔΟΜΕΝΩΝ
    # ==========================================
    print("[ΣΥΣΤΗΜΑ] Δημιουργία Οπτικών Αποτελεσμάτων...")
    
    # --- 1. ΔΗΜΙΟΥΡΓΙΑ ΠΛΕΓΜΑΤΟΣ 4x4 ΜΕ ΤΑ CROPPED ΠΡΟΣΩΠΑ ---
    grid_frames = []
    for idx, f_crop in enumerate(cropped_frames_for_model):
        frame_bgr = cv2.cvtColor(np.array(f_crop), cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, (200, 200))
        cv2.putText(frame_resized, f"Crop {idx+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        grid_frames.append(frame_resized)
        
    row1 = np.hstack(grid_frames[0:4])
    row2 = np.hstack(grid_frames[4:8])
    row3 = np.hstack(grid_frames[8:12])
    row4 = np.hstack(grid_frames[12:16])
    full_grid = np.vstack([row1, row2, row3, row4])

    # --- 2. ΠΡΟΕΤΟΙΜΑΣΙΑ ANIMATION ΜΕ ΤΑ CROPS ---
    animated_frames = []
    header_height = 100 

    for f_crop in cropped_frames_for_model:
        img_bgr = cv2.cvtColor(np.array(f_crop), cv2.COLOR_RGB2BGR)
        img_bgr = normalize_image_size(img_bgr, target_width=400)
        
        img_with_header = cv2.copyMakeBorder(img_bgr, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            
        y_offset = 30
        for text in top3_texts:
            cv2.putText(img_with_header, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30
            
        animated_frames.append(img_with_header)

    # --- 3. ΠΡΟΒΟΛΗ ΚΑΙ ΔΙΑΧΕΙΡΙΣΗ ΠΑΡΑΘΥΡΩΝ ---
    cv2.namedWindow("All 16 Extracted Crops", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DFEW Emotion Recognition - Cropped Animation", cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow("All 16 Extracted Crops", 800, 800)
    cv2.resizeWindow("DFEW Emotion Recognition - Cropped Animation", 400, 400 + header_height)

    cv2.imshow("All 16 Extracted Crops", full_grid)
    print("[INFO] Η αναπαραγωγή ξεκίνησε! Πάτα το γράμμα 'q' για έξοδο.")
    
    while True:
        for frame in animated_frames:
            cv2.imshow("DFEW Emotion Recognition - Cropped Animation", frame)
            if cv2.waitKey(150) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return 

# === ΕΚΤΕΛΕΣΗ ===
if __name__ == "__main__":
    
    # Αρχείο βίντεο:
    TEST_INPUT = r"Testing_Videos\600.mp4" 
    
    # Ή φάκελο DFEW:
    #TEST_INPUT = r"Testing_Videos\02854"
    
    BEST_WEIGHTS = r"work_dir\hybrid_model_with_WCEL\my_hybrid_fold5_20260609_112814\pth\my_hybrid_BEST_fold5_epo008_UAR60.39_WAR61.69.pth"
    
    predict_emotion(TEST_INPUT, BEST_WEIGHTS)