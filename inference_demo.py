"""
Demo αναγνώρισης συναισθήματος με το υβριδικό μοντέλο. Δέχεται ένα βίντεο ή έναν φάκελο με
καρέ, εξάγει 16 frames, τα περνά από το εκπαιδευμένο μοντέλο και εμφανίζει το Top-3 αποτέλεσμα,
μαζί με οπτικοποίηση (πλέγμα 4x4 και animation). Χρησιμοποιεί single center-crop, όχι ten-crop.

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
    Η συνάρτηση διαβάζει εικόνες μέσα από έναν φάκελο (όπως η μορφή του DFEW dataset)
    και εξάγει 16 καρέ.
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
    Εφαρμόζεtαι το απαραίτητο preprocessing ώστε να ταιριάζει με τα δεδομένα εκπαίδευσης.
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


def normalize_image_size(image, target_width=600):

    """Αλλάζει το μέγεθος μιας εικόνας σε σταθερό πλάτος (διατηρώντας τις αναλογίες), ώστε τα κείμενα που μπαίνουν από πάνω να έχουν σταθερό μέγεθος."""
    
    h, w = image.shape[:2]
    ratio = target_width / float(w)
    new_h = int(h * ratio)
    interp = cv2.INTER_AREA if w > target_width else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, new_h), interpolation=interp)


def predict_emotion(input_path, model_weights_path):
    """
    Κύρια συνάρτηση. Αναγνωρίζει αν το input_path είναι βίντεο ή φάκελος.
    """
    emotion_dict = {
        0: "Ευτυχία (Happy)", 1: "Λύπη (Sad)", 2: "Ουδέτερο (Neutral)", 
        3: "Θυμός (Angry)", 4: "Έκπληξη (Surprise)", 5: "Αηδία (Disgust)", 6: "Φόβος (Fear)"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ΣΥΣΤΗΜΑ] Εκτέλεση σε υπολογιστικό περιβάλλον: {device.type.upper()}")

    print("[ΣΥΣΤΗΜΑ] Ανάλυση Εισόδου και Εξαγωγή Καρέ...")
    if os.path.isdir(input_path):
        print(f"[ΣΥΣΤΗΜΑ] Εντοπίστηκε Φάκελος. Φόρτωση καρέ από: {input_path}")
        frames = extract_frames_from_folder(input_path, required_frames=16)
    elif os.path.isfile(input_path):
        print(f"[ΣΥΣΤΗΜΑ] Εντοπίστηκε Αρχείο Βίντεο. Εξαγωγή καρέ από: {input_path}")
        frames = extract_uniform_frames(input_path, required_frames=16)
    else:
        raise ValueError(f"Η διαδρομή {input_path} δεν υπάρχει!")

    model_input = transform_frames_for_model(frames).to(device)

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
    # ΠΡΟΣΘΗΚΗ ΟΠΤΙΚΗΣ ΑΠΕΙΚΟΝΙΣΗΣ
    # ==========================================
    print("[ΣΥΣΤΗΜΑ] Δημιουργία Οπτικών Αποτελεσμάτων...")
    
    # --- 1. ΔΗΜΙΟΥΡΓΙΑ ΠΛΕΓΜΑΤΟΣ 4x4 ---
    grid_frames = []
    for idx, f in enumerate(frames):
        frame_bgr = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, (200, 200))
        cv2.putText(frame_resized, f"Frame {idx+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        grid_frames.append(frame_resized)
        
    row1 = np.hstack(grid_frames[0:4])
    row2 = np.hstack(grid_frames[4:8])
    row3 = np.hstack(grid_frames[8:12])
    row4 = np.hstack(grid_frames[12:16])
    full_grid = np.vstack([row1, row2, row3, row4])

    # --- 2. ΠΡΟΕΤΟΙΜΑΣΙΑ ΤΩΝ 16 ΚΑΡΕ ΓΙΑ ΤΟ ANIMATION (ΜΕ HEADER) ---
    animated_frames = []
    header_height = 100 # Το ύψος της μαύρης μπάρας στο πάνω μέρος

    for f in frames:
        img_bgr = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
        
        # Κανονικοποίηση της εικόνας ώστε τα γράμματα να έχουν ιδανικό μέγεθος
        img_bgr = normalize_image_size(img_bgr, target_width=600)
        
        
        img_with_header = cv2.copyMakeBorder(img_bgr, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            
        # τα Top-3 συναισθήματα
        y_offset = 30
        for text in top3_texts:
            cv2.putText(img_with_header, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30
            
        animated_frames.append(img_with_header)

    # --- 3. ΠΡΟΒΟΛΗ ΚΑΙ ΔΙΑΧΕΙΡΙΣΗ ΠΑΡΑΘΥΡΩΝ ---
    cv2.namedWindow("All 16 Extracted Frames", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DFEW Emotion Recognition - Animation", cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow("All 16 Extracted Frames", 800, 800)
    
    cv2.resizeWindow("DFEW Emotion Recognition - Animation", 600, 600 + header_height)

    cv2.imshow("All 16 Extracted Frames", full_grid)
    print("[INFO] Πάτα το γράμμα 'q' πάνω στο βίντεο για έξοδο.")
    
    while True:
        for frame in animated_frames:
            cv2.imshow("DFEW Emotion Recognition - Animation", frame)
            if cv2.waitKey(150) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return 

# === ΕΚΤΕΛΕΣΗ ===
if __name__ == "__main__":
    
    #TEST_INPUT = r"Testing_Videos\The_Dark_Knight.mkv" 

    

    # (Happy:1)
    #TEST_INPUT = r"Testing_Videos\05838"

    # (Sad:2)
    #TEST_INPUT = r"Testing_Videos\08023"

    # (Neutral:3)
    #TEST_INPUT = r"Testing_Videos\00087"

    # File from DFEW Dataset (Angry:4)
    #TEST_INPUT = r"Testing_Videos\03194"

    # (Surprise:5)
    #TEST_INPUT = r"Testing_Videos\01378"

    # (Disgust:6)
    TEST_INPUT = r"Testing_Videos\02854"

    # (Fear:7)
    #TEST_INPUT = r"Testing_Videos\12259"

    

    

    BEST_WEIGHTS = r"work_dir\hybrid_model_with_WCEL\my_hybrid_fold5_20260609_112814\pth\my_hybrid_BEST_fold5_epo008_UAR60.39_WAR61.69.pth"
    
    predict_emotion(TEST_INPUT, BEST_WEIGHTS)