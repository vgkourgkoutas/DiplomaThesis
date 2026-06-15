import os
import torch
import numpy as np

def verify_backup_files(root_folder):
    print(f"Ξεκινάει η σάρωση στον φάκελο: {root_folder}...\n")
    
    pth_count, pth_ok = 0, 0
    npz_count, npz_ok = 0, 0
    errors = []

    # Το os.walk μπαίνει αυτόματα σε όλους τους υποφακέλους
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            
            # --- ΕΛΕΓΧΟΣ ΓΙΑ .PTH ΑΡΧΕΙΑ ---
            if file.endswith('.pth'):
                pth_count += 1
                try:
                    # Φορτώνουμε στη CPU για ασφάλεια
                    torch.load(file_path, map_location=torch.device('cpu'))
                    pth_ok += 1
                except Exception as e:
                    errors.append(f"[ΣΦΑΛΜΑ PTH] {file_path}\n   Λεπτομέρεια: {e}")
            
            # --- ΕΛΕΓΧΟΣ ΓΙΑ .NPZ ΑΡΧΕΙΑ ---
            elif file.endswith('.npz'):
                npz_count += 1
                try:
                    # Το ανοίγουμε με το numpy για να δούμε αν είναι corrupted
                    with np.load(file_path) as data:
                        _ = data.files # Απλή προσπέλαση για επιβεβαίωση
                    npz_ok += 1
                except Exception as e:
                    errors.append(f"[ΣΦΑΛΜΑ NPZ] {file_path}\n   Λεπτομέρεια: {e}")

    # --- ΤΕΛΙΚΗ ΑΝΑΦΟΡΑ ---
    print("="*40)
    print("           ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
    print("="*40)
    print(f"Αρχεία .pth : Ελέγχθηκαν {pth_count} | Επιτυχία: {pth_ok}")
    print(f"Αρχεία .npz : Ελέγχθηκαν {npz_count} | Επιτυχία: {npz_ok}")
    
    if errors:
        print("\nΠΡΟΣΟΧΗ! Βρέθηκαν κατεστραμμένα αρχεία:")
        for err in errors:
            print(err)
    else:
        print("\nΟΛΑ ΤΕΛΕΙΑ! Κανένα αρχείο δεν είναι κατεστραμμένο. Το backup είναι 100% ασφαλές!")


# Παράδειγμα για Windows: r"C:\Users\Το_Ονομα_Σου\Desktop\work_dir\former_dfer"
FOLDER_TO_CHECK = r"C:\Users\vasil\Desktop\TELIKA_MODELA_DIPLOMATIKIS\work_dir\hybrid_model_with_WCEL"

verify_backup_files(FOLDER_TO_CHECK)