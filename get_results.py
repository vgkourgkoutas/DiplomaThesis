#Εκτέλεση: python get_results.py --dir work_dir/(όνομα μοντέλου)
#coding: utf-8

"""
Βοηθητικό script: διαβάζει τα log.txt ενός μοντέλου, βρίσκει το best epoch ανά fold
(με κριτήριο το UAR) και τυπώνει έναν πίνακα με UAR/WAR ανά fold μαζί με τον μέσο όρο των 5 folds.
"""

import os
import re
import glob
import argparse

def get_best_results(model_dir):
    pattern = r"\[Test\] epo:(\d+)/\d+, .*? WAR_te:([\d.]+)%, UAR_te:([\d.]+)%"
    search_path = os.path.join(model_dir, "*", "log.txt")
    log_files = sorted(glob.glob(search_path))
    
    if not log_files:
        print(f"Δεν βρέθηκαν αρχεία log.txt στον φάκελο: {model_dir}")
        return
        
    best_per_fold = {}
    
    for filepath in log_files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        matches = re.findall(pattern, content)
        if not matches:
            continue
            
        best_uar = -1.0
        best_war = -1.0
        best_epoch = -1
        
        for epoch_str, war_str, uar_str in matches:
            uar = float(uar_str)
            war = float(war_str)
            epoch = int(epoch_str)
            
            if uar > best_uar:
                best_uar = uar
                best_war = war
                best_epoch = epoch
                
        folder_name = os.path.basename(os.path.dirname(filepath))
        
        if folder_name not in best_per_fold or best_uar > best_per_fold[folder_name]['uar']:
            best_per_fold[folder_name] = {'epoch': best_epoch, 'uar': best_uar, 'war': best_war}

    if not best_per_fold:
        print("Δεν βρέθηκαν έγκυρα αποτελέσματα στα logs.")
        return

    # Εκτύπωση του γραφήματος
    print("\n" + "="*85)
    print(f"{'Φάκελος (Καλύτερο Run ανά Fold)':<45} | {'Best Epoch':<10} | {'UAR (%)':<8} | {'WAR (%)':<8}")
    print("-" * 85)
    
    sum_uar = 0
    sum_war = 0
    count = 0
    
    for folder, data in sorted(best_per_fold.items()):
        print(f"{folder:<45} | {data['epoch']:<10} | {data['uar']:<8.2f} | {data['war']:<8.2f}")
        sum_uar += data['uar']
        sum_war += data['war']
        count += 1
        
    print("-" * 85)
    avg_uar = sum_uar / count
    avg_war = sum_war / count
    print(f"{'ΜΕΣΟΣ ΟΡΟΣ (' + str(count) + ' Folds)':<45} | {'-':<10} | {avg_uar:<8.2f} | {avg_war:<8.2f}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Υπολογισμός μέσου όρου αποτελεσμάτων ανά φάκελο")
    parser.add_argument("--dir", type=str, required=True, help="Η διαδρομή προς τον φάκελο του μοντέλου (π.χ. ./work_dir/mc3_18_v2)")
    
    args = parser.parse_args()
    
    get_best_results(args.dir)