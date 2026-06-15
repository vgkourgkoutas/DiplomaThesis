# Εκτέλεση: python plot_curves.py --model former_dfer_with_WCEL (όνομα του μοντέλου)
#coding: utf-8

"""
Βοηθητικό script: για κάθε fold ενός μοντέλου διαβάζει το καλύτερο log.txt και σχεδιάζει τις
καμπύλες εκπαίδευσης (train accuracy, test UAR/WAR, train loss x50) ανά epoch, αποθηκεύοντας
PNG και PDF.
"""

import re
import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'figure.figsize': (10, 6)
})


COLOR_TRAIN_ACC = '#1f77b4'
COLOR_TEST_UAR  = '#e74c3c'
COLOR_TEST_WAR  = '#27ae60'
COLOR_LOSS      = '#e67e22'

def get_best_logs_per_fold(model_dir):
    """Βρίσκει το καλύτερο log.txt για κάθε Fold (αγνοώντας τα crashed runs)"""
    pattern = r"\[Test\] epo:(\d+)/\d+, .*? WAR_te:([\d.]+)%, UAR_te:([\d.]+)%"
    search_path = os.path.join(model_dir, "*", "log.txt")
    log_files = sorted(glob.glob(search_path))
    
    best_per_fold = {}
    
    for filepath in log_files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        matches = re.findall(pattern, content)
        if not matches:
            continue
            
        best_uar = -1.0
        for epoch_str, war_str, uar_str in matches:
            if float(uar_str) > best_uar:
                best_uar = float(uar_str)
                
        folder_name = os.path.basename(os.path.dirname(filepath))
        # Ψάχνει να βρει το '_foldX_' στο όνομα του φακέλου
        fold_match = re.search(r'_fold(\d+)_', folder_name)
        if fold_match:
            fold_num = int(fold_match.group(1))
        else:
            # Αν δεν το βρει με το παραπάνω regex, προσπαθεί εναλλακτικά
            fold_match_alt = re.search(r'fold_(\d+)', folder_name)
            fold_num = int(fold_match_alt.group(1)) if fold_match_alt else folder_name
        
        if fold_num not in best_per_fold or best_uar > best_per_fold[fold_num]['uar']:
            best_per_fold[fold_num] = {
                'filepath': filepath,
                'uar': best_uar
            }
            
    return best_per_fold

def plot_combined_curves_per_fold(log_file, output_image, model_name, fold_num):
    #Διαβάζει το log_file και δημιουργείται το γράφημα
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    train_pattern = r"\[Train\] epo:(\d+)/\d+, .*? running_loss_tr:([\d.]+), train acc:([\d.]+)%"
    test_pattern = r"\[Test\] epo:(\d+)/\d+, .*? WAR_te:([\d.]+)%, UAR_te:([\d.]+)%"

    train_matches = re.findall(train_pattern, content)
    test_matches = re.findall(test_pattern, content)

    train_data = {int(m[0]): {'loss': float(m[1]), 'acc': float(m[2])} for m in train_matches}
    test_data = {int(m[0]): {'war': float(m[1]), 'uar': float(m[2])} for m in test_matches}

    valid_epochs = sorted(list(set(train_data.keys()) & set(test_data.keys())))
    
    if not valid_epochs:
        print(f"  [!] Δεν βρέθηκαν επαρκή δεδομένα στο {log_file}")
        return

    epochs = valid_epochs
    train_acc = [train_data[e]['acc'] for e in valid_epochs]
    test_war = [test_data[e]['war'] for e in valid_epochs]
    test_uar = [test_data[e]['uar'] for e in valid_epochs]
    scaled_train_loss = [train_data[e]['loss'] * 50 for e in valid_epochs]

    plt.figure()
    
    # Σχεδιασμός των γραμμών με συμπαγείς γραμμές (solid) και γεωμετρικά σύμβολα (markers)
    plt.plot(epochs, train_acc, color=COLOR_TRAIN_ACC, label='Train Accuracy', alpha=0.9, linestyle='-', marker='o', markersize=6)
    plt.plot(epochs, test_uar, color=COLOR_TEST_UAR, label='Test UAR', alpha=0.9, linestyle='-', marker='s', markersize=6)
    plt.plot(epochs, test_war, color=COLOR_TEST_WAR, label='Test WAR', alpha=0.9, linestyle='-', marker='^', markersize=6)
    plt.plot(epochs, scaled_train_loss, color=COLOR_LOSS, label='Train Loss (x50)', alpha=0.7, linestyle='-', linewidth=2)

    plt.title(f'Εκπαίδευση - {model_name.upper()} (Fold {fold_num})')
    plt.xlabel('Εποχές (Epochs)')
    plt.ylabel('Ποσοστό (%) / Απώλεια')

    # Ρύθμιση αξόνων
    max_epoch = max(epochs)
    plt.xlim(1, max_epoch)
    if max_epoch <= 25:
        plt.xticks(np.arange(1, max_epoch + 1, step=2))
    elif max_epoch <= 100:
        plt.xticks(np.arange(0, max_epoch + 1, step=5))
    else:
        plt.xticks(np.arange(0, max_epoch + 1, step=10))
        
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, step=10))
    
    #plt.legend(loc='lower right')

    # Τοποθέτηση υπομνήματος έξω από το γράφημα, πάνω δεξιά
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    
    # 1. Αποθήκευση σε PNG
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    
    # 2. Αποθήκευση σε PDF (για LaTeX)
    pdf_output = output_image.replace('.png', '.pdf')
    plt.savefig(pdf_output, bbox_inches='tight')
    
    plt.close()
    print(f"  ✅ Αποθηκεύτηκε: {output_image} και {pdf_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base", type=str, default="work_dir")
    parser.add_argument("--outdir", type=str, default="plots")
    
    args = parser.parse_args()
    target_dir = os.path.join(args.base, args.model)
    
    if not os.path.exists(target_dir):
        print(f"Σφάλμα: Ο φάκελος {target_dir} δεν υπάρχει.")
        exit(1)
        
    print(f">>> Αναζήτηση καλύτερων logs για το μοντέλο: {args.model}")
    best_logs = get_best_logs_per_fold(target_dir)
    
    if not best_logs:
        print("Δεν βρέθηκαν ολοκληρωμένα runs. Ελέγξτε αν τα ονόματα των φακέλων περιέχουν τη λέξη '_fold1_' κλπ.")
        exit(1)
        
    print(f"Βρέθηκαν {len(best_logs)} Folds. Ξεκινάει η δημιουργία γραφημάτων...\n")
    
    model_plot_dir = os.path.join(args.outdir, args.model)
    os.makedirs(model_plot_dir, exist_ok=True)
    
    for fold_num in sorted(best_logs.keys()):
        log_path = best_logs[fold_num]['filepath']
        output_name = os.path.join(model_plot_dir, f"{args.model}_fold{fold_num}_plot.png")
        plot_combined_curves_per_fold(log_path, output_name, args.model, fold_num)
        
    print(f"\n🎉 Όλα τα γραφήματα αποθηκεύτηκαν στον φάκελο: {model_plot_dir}/")