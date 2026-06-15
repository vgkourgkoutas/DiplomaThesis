#coding: utf-8

"""
Βοηθητικό script: υπολογίζει τον μέσο όρο UAR/WAR (best epoch ανά fold) για κάθε μοντέλο
από τα log.txt και φτιάχνει το συγκριτικό bar chart όλων των μοντέλων (final_comparison_full.png).
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_average_metrics(model_dir):
    pattern = r"\[Test\] epo:(\d+)/\d+, .*? WAR_te:([\d.]+)%, UAR_te:([\d.]+)%"
    search_path = os.path.join(model_dir, "*", "log.txt")
    log_files = sorted(glob.glob(search_path))
    
    if not log_files:
        return 0.0, 0.0
        
    best_per_fold = {}
    
    for filepath in log_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        matches = re.findall(pattern, content)
        if not matches:
            continue
            
        best_uar = -1.0
        best_war = -1.0
        
        for epoch_str, war_str, uar_str in matches:
            uar = float(uar_str)
            war = float(war_str)
            if uar > best_uar:
                best_uar = uar
                best_war = war
                
        folder_name = os.path.basename(os.path.dirname(filepath))
        fold_match = re.search(r'_fold(\d+)_', folder_name)
        fold_num = int(fold_match.group(1)) if fold_match else folder_name
        
        if fold_num not in best_per_fold or best_uar > best_per_fold[fold_num]['uar']:
            best_per_fold[fold_num] = {'uar': best_uar, 'war': best_war}
            
    if not best_per_fold:
        return 0.0, 0.0
        
    avg_uar = sum(item['uar'] for item in best_per_fold.values()) / len(best_per_fold)
    avg_war = sum(item['war'] for item in best_per_fold.values()) / len(best_per_fold)
    
    return avg_uar, avg_war

def generate_comparison_chart(base_dir="work_dir"):
    # Τώρα το script θα ψάχνει ΑΥΤΟΜΑΤΑ και τους φακέλους "_scratch"
    models_mapping = {
        'r3d_18_scratch': 'R3D_18 (Scratch)',
        'mc3_18_scratch': 'MC3_18 (Scratch)',
        'r3d_18_with_CEL': 'R3D_18 (Fine-Tuned)',
        'mc3_18_with_WCEL': 'MC3_18 (Fine-Tuned)',
        'timesformer_with_WCEL': 'TimeSformer',
        'videomae_with_WCEL': 'VideoMAE',
        'former_dfer_with_WCEL': 'Former-DFER',
        'hybrid_model_with_WCEL': 'Hybrid Model'
    }
    
    display_names = []
    uar_scores = []
    war_scores = []
    
    print(">>> Υπολογισμός τελικών σκορ από τους φακέλους...")
    
    for folder_name, display_name in models_mapping.items():
        model_path = os.path.join(base_dir, folder_name)
        
        if os.path.exists(model_path):
            avg_uar, avg_war = get_average_metrics(model_path)
        else:
            avg_uar, avg_war = 0.0, 0.0 
            
        display_names.append(display_name)
        uar_scores.append(avg_uar)
        war_scores.append(avg_war)
        
        print(f" - {display_name}: UAR = {avg_uar:.2f}% | WAR = {avg_war:.2f}%")

    # --- ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΟΣ ---
    x = np.arange(len(display_names))
    width = 0.35

    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    rects1 = ax.bar(x - width/2, uar_scores, width, label='UAR (%)', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, war_scores, width, label='WAR (%)', color='#ff7f0e', edgecolor='black')

    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Συνολική Σύγκριση Απόδοσης Μοντέλων (Scratch vs Fine-Tuned vs SOTA)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11, fontweight='bold', rotation=20, ha="right")
    ax.legend(fontsize=12)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 75) 

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    output_filename = 'final_comparison_full.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n✅ Το νέο γράφημα αποθηκεύτηκε ως: {output_filename}")

if __name__ == "__main__":
    generate_comparison_chart()