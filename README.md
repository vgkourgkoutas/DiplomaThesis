# Ανίχνευση Συναισθήματος Προσώπων σε Σκηνές Σειρών/Ταινιών με Χρήση Μηχανικής Μάθησης

Προπτυχιακή διπλωματική εργασία, Πανεπιστήμιο Πατρών, Τμήμα Μηχανικών Η/Υ & Πληροφορικής.

Το αποθετήριο περιέχει τον κώδικα για τη σύγκριση αρχιτεκτονικών βαθιάς μάθησης στο πρόβλημα της δυναμικής αναγνώρισης συναισθήματος (Dynamic Facial Expression Recognition) πάνω στο dataset **DFEW**. Αξιολογούνται έξι μοντέλα (CNN 3D, Transformers, ένα benchmark της βιβλιογραφίας) και προτείνεται ένα υβριδικό μοντέλο CNN + Transformer που πετυχαίνει το καλύτερο UAR.

Η αξιολόγηση γίνεται με το πρωτόκολλο **5-fold cross-validation** του DFEW, με μετρικές **WAR** (Weighted Average Recall) και **UAR** (Unweighted Average Recall). Για κάθε fold κρατάμε το epoch με το υψηλότερο UAR.

## Περιεχόμενα

- [Αποτελέσματα](#αποτελέσματα)
- [Δομή του αποθετηρίου](#δομή-του-αποθετηρίου)
- [Απαιτήσεις και εγκατάσταση](#απαιτήσεις-και-εγκατάσταση)
- [Δεδομένα (DFEW)](#δεδομένα-dfew)
- [Εκτέλεση](#εκτέλεση)
- [Μετρικές](#μετρικές)
- [Ευχαριστίες και αναφορές](#ευχαριστίες-και-αναφορές)
- [Συγγραφέας](#συγγραφέας)

## Αποτελέσματα

Μέσοι όροι των 5 folds (best epoch ανά fold με κριτήριο το UAR):

| Μοντέλο | UAR (%) | WAR (%) |
| --- | :---: | :---: |
| R3D-18 (from scratch) | 44.65 | 53.14 |
| R3D-18 (fine-tuned, Kinetics-400) | 51.17 | 61.76 |
| MC3-18 (from scratch) | 46.74 | 55.16 |
| MC3-18 (fine-tuned, Kinetics-400) | 54.59 | 60.06 |
| TimeSformer (ImageNet + K400) | 51.14 | 60.77 |
| VideoMAE (Kinetics-400) | 53.08 | 62.62 |
| Former-DFER (benchmark) | 54.75 | 57.32 |
| **Υβριδικό (ResNet50 + Transformer)** | **58.15** | **61.53** |

Το υβριδικό μοντέλο δίνει το υψηλότερο UAR. Πρέπει να σημειωθεί ότι η σύγκριση με το Former-DFER δεν είναι ισοβαρής σε χωρητικότητα: το υβριδικό έχει περίπου 73.93M παραμέτρους έναντι 18.03M του Former-DFER (backbone ResNet50 έναντι ResNet18, ανάλυση εισόδου 224×224 έναντι 112×112). Η ανάλυση των αιτίων γίνεται στο κείμενο της εργασίας.

## Δομή του αποθετηρίου

```
DiplomaThesis/
├── main.py                          # Σημείο εκκίνησης (διαβάζει config/args, ξεκινά την εκπαίδευση)
├── config/                          # Αρχεία ρυθμίσεων ανά πείραμα (.yaml)
│
├── libs/
│   ├── Performer.py                 # Μηχανή εκπαίδευσης/αξιολόγησης (training loop, save best-UAR)
│   ├── DFEW_Dataset.py              # Φόρτωση DFEW, δειγματοληψία frames, augmentation, ten-crop
│   ├── model_metrics.py             # Μετρικές WAR, UAR, confusion matrix
│   ├── my_model.py                  # Υβριδικό μοντέλο (CNN_Transformer) — δική μου συνεισφορά
│   ├── ST_Former.py                 # Former-DFER: σύνθεση S-Former + T-Former (κώδικας δημιουργών)
│   ├── S_Former.py                  # Former-DFER: χωρικός encoder (κώδικας δημιουργών)
│   └── T_Former.py                  # Former-DFER: χρονικός encoder (κώδικας δημιουργών)
│
├── Loading_models/
│   └── my_models.py                 # Wrappers για TimeSformer, VideoMAE, Former-DFER
│
├── count_params.py                  # Μέτρηση παραμέτρων όλων των μοντέλων
├── get_results.py                   # Μέσοι όροι UAR/WAR ανά fold από τα logs
├── models_accuracy_comparison.py    # Συγκριτικό bar chart όλων των μοντέλων
├── plot_curves.py                   # Καμπύλες εκπαίδευσης (accuracy/UAR/WAR/loss) ανά fold
├── inference_demo.py                # Demo πρόβλεψης σε βίντεο ή φάκελο με καρέ
│
├── requirements.txt                 # Βιβλιοθήκες του environment
└── work_dir/                        # Έξοδοι εκπαίδευσης (logs, βάρη .pth, .npz)
```
## Απαιτήσεις και εγκατάσταση

Τα πειράματα έτρεξαν σε απομακρυσμένο μηχάνημα του πανεπιστημίου με GPU NVIDIA (Tesla V100 32GB). Απαιτείται εγκατάσταση CUDA συμβατή με την έκδοση του PyTorch.

Έκδοση Python του environment: **Python 3.10.19**

```bash
# 1. Δημιουργία και ενεργοποίηση virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 2. Εγκατάσταση βιβλιοθηκών
pip install -r requirements.txt
```

Οι βασικές εξαρτήσεις είναι: PyTorch και torchvision (μοντέλα και εκπαίδευση), Transformers της Hugging Face (TimeSformer, VideoMAE), scikit-learn (μετρικές), OpenCV (demo), Matplotlib (γραφικές), καθώς και NumPy, pandas, Pillow, einops, tensorboardX και PyYAML. Οι ακριβείς εκδόσεις βρίσκονται στο `requirements.txt`.

## Δεδομένα (DFEW)

Το DFEW **δεν περιλαμβάνεται** στο αποθετήριο. Η πρόσβαση γίνεται κατόπιν αίτησης στους δημιουργούς, μέσω της επίσημης σελίδας: https://dfew-dataset.github.io/

Η εργασία χρησιμοποιεί τα single-labeled δείγματα. Μετά τη λήψη, τα δεδομένα πρέπει να ακολουθούν τη δομή που ορίζουν οι δημιουργοί:

```
.../DFEW/data_affine/single_label/
├── data/
│   ├── 00001/
│   │   ├── 00001_00001.jpg
│   │   └── ...
│   └── 16372/
│       └── ...
└── label/
    ├── single_trainset_1.csv  ...  single_trainset_5.csv
    └── single_testset_1.csv   ...  single_testset_5.csv
```

Αντιστοίχιση ετικετών: `1: Happy, 2: Sad, 3: Neutral, 4: Angry, 5: Surprise, 6: Disgust, 7: Fear`.

Η διαδρομή προς τα δεδομένα ορίζεται στο `main.py` (παράμετρος `--data_root`) και στη μέθοδο `make_dataloader` του `libs/Performer.py`. Στο μηχάνημα του πανεπιστημίου η ρίζα ήταν `/data` και ενεργοποιούνταν για συγκεκριμένα `gpu_id`. Σε διαφορετικό περιβάλλον χρειάζεται προσαρμογή αυτών των διαδρομών.

## Εκτέλεση

### Εκπαίδευση

Κάθε πείραμα ορίζεται από ένα αρχείο `.yaml` στον φάκελο `config/` (μοντέλο, optimizer, scheduler, loss, augmentation κ.λπ.). Η εκπαίδευση τρέχει ανά fold:

```bash
python main.py --config config/<πείραμα>.yaml --fold_idx 1
# επανάληψη για --fold_idx 2 ... 5
```

Διαθέσιμα μοντέλα (παράμετρος `model_name`): `r3d_18`, `mc3_18`, `timesformer`, `videomae`, `former_dfer`, `my_hybrid`.

### Συγκέντρωση αποτελεσμάτων

```bash
python get_results.py --dir work_dir/<φάκελος_μοντέλου>
```

Τυπώνει το best epoch, το UAR και το WAR ανά fold, μαζί με τον μέσο όρο των 5 folds.

### Γραφικές παραστάσεις

```bash
# Συγκριτικό γράφημα όλων των μοντέλων -> final_comparison_full.png
python models_accuracy_comparison.py

# Καμπύλες εκπαίδευσης ανά fold (PNG + PDF)
python plot_curves.py --model <φάκελος_μοντέλου>
```

### Μέτρηση παραμέτρων

```bash
python count_params.py
```

### Demo

Τα δείγματα βίντεο παρέχονται συμπιεσμένα στο `Testing_Videos.zip` (για λόγους χωρητικότητας). Πριν την εκτέλεση, γίνεται αποσυμπίεση ώστε να δημιουργηθεί ο φάκελος `Testing_Videos/`:

```bash
unzip Testing_Videos.zip
python inference_demo.py
```

Πριν την εκτέλεση, ορίζονται μέσα στο αρχείο οι μεταβλητές `TEST_INPUT` (βίντεο ή φάκελος με καρέ) και `BEST_WEIGHTS` (αρχείο βαρών `.pth`). Οι προεπιλεγμένες διαδρομές είναι σε μορφή Windows. Το demo χρησιμοποιεί single center-crop (όχι ten-crop).

Επειδή τα βάρη (`.pth`) δεν περιλαμβάνονται στο αποθετήριο, η μεταβλητή `BEST_WEIGHTS` πρέπει να δείχνει σε ένα μοντέλο που έχετε εκπαιδεύσει τοπικά.

## Μετρικές

- **WAR (Weighted Average Recall):** η συνολική ακρίβεια, δηλαδή το ποσοστό σωστών προβλέψεων σε όλο το test set.
- **UAR (Unweighted Average Recall):** ο μέσος όρος της ακρίβειας κάθε κλάσης ξεχωριστά. Είναι πιο αντιπροσωπευτική μετρική σε ανισόρροπα δεδομένα, όπου κάποια συναισθήματα (π.χ. disgust) έχουν πολύ λίγα δείγματα. Για αυτόν τον λόγο το UAR θεωρείται η κύρια μετρική αξιολόγησης.

## Ευχαριστίες και αναφορές

Η εργασία βασίζεται σε ανοιχτά δεδομένα και κώδικα. Ευχαριστώ τους δημιουργούς τους.

### DFEW dataset

Xingxun Jiang, Yuan Zong, Wenming Zheng, Chuangao Tang, Wanchuang Xia, Cheng Lu, Jiateng Liu. *DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild.* Proceedings of the 28th ACM International Conference on Multimedia (MM '20), 2020, pp. 2881–2889.

Σελίδα: https://dfew-dataset.github.io/ — Αποθετήριο: https://github.com/jiangxingxun/DFEW — arXiv: 2008.05924

```bibtex
@inproceedings{jiang2020dfew,
  title={DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild},
  author={Jiang, Xingxun and Zong, Yuan and Zheng, Wenming and Tang, Chuangao and Xia, Wanchuang and Lu, Cheng and Liu, Jiateng},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2881--2889},
  year={2020}
}
```

### Former-DFER

Ο κώδικας στα αρχεία `libs/ST_Former.py`, `libs/S_Former.py` και `libs/T_Former.py` προέρχεται από το επίσημο αποθετήριο των δημιουργών του Former-DFER.

Zengqun Zhao, Qingshan Liu. *Former-DFER: Dynamic Facial Expression Recognition Transformer.* Proceedings of the 29th ACM International Conference on Multimedia (MM '21), 2021, pp. 1553–1561.

Αποθετήριο: https://github.com/zengqunzhao/Former-DFER

```bibtex
@inproceedings{zhao2021former,
  title={Former-DFER: Dynamic Facial Expression Recognition Transformer},
  author={Zhao, Zengqun and Liu, Qingshan},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1553--1561},
  year={2021}
}
```

### Προεκπαιδευμένα μοντέλα

- **R3D-18 / MC3-18:** torchvision, προεκπαιδευμένα σε Kinetics-400. Tran et al., *A Closer Look at Spatiotemporal Convolutions for Action Recognition*, CVPR 2018.
- **TimeSformer:** Bertasius et al., *Is Space-Time Attention All You Need for Video Understanding?*, ICML 2021. Βάρη: `facebook/timesformer-base-finetuned-k400` (Hugging Face).
- **VideoMAE:** Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*, NeurIPS 2022. Βάρη: `MCG-NJU/videomae-base-finetuned-kinetics` (Hugging Face).

Τα δεδομένα DFEW και ο κώδικας τρίτων διέπονται από τις δικές τους άδειες χρήσης.

## Συγγραφέας

Βασίλης Γκουργκούτας
Πανεπιστήμιο Πατρών, Τμήμα Μηχανικών Η/Υ & Πληροφορικής
Έτος: 2026
