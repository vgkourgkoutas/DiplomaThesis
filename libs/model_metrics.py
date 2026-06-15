"""
Μετρικές αξιολόγησης για το DFEW: WAR (σταθμισμένη ακρίβεια) και UAR (μη σταθμισμένη,
μέσος όρος ακρίβειας ανά κλάση), καθώς και ο πίνακας σύγχυσης (confusion matrix).
"""

import sklearn

def get_WAR(trues_te, pres_te):
    """WAR: συνολική ακρίβεια, δηλαδή το ποσοστό των σωστών προβλέψεων σε ολόκληρο το test set."""
    WAR  = sklearn.metrics.accuracy_score(trues_te, pres_te)
    return WAR

def get_UAR(trues_te, pres_te):
    """UAR: μέσος όρος της ακρίβειας κάθε κλάσης ξεχωριστά. Πιο δίκαιη μετρική όταν τα δεδομένα είναι ανισόρροπα (π.χ. λίγα δείγματα disgust)."""
    cm = sklearn.metrics.confusion_matrix(trues_te, pres_te) 
    acc_per_cls = [ cm[i,i]/sum(cm[i]) for i in range(len(cm))]
    UAR = sum(acc_per_cls)/len(acc_per_cls)
    return UAR

def get_cm(trues_te, pres_te):
    """Επιστρέφει τον πίνακα σύγχυσης (confusion matrix) των πραγματικών έναντι των προβλεπόμενων ετικετών."""
    cm = sklearn.metrics.confusion_matrix(trues_te, pres_te) 
    return cm