"""
Κεντρική κλάση εκπαίδευσης/αξιολόγησης (Performer). Στήνει δεδομένα, μοντέλο, loss,
optimizer και scheduler, και τρέχει τον βρόχο εκπαίδευσης με test σε κάθε epoch.
Βασίζεται στον σκελετό των δημιουργών του DFEW, με δικές μου προσθήκες
(νέα μοντέλα, WCEL, επιλογή scheduler, ten-crop testing, αποθήκευση best-UAR ανά fold).

"""

import os
import time
import yaml

from libs import DFEW_Dataset
from libs import model_metrics

from libs import my_model

from Loading_models import my_models 

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import random

import sklearn.metrics
from tensorboardX import SummaryWriter

class Performer():
    def __init__(self, args):

        """Αρχικοποίηση: αποθήκευση παραμέτρων, ρύθμιση GPU, ετοιμασία δεδομένων/dataloaders, φόρτωση μοντέλων και optimizer."""

        self.args = args
        self.save_args()
        self.seed = self.load_torch_optimize()
        self.make_dataloader(loader_types=["train", "test"])
        self.load_model()
        self.load_optim()

        self.best_UAR = 0


    def save_args(self):

        """Αποθηκεύει τις παραμέτρους του πειράματος (config.yaml) σε φάκελο work_dir με περιγραφικό όνομα ανά μοντέλο/fold."""

        arg_dict = vars(self.args)
        timestamp = "{time:s}".format(time=time.strftime("%Y%m%d_%H%M%S", time.localtime()))

        
        folder_name = "{model}_fold{fold}_{time}".format(
            model=self.args.model_name,
            fold=self.args.fold_idx,
            time=timestamp
        )

        self.work_dir = os.path.join(self.args.work_dir, folder_name)

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        yaml_name = "{pth}/config.yaml".format(pth=self.work_dir)
        with open(yaml_name, "w") as f:
            yaml.dump(arg_dict, f)


    def load_torch_optimize(self):

        """Ορίζει τυχαίους σπόρους (seeds) και ενεργοποιεί τις βελτιστοποιήσεις της GPU."""

        seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if self.args.torch_optimize == True:
            torch.backends.cudnn.enabled   = True
            torch.backends.cudnn.benchmark = True

        return seed


    def make_dataloader(self, loader_types=[]):

        """Φτιάχνει τα datasets και τους dataloaders για το train και το test."""
        
        if int(self.args.gpu_id) in [7, 8]:
            data_ori_path = "/data"

        self.args.data_root = os.path.join(data_ori_path, self.args.data_root)
        data_type = self.args.data_type

        
        if "train" in loader_types:
            self.train_data       = DFEW_Dataset.DFEW_Dataset(args  = self.args,
                                                              phase = "train")
            self.train_dataloader = DataLoader(self.train_data,
                                               batch_size=self.args.batch_size, 
                                               shuffle=True,
                                               num_workers=self.args.num_workers,
                                               pin_memory=self.args.pin_memory)

        if "test" in loader_types:
            self.test_data        = DFEW_Dataset.DFEW_Dataset(args  = self.args,
                                                              phase = "test")
            self.test_dataloader  = DataLoader(self.test_data, 
                                               batch_size=self.args.batch_size, 
                                               shuffle=False,
                                               num_workers=self.args.num_workers,
                                               pin_memory=self.args.pin_memory)


    def load_model(self):

        """Επιλέγει και φορτώνει το μοντέλο με βάση το model_name, αρχικοποιεί βάρη όπου χρειάζεται και μεταφέρει το μοντέλο στην GPU."""

        # ΚΩΔΙΚΑΣ ΓΙΑ TIMESFORMER
        if self.args.model_name == "timesformer":
            self.model = my_models.TimeSformerDFEW(num_classes=self.args.num_classes, pretrained=True)
            
        # ----------------------------------------------------------------

        # --- FORMER-DFER ---
        if self.args.model_name == "former_dfer":
            self.model = my_models.FormerDFER_Wrapper(num_classes=self.args.num_classes)
        # ----------------------------------------------------------------

        # --- HYBRID MODEL ---
        if self.args.model_name == "my_hybrid":
            self.model = my_model.CNN_Transformer(num_classes=self.args.num_classes, num_frames=self.args.nframe)

        # --- VIDEOMAE ---
        if self.args.model_name == "videomae":
            self.model = my_models.VideoMAEDFEW(num_classes=self.args.num_classes, pretrained=True)

        # ----------------------------------------------------------------

        # models & training loss

        # --- R3D MODEL ---
        if self.args.model_name == "r3d_18":         
            #self.model = torchvision.models.video.resnet.r3d_18(pretrained=self.args.model_pretrain)
            self.model = torchvision.models.video.resnet.r3d_18(weights="DEFAULT" if self.args.model_pretrain else None)
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.args.num_classes)

        # --- MC3 MODEL ---
        if self.args.model_name == "mc3_18":
            print("Loading MC3_18...")
            
            #self.model = torchvision.models.video.mc3_18(pretrained=self.args.model_pretrain)
            self.model = torchvision.models.video.mc3_18(weights="DEFAULT" if self.args.model_pretrain else None)
            
            # Αλλάζουμε την τελευταία στρώση για να βγάζει 7 κλάσεις (DFEW)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.args.num_classes)

        # Αρχικοποίηση παραμέτρων του μοντέλου

        if self.args.model_init == True and self.args.model_name != "timesformer" and self.args.model_name != "videomae" and self.args.model_name != "former_dfer":
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias,   0)
            string = "Initilize: 1. Conv2d & Linear with kaiming_normal_ 2. BatchNorm2d with constant"
            self.txt_log(string)

        # data parallel
        if self.args.Flag_mGPU_blocks == True:
            device_ids_list = [int(ele) for ele in self.args.List_mGPU_blocks]
            self.model = nn.DataParallel(self.model, device_ids=device_ids_list)
        
        self.model.cuda()

        # Επιλογή Συνάρτησης Κόστους (Loss Function)
        if self.args.loss_type == "WCEL":
            train_labels = self.train_data.single_labels.numpy().copy()
            
            if self.args.y_start_from_zero == False:
                train_labels = train_labels - 1
                
            class_counts = np.bincount(train_labels, minlength=self.args.num_classes)
            total_samples = len(train_labels)
            
            class_counts = np.where(class_counts == 0, 1, class_counts)
            
            weights_list = total_samples / (self.args.num_classes * class_counts)
            class_weights = torch.FloatTensor(weights_list).cuda()
            
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
            self.txt_log(f">>> [INFO] Χρήση Weighted CEL (WCEL) - Δυναμικά Βάρη Fold {self.args.fold_idx}: {np.round(weights_list, 3)}")
            
        elif self.args.loss_type == "CEL":
            self.loss = nn.CrossEntropyLoss()
            self.txt_log(f">>> [INFO] Χρήση απλής CEL (Χωρίς Στάθμιση Βαρών)")
            
            
        else:
            raise ValueError(f">>> [ΣΦΑΛΜΑ] Μη αναγνωρίσιμος τύπος Loss: {self.args.loss_type}")

        self.loss.cuda()

    
    def load_optim(self):

        """Φορτώνει τον optimizer και τον scheduler του πειράματος."""

        params = self.model.parameters()
        
        # Optimizer Type
        if self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(params,
                                        lr=self.args.lr_init)
        if self.args.optimizer == "SGD":
            self.optimizer = optim.SGD(params,
                                   lr=self.args.lr_init,
                                   momentum=0.9,
                                   weight_decay=self.args.weight_decay) 
        
        #if self.args.lr_strategy == True:
        #    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,gamma=0.9)

        # 2. SCHEDULER SELECTION logic
        if self.args.lr_strategy:
            if self.args.scheduler == "StepLR":
                # Περίπτωση 1: Former-DFER
                print(f"Scheduler: StepLR enabled (Step={self.args.step_size}, Gamma={self.args.lr_gamma})")
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                 step_size=self.args.step_size, 
                                                                 gamma=self.args.lr_gamma)
            
            elif self.args.scheduler == "ExponentialLR":
                # Περίπτωση 2: Τα παλιά μοντέλα
                print(f"Scheduler: ExponentialLR enabled (Gamma={self.args.lr_gamma})")
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 
                                                                        gamma=self.args.lr_gamma)
            else:
                # Default fallback
                print("Unknown scheduler! Defaulting to ExponentialLR (gamma=0.9)")
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)



    def txt_log(self, string):

        """Καταγράφει μηνύματα στην οθόνη και στο αρχείο log.txt κάθε μοντέλου."""

        localtime = time.asctime(time.localtime(time.time()))
        string    = "[{localtime}] {string}".format(localtime=localtime, string=string)
        if self.args.txt_log_toScreen == True:
            print(string)
        if self.args.txt_log == True:
            with open("{}/log.txt".format(self.work_dir),"a") as f:
                print(string, file=f)



    def start_train_epochTest(self):

        """Εκπαιδεύει το μοντέλο και το αξιολογεί (test) σε κάθε epoch."""

        # --- ΠΡΟΣΘΗΚΗ: ΚΑΤΑΓΡΑΦΗ ΒΑΣΙΚΩΝ ΡΥΘΜΙΣΕΩΝ ΣΤΟ LOG.TXT ---
        self.txt_log("\n" + "="*55)
        self.txt_log("--- ΒΑΣΙΚΕΣ ΡΥΘΜΙΣΕΙΣ ΠΕΙΡΑΜΑΤΟΣ ---")
        self.txt_log(f"Μοντέλο    : {self.args.model_name}")
        self.txt_log(f"Optimizer  : {self.args.optimizer} (LR: {self.args.lr_init})")
        self.txt_log(f"Scheduler  : {getattr(self.args, 'scheduler', 'ExponentialLR')}")
        
        # Έλεγχος για το είδος του Testing
        if getattr(self.args, 'test_ten_crop', True):
            self.txt_log("Αξιολόγηση : TEN-CROP (10 Crops)")
        else:
            self.txt_log("Αξιολόγηση : SINGLE CENTER-CROP (1 Crop)")
        self.txt_log("="*55 + "\n")
        # ---------------------------------------------------------


        if self.args.tensorboard == True:
            cmt    = "__fold#{fold}_lr#{lr}_bt#{bt}".format(fold = self.args.fold_idx,
                                                            lr   = self.args.lr_init,
                                                            bt   = self.args.batch_size)
            writer = SummaryWriter(comment=cmt)

        WAR_TEST_MAX, UAR_TEST_MAX = 0.0, 0.0
        for epo in range(0, self.args.num_epoch):
            # Train: [1. Param]
            pres_tr, trues_tr = [], []
            running_loss_tr   = 0.0


            # Train: [2. Start] 
            self.model.train()
            for idx_tr, (X_tr, y_tr_s) in enumerate(self.train_dataloader):
                step_tr   = epo*len(self.train_dataloader) + idx_tr + 1
                if self.args.y_start_from_zero == False:
                    y_tr_s  = y_tr_s - 1

                # Μεταφορά στη GPU
                X_tr   = X_tr.type(torch.FloatTensor).cuda().requires_grad_()
                y_tr_s = y_tr_s.type(torch.LongTensor).cuda()

                # --- FIX ΓΙΑ R3D_18 (Handle Ten-Crop in Training) ---
                if X_tr.dim() == 6:
                    # Αν το dataset στέλνει 10 crops [Batch, 10, C, T, H, W]
                    # Διαλέγουμε ΤΥΧΑΙΑ ένα crop για εκπαίδευση.
                    # Αυτό λειτουργεί σαν Data Augmentation και σώζει μνήμη
                    random_crop_idx = random.randint(0, 9)
                    X_tr = X_tr[:, random_crop_idx, ...] # Γίνεται [Batch, C, T, H, W]
                # ----------------------------------------------------

                # Model and Loss
                if self.args.loss_type == "CEL" or self.args.loss_type == "WCEL":
                    out_tr  = self.model(X_tr)
                    loss_tr = self.loss(out_tr, y_tr_s)

                # Loss Backward Optimizer
                running_loss_tr += loss_tr.item() 
                loss_tr.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                _, pre_tr  = torch.max(out_tr, 1)
                pres_tr   += pre_tr.cpu().numpy().tolist()
                trues_tr  += y_tr_s.cpu().numpy().tolist()
                
                if self.args.tensorboard == True:  writer.add_scalar("loss_tr", loss_tr.data, step_tr)

            # Train: [3. Report]
            pres_tr, trues_tr = [[ele] for ele in pres_tr], [[ele] for ele in trues_tr]
            acc_tr = sklearn.metrics.accuracy_score(trues_tr, pres_tr)
            if self.args.tensorboard == True:  writer.add_scalar("train acc", acc_tr, step_tr)
            if self.args.txt_log     == True:  self.txt_log("[Train] epo:{epo}/{num_epoch}, batch_idx_tr:{idx_tr}/{len_train_dataloader}, running_loss_tr:{running_loss_tr:.2f}, train acc:{acc_tr:.2f}%".format(epo       = epo+1,
                                                                                                                                                                                                                 num_epoch = self.args.num_epoch,
                                                                                                                                                                                                                 idx_tr    = idx_tr+1,
                                                                                                                                                                                                                 len_train_dataloader = len(self.train_dataloader),
                                                                                                                                                                                                                 running_loss_tr      = running_loss_tr/len(self.train_dataloader),
                                                                                                                                                                                                                 acc_tr    = acc_tr*100)) 

            
            # Test: [1. Param]
            pres_te, trues_te = [], []

            
            # Test: [2. Start]
            self.model.eval()
            for idx_te, (X_te, y_te_s) in enumerate(self.test_dataloader):
                with torch.no_grad():
                    if self.args.y_start_from_zero == False:
                        y_te_s = y_te_s - 1

                    y_te_s = y_te_s.type(torch.LongTensor).cuda()
                    
                    # --- UNIVERSAL LOGIC (Δουλεύει και για 1 και για 10 Crops) ---
                    
                    # Έλεγχος διαστάσεων:
                    # 5 διαστάσεις = Single Crop [Batch, C, T, H, W] -> Παλιά μοντέλα
                    # 6 διαστάσεις = Ten Crop    [Batch, 10, C, T, H, W] -> Former-DFER (SOTA)
                    
                    if X_te.dim() == 5:
                        # Περίπτωση 1: Single Crop (Κλασική εκτέλεση)
                        X_te = X_te.type(torch.FloatTensor).cuda()
                        out_te = self.model(X_te)
                    
                    elif X_te.dim() == 6:
                        # Περίπτωση 2: Ten Crop (SOTA εκτέλεση)
                        bs, ncrops, c, t, h, w = X_te.size()
                        
                        # Ενώνουμε Batch και Crops -> [Batch*10, C, T, H, W]
                        input_tensor = X_te.view(-1, c, t, h, w).cuda()
                        
                        # Πέρασμα από το μοντέλο
                        outputs = self.model(input_tensor) # [Batch*10, 7]
                        
                        # Ξεχωρίζουμε τα crops και παίρνουμε μέσο όρο
                        outputs = outputs.view(bs, ncrops, -1)
                        out_te = outputs.mean(1) # [Batch, 7]
                    
                    # -------------------------------------------------------------

                    _, pre_te = torch.max(out_te, 1)
                    pres_te  += pre_te.cpu().numpy().tolist() 
                    trues_te += y_te_s.cpu().numpy().tolist()

            # Test: [3. Report]
            pres_te, trues_te = [[ele] for ele in pres_te], [[ele] for ele in trues_te]
            acc_te  = model_metrics.get_WAR(trues_te, pres_te)
            WAR_te  = acc_te
            cm      = sklearn.metrics.confusion_matrix(trues_te, pres_te)
            UAR_te  = model_metrics.get_UAR(trues_te, pres_te)

            
            
            # --- ΝΕΟΣ ΚΩΔΙΚΑΣ ΓΙΑ ΕΞΟΙΚΟΝΟΜΗΣΗ ΧΩΡΟΥ ---
            
            
            # Ελέγχουμε αν το τωρινό UAR είναι καλύτερο από το ρεκόρ μας
            if UAR_te > self.best_UAR:
                print(f"New Best Model! UAR improved from {self.best_UAR*100:.2f}% to {UAR_te*100:.2f}%")
                self.best_UAR = UAR_te  # Ενημερώνουμε το ρεκόρ

                pth_fold = os.path.join(self.work_dir, "pth")
                if not os.path.exists(pth_fold):
                    os.makedirs(pth_fold)
                
                # Ορίζουμε τα ονόματα των νέων αρχείων (Προσθήκη +1 στην εποχή και το WAR)
                pth_name = os.path.join(pth_fold,                         
                                    "{model_name}_BEST_fold{fold}_epo{epo}_UAR{UAR_te:.2f}_WAR{WAR_te:.2f}.pth".format(
                                        model_name = self.args.model_name,
                                        fold       = self.args.fold_idx,
                                        epo        = str(epo + 1).zfill(3), 
                                        UAR_te     = UAR_te*100,
                                        WAR_te     = WAR_te*100))
                
                npz_name = os.path.join(pth_fold, 
                                    "{model_name}_BEST_fold{fold_idx}_epo{epo}_UAR{UAR_te:.2f}_WAR{WAR_te:.2f}.npz".format(
                                        model_name = self.args.model_name, 
                                        fold_idx   = self.args.fold_idx, 
                                        epo        = str(epo + 1).zfill(3), 
                                        UAR_te     = UAR_te*100,
                                        WAR_te     = WAR_te*100))

                # --- ΝΕΑ ΛΟΓΙΚΗ: Διαγραφή των προηγούμενων καλύτερων αρχείων ---
                if hasattr(self, 'current_best_pth') and os.path.exists(self.current_best_pth):
                    os.remove(self.current_best_pth)
                if hasattr(self, 'current_best_npz') and os.path.exists(self.current_best_npz):
                    os.remove(self.current_best_npz)

                # Αποθήκευση των νέων βαρών
                torch.save({
                    'seed'                 : self.seed,
                    'epo'                  : epo,
                    'model_state_dict'     : self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    },
                    pth_name)

                # Αποθήκευση των νέων προβλέψεων
                np.savez(npz_name, pres_te=pres_te, trues_te=trues_te)

                # Αποθήκευση των μονοπατιών στη μνήμη για να τα διαγράψει την επόμενη φορά
                self.current_best_pth = pth_name
                self.current_best_npz = npz_name
            
            else:
                # Αν δεν υπάρχει καλύτερο, δεν αποθηκεύουμε.
                pass

            
            if self.args.tensorboard == True:  writer.add_scalar("test acc", UAR_te, step_tr)
            if self.args.tensorboard == True:  writer.add_scalar("max test UAR", UAR_TEST_MAX, step_tr)
            if self.args.txt_log     == True:  self.txt_log("[Test] epo:{epo}/{num_epoch}, batch_idx_tr:{idx_tr}/{len_train_dataloader}, WAR_te:{WAR_te:.2f}%, UAR_te:{UAR_te:.2f}%".format(epo          = epo+1,
                                                                                                                                                                                            num_epoch    = self.args.num_epoch,     
                                                                                                                                                                                            idx_tr       = idx_tr+1,
                                                                                                                                                                                            len_train_dataloader = len(self.train_dataloader),
                                                                                                                                                                                            UAR_te       = UAR_te*100,
                                                                                                                                                                                            WAR_te       = WAR_te*100))
