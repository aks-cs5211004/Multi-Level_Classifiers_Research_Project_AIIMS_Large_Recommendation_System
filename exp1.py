import re
import warnings
from collections import Counter, defaultdict
from pickle import dump, load
from time import time

import numpy as np
import sklearn.metrics as metrics
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.ensemble import (BalancedBaggingClassifier,
                               BalancedRandomForestClassifier)
from sklearn.base import clone as CLONE
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.exceptions import ConvergenceWarning
# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
# Models
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
# Custom functions
from src.utils import *


class EXP1():
    def __init__(self, args) -> None:
        self.data_dir = os.path.relpath(args.data_dir)
        self.results_dir = os.path.relpath(args.results_dir)
        self.models_dir = os.path.relpath(args.models_dir)
        self.config_dir = os.path.relpath(args.config_dir)
        self.drop_all = args.drop_all
        self.k_fold = args.k_fold
        self.n_jobs = args.n_jobs
        self.random_state = args.random_state
        self.overwrite_models = args.overwrite_models
        self.imputation = None
        self.classification = None
        self.pred_class_list = ["GT", "BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC", "CBC"]
        self.predict=args.predict
        self.include=args.include
        

        #Ignore this
        if self.drop_all:            
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp1")
            self.exp_result_dir = os.path.join(self.results_dir, "removed_all","exp1")
            self.exp_models_dir = os.path.join(self.models_dir, "removed_all", "exp1")
            
        #RELATIVE PATHS
        self.temp_dir = os.path.join(args.data_dir, "temp","exp1", str(self.include), self.predict.lower())
        self.exp_result_dir = os.path.join(self.results_dir, "exp1", str(self.include))
        self.exp_models_dir = os.path.join(self.models_dir, "exp1", str(self.include),self.predict.lower())
        self.result_excel=os.path.join(self.exp_result_dir,self.predict.lower()+".xlsx")
        
        if self.include==True:
            wf="_wf"
        else:
            wf=""
            
         
        
        #ASSIGNING ENCODED DATA FILES DEPENDING ON THE PREDICTION REQUIRED
        if self.predict=="Thrombolysis": 
            self.predict_column="Thrombolysis"
            self.data_csv = os.path.join(self.data_dir, "target_encoded_data_thrombolysis"+wf+".csv")
        elif self.predict=="Mechanical thrombectomy": 
            self.predict_column="Mechanical thrombectomy"  
            self.data_csv = os.path.join(self.data_dir, "target_encoded_data_mech_thromb"+wf+".csv")
        elif self.predict=="Antiplatelet":  
            self.predict_column='Antiplatelet  (Y/N)'
            self.data_csv = os.path.join(self.data_dir, "target_encoded_data_antiplatelet"+wf+".csv")
           
        elif self.predict=="Anticoagulation": 
            self.predict_column= 'Anticoagulation  (Y/N)'
            self.data_csv = os.path.join(self.data_dir, "target_encoded_data_anticoagulation"+wf+".csv")
            
        elif self.predict=="Antihypertensive":  
            self.predict_column= 'Antihypertensive  (Y/N)'
            self.data_csv = os.path.join(self.data_dir, "target_encoded_data_antihypertensive"+wf+".csv")

        elif self.predict=="Statins": 
            self.predict_column= 'Statins  (Y/N)'
            self.data_csv = os.path.join(self.data_dir, "target_encoded_data_statins"+wf+".csv")
            
        elif self.predict=="Death": 
            self.predict_column= 'Death in Hospital'
            self.data_csv = os.path.join(self.data_dir, "target_encoded_data_death"+wf+".csv")
           
        elif self.predict=="Stroke type": 
            self.predict_column='Diagnosis - stroke type - coded'
            self.data_csv = os.path.join(self.data_dir, "target_encoded_data_stroke_type"+wf+".csv")
        
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        self.drop_cols_all = os.path.join(self.config_dir, "remove_cols_all.xlsx")


        
    def models_init(self) -> None:
        """
        Initialize ML Models for Imputation and Prediction.
        """
        
        imputation_models = [
            
            AdaBoostRegressor(random_state = self.random_state)
            
            ]
        imputation = defaultdict(list)
        imputation["MODEL"].extend(imputation_models)
        imputation["ALIAS"].extend(["ADBR"])
        imputation["NAME"].extend([
           
            "AdaBoostRegressor"
            
            ])
        self.imputation = imputation

        classification_models = [
            BaggingClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            RandomForestClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            AdaBoostClassifier(random_state = self.random_state),
            GradientBoostingClassifier(random_state = self.random_state),
            BalancedRandomForestClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            BalancedBaggingClassifier(random_state = self.random_state, n_jobs=self.n_jobs),
            DecisionTreeClassifier(random_state = self.random_state), 
            CatBoostClassifier(silent = True, random_seed = self.random_state, thread_count=self.n_jobs)
            ]
        
        classification = defaultdict(list)
        classification["MODEL"].extend(classification_models)
        classification["ALIAS"].extend(["BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC", "CBC"])
        classification["NAME"].extend([
            "BaggingClassifier",
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "BalancedRandomForestClassifier",
            "BalancedBaggingClassifier",
            "DecisionTreeClassifier",
            "CatBoostClassifier"])
        self.classification = classification




    def generate_results(self) -> None:

        # variables
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        PREDICTIONS = dict()
        RESULTS = defaultdict(defaultdict)
        NAMES = []   
        cm_list = ["TP", "FN", "FP", "TN"]

        if not os.path.isdir(self.temp_dir):
            
            os.mkdir(self.temp_dir)
            print("here")

        if not os.path.isdir(self.exp_result_dir):
            os.mkdir(self.exp_result_dir)

        if not os.path.isdir(self.exp_models_dir):
            os.mkdir(self.exp_models_dir)
        
        if not os.path.isfile(self.result_excel):
            open(self.result_excel,"w+")


        # init models
        self.models_init()
        
        # read data and config
        data_df = read_data(self.data_csv)
        # config = read_data(self.config)
        
        if self.drop_all:
            drop_cols = read_data(self.drop_cols_all)
        else:
            drop_cols = read_data(self.drop_cols)

        print(data_df.shape, drop_cols.shape)

        drop_list = list(drop_cols["attribute_name"])
        data_df.drop(drop_list, axis=1, inplace=True)
        data_df.fillna(np.nan, inplace=True)

        cols_list = list(data_df.columns)
        cols_list.remove(self.predict_column)
         
        
        #CREATING DIFFERENT DATAFRAMES FOR FEATURES X AND LABELS y
        X = data_df.loc[:, cols_list]
        y = data_df.loc[:, self.predict_column].astype(int)
        print("Shape of X = {}, Shape of y = {}".format(X.shape, y.shape))
        assert len(X) == len(y), "Shape of data and labels is not same."

        
        # data split FOR IMPUTATION
        k_fold = self.k_fold
        data_splitter = RepeatedStratifiedKFold(n_splits = k_fold, n_repeats = 1, random_state = self.random_state)
        data_split = data_splitter.split(X,y)
        fold_idx = 0
        for train_idx, test_idx in data_split:
            x_train_list.append(X.iloc[train_idx,:])
            x_test_list.append(X.iloc[test_idx,:])
            y_train_list.append(y.iloc[train_idx])
            y_test_list.append(y.iloc[test_idx])
            print("# Fold = {}, Total Samples = {}, Train = {} {}, Test = {} {}"
                .format(fold_idx, len(y), len(train_idx), Counter(y[train_idx]), len(test_idx), Counter(y[test_idx])))
            fold_idx += 1



        for i in range(k_fold):
            for j in self.imputation["ALIAS"]:
                labels = y_test_list[i]
                key = str(i) + "_" + j
                PREDICTIONS[key] = np.zeros((len(labels), len(self.classification["MODEL"])+1))
                PREDICTIONS[key][:,0] = labels


        for j, base in enumerate(self.imputation["MODEL"]):
            for i, data in enumerate(zip(x_train_list, y_train_list, x_test_list, y_test_list)):

                X_train, y_train, X_test, y_test = data
        
                imputer_alias = self.imputation['ALIAS'][j]
                imputer_name = self.imputation['NAME'][j]
                
                        
                print("\nFOLD: {}, IMPUTATION: {}".format(i+1, imputer_name),imputer_name)
                
                

                imputer_dump_name = os.path.join(self.exp_models_dir, "{}_{}_imputer.pkl".format(i, imputer_alias))
                if os.path.isfile(imputer_dump_name) and not self.overwrite_models:
                    print("Loading previously saved imputation model at = {}".format(imputer_dump_name))
                    imputer = load(open(imputer_dump_name, 'rb'))
                    X_train_ = imputer.transform(X_train)
                    X_test_ = imputer.transform(X_test)
                else:
                    print("Performing fresh imputation.")
                    base_estimator = CLONE(base, safe=True)   
                    imputer = IterativeImputer(estimator=base_estimator, random_state=101, initial_strategy="mean",
                                        max_iter=20, verbose=2)
                    X_train_ = imputer.fit_transform(X_train)
                    X_test_ = imputer.transform(X_test)
                    dump(imputer, open(imputer_dump_name, 'wb'))
                    print("Model saved at = {}".format(imputer_dump_name))

                

                #save data to temp dir
                D_train = pd.DataFrame(np.hstack([X_train_, y_train.to_numpy().reshape(-1,1)]), columns=list(data_df.columns))
                D_test = pd.DataFrame(np.hstack([X_test_, y_test.to_numpy().reshape(-1,1)]), columns=list(data_df.columns))
                
                D_train.to_csv(os.path.join(self.temp_dir,"{}_{}_train.csv".format(i, imputer_alias)), index=False)
                D_test.to_csv(os.path.join(self.temp_dir,"{}_{}_test.csv".format(i, imputer_alias)), index=False)

                for k, model in enumerate(self.classification["MODEL"]):
            
                    model_alias = self.classification['ALIAS'][k]
                    model_name = self.classification['NAME'][k]
                    key = imputer_alias + "_" + model_alias

                    if key not in RESULTS:
                        RESULTS[key] = {m: 0 for m in cm_list}
                        NAMES.append(imputer_name + "_" + model_name)
                    
                    print("\nFOLD: {}, IMPUTATION: {}, CLASSIFICATION: {}, KEY = {}".format(i+1, imputer_name, model_name, key))

                    # Classifier
                    clf_dump_name = os.path.join(self.exp_models_dir, "{}_{}_clf.pkl".format(i, key))
                    if os.path.isfile(clf_dump_name) and not self.overwrite_models:
                        print("Loading previously saved classification model at = {}".format(clf_dump_name))
                        clf = load(open(clf_dump_name, 'rb'))
                        pred = clf.predict(X_test_)
                        
                    else:
                        print("Training classifier.")
                        clf = CLONE(model, safe=True)
                        clf.fit(X_train_, y_train)
                        pred = clf.predict(X_test_)
                        dump(clf, open(clf_dump_name, 'wb'))
                        print("Model saved at = {}".format(clf_dump_name))


                    
                    
                    PREDICTIONS[str(i) + "_" + imputer_alias][:,k+1] = list(pred)

                    tp, fn, fp, tn = metrics.confusion_matrix(y_test, pred).ravel()
                    
                    RESULTS[key]['TP'] += int(tp)
                    RESULTS[key]['FN'] += int(fn)
                    RESULTS[key]['FP'] += int(fp)
                    RESULTS[key]['TN'] += int(tn)

                    performance = evaluate_performance(RESULTS[key]['TP'], RESULTS[key]['FN'], RESULTS[key]['FP'], RESULTS[key]['TN'])
                    RESULTS[key].update(performance)
                    print("{}".format(RESULTS[key]))


                R = np.zeros([len(RESULTS.keys()), 12])
                for t, val in enumerate(RESULTS.items()):
                    p, performance = val
                    R[t,:] = list(performance.values())

                R = pd.DataFrame(R, index=RESULTS.keys(), columns=['TP','FN','FP','TN','WA', 'ACCU', 'SEN', 'SPE', 'GM', 'PRE', 'RECALL', 'F1'])
        
                R.loc[:,"NAME"] = NAMES
                
                
                R.to_excel(self.result_excel)
               
                print("HERE1")
                for t,val in PREDICTIONS.items():
                    if t == str(i) + "_" + imputer_alias:
                        path = os.path.join(self.temp_dir, "{}_predictions.csv".format(t))
                        val_df = pd.DataFrame(val,columns=self.pred_class_list)
                        val_df.to_csv(path, index=False)
                print("HERE")
                print("\n\n%%%%%%%%%%%%%%%%%%%%%% EXPERIMENT 1 RESULTS %%%%%%%%%%%%%%%%%%%%%%")
                print(R)
                print(self.result_excel)