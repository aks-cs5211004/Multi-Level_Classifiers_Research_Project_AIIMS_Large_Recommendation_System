import re
import warnings
from collections import Counter, defaultdict
from pickle import dump, load
from time import time

import numpy as np
import sklearn.metrics as metrics
from anytree import Node, RenderTree, find
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


class EXP11():
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
        self.pred_class_list = ["GT", "BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC"]
        self.predict=args.predict
        self.pred_dir=args.pred_dir
        
        #Create 3 TREEs
        Tree=[ 0 for i in range(5)]
        
        Tree[0] = Node("Stroke_Type")
        thrombolysis = Node("Thrombolysis", parent=Tree[0])
        window = Node("Window_Period_Thrombolysis", parent=thrombolysis)
        agent = Node("Thrombolytic_Agent", parent=thrombolysis)
        complication = Node("Post_Thrombolytic_Complications", parent=thrombolysis)
        antiplatelet = Node("Antiplatelet", parent=Tree[0])
        antiplatelet_drug_1= Node("Antiplatelet_Type_1",antiplatelet)
        antiplatelet_drug_2= Node("Antiplatelet_Type_2",antiplatelet)
        antiplatelet_drug_3= Node("Antiplatelet_Type_3",antiplatelet)
        antiplatelet_drug_4= Node("Antiplatelet_Type_4",antiplatelet)
        antiplatelet_drug_5= Node("Antiplatelet_Type_5",antiplatelet)
        statins= Node("Statins",Tree[0])
        statins_type=Node("Statins_Types",statins)
        
        Tree[1] = Node("Mechanical_Thrombectomy")
        direct_bridging=Node("Direct_Vs_Bridging",Tree[1])
        
        Tree[2] = Node("Anticoagulation")
        anticoagulation_drug_1=Node("Anticoagulation_Type_1", Tree[2])
        anticoagulation_drug_2=Node("Anticoagulation_Type_2", Tree[2])
        anticoagulation_drug_3=Node("Anticoagulation_Type_3", Tree[2])
        anticoagulation_drug_4=Node("Anticoagulation_Type_4", Tree[2])
        anticoagulation_drug_5=Node("Anticoagulation_Type_5", Tree[2])
        anticoagulation_drug_6=Node("Anticoagulation_Type_6", Tree[2])
        
        Tree[3]=Node("Antihypertensive")
        
        Tree[4]=Node("Death")
        
        
        

        #Ignore this
        if self.drop_all:
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp11")
            self.exp_result_dir = os.path.join(self.results_dir, "removed_all","exp11")
            self.exp_models_dir = os.path.join(self.models_dir, "removed_all", "exp11")
            
        #RELATIVE PATHS
        self.temp_dir = os.path.join(args.data_dir, "temp","exp11", self.predict.lower())
        self.exp_result_dir = os.path.join(self.results_dir, "exp11")
        self.exp_models_dir = os.path.join(self.models_dir, "exp11", self.predict.lower())
        self.result_excel=os.path.join(self.exp_result_dir,self.predict.lower()+".xlsx")
        
        self.pred_excel=os.path.join(self.pred_dir,"exp11",self.predict.lower()+".xlsx")
        

        #ASSIGNING ENCODED DATA FILES DEPENDING ON THE PREDICTION REQUIRED
        list_of_sheets_target_encoded=["stroke_type.csv","thrombolysis_based_stroke.csv","mech_thromb.csv","antiplatelet_based_stroke.csv",
                               "anticoagulation.csv","antihypertensive.csv","statins_based_stroke.csv","death.csv","thromb_agent_based_throbolysis_stroke.csv",
                               "window_based_thrombolysis_stroke.csv","complications_based_thrombolysis_stroke.csv","directbridging_based_mech_thromb.csv",
                               "statins_drug_based_statins_stroke.csv","anticoagulation_1.csv","anticoagulation_2.csv",
                               "anticoagulation_3.csv","anticoagulation_4.csv","anticoagulation_5.csv",
                               "anticoagulation_6.csv",
                               "antiplatelet_1_based_stroke.csv","antiplatelet_2_based_stroke.csv","antiplatelet_3_based_stroke.csv",
                               "antiplatelet_4_based_stroke.csv","antiplatelet_5_based_stroke.csv"
                               ]
        
        list=['Diagnosis - stroke type - coded',
            'Thrombolysis',
            'Mechanical thrombectomy',
            'Antiplatelet  (Y/N)',
            'Anticoagulation  (Y/N)',
            'Antihypertensive  (Y/N)',
            'Statins  (Y/N)',
            'Death in Hospital',

            'Thrombolytic agent',
            'Window period for thrombolysis .1',
            'Post thrombolysis complication',
            'Mechanical thrombectomy- direct vs bridging',
            'Statins- drugs (help lower cholesterol levels in the blood.)  (NA)',
            'Anticoaguation- Drug_1',
            'Anticoaguation- Drug_2',
            'Anticoaguation- Drug_3',
            'Anticoaguation- Drug_4',
            'Anticoaguation- Drug_5',
            'Anticoaguation- Drug_6',
            'Antiplatelet_1',
            'Antiplatelet_2',
            'Antiplatelet_3',
            'Antiplatelet_4',
            'Antiplatelet_5'

            ]
        
        def find_column_label_data_csv_predict_csv(predict):
            
            df=pd.DataFrame()
            
            if predict=="Stroke_Type": 
                predict_column=list[0]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[0])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Thrombolysis": 
                predict_column=list[1]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[1])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Mechanical_Thrombectomy": 
                predict_column=list[2]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[2])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Antiplatelet": 
                predict_column=list[3]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[3])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Anticoagulation": 
                predict_column=list[4]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[4])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Antihypertensive": 
                predict_column=list[5]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[5])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Statins": 
                predict_column=list[6]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[6])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Death": 
                predict_column=list[7]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[7])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Thrombolytic_Agent": 
                predict_column=list[8]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[8])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Window_Period_Thrombolysis": 
                predict_column=list[9]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[9])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Post_Thrombolytic_Complications": 
                predict_column=list[10]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[10])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Direct_Vs_Bridging": 
                predict_column=list[11]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[11])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Statins_Types": 
                predict_column=list[12]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[12])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Anticoagulation_Type_1": 
                predict_column=list[13]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[13])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Anticoagulation_Type_2": 
                predict_column=list[14]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[14])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Anticoagulation_Type_3": 
                predict_column=list[15]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[15])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Anticoagulation_Type_4": 
                predict_column=list[16]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[16])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Anticoagulation_Type_5": 
                predict_column=list[17]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[17])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Anticoagulation_Type_6": 
                predict_column=list[18]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[18])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Antiplatelet_Type_1": 
                predict_column=list[19]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[19])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Antiplatelet_Type_2": 
                predict_column=list[20]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[20])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Antiplatelet_Type_3": 
                predict_column=list[21]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[21])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Antiplatelet_Type_4": 
                predict_column=list[22]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[22])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
            if predict=="Antiplatelet_Type_5": 
                predict_column=list[23]
                data_csv = os.path.join(self.data_dir, list_of_sheets_target_encoded[23])
                sheet=os.path.join(self.pred_dir,"exp11",predict.lower()+".xlsx")
                
            return(predict_column,data_csv,sheet)
                
            
        
        replace_with_pred_label=[]
        replace_with_pred_sheet=[]
        
        
        for i in range(len(Tree)):
              node  = find(Tree[i], lambda node: node.name == self.predict)
              print(node)
              if node!=None:
                  break
        
        #finding all parents of node and 
        
        def find_parents(node):
              if(node.parent):
                  replace_with_pred_label.append(find_column_label_data_csv_predict_csv(node.parent.name)[0])
                  replace_with_pred_sheet.append(find_column_label_data_csv_predict_csv(node.parent.name)[2])
                  find_parents(node.parent)
        find_parents(node)
        
        
        #Finding prediction column and data sheet
        self.predict_column= find_column_label_data_csv_predict_csv(self.predict)[0]
        self.data_csv= find_column_label_data_csv_predict_csv(self.predict)[1]
        
        print(replace_with_pred_label)
        print(replace_with_pred_sheet)
        self.df = read_data(self.data_csv)
        print(self.data_csv)
        print(self.df.shape)
        for i in range(len(replace_with_pred_label)):
              self.df.drop([replace_with_pred_label[i]], axis=1)
              print(self.df.shape)
              prev_predicted_data=read_data(replace_with_pred_sheet[i]).loc[:,replace_with_pred_label[i]]
              self.df.loc[:,replace_with_pred_label[i]]=prev_predicted_data.values.tolist()       
        
        
        print(self.df.shape)
        
        
        
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        self.drop_cols_all = os.path.join(self.config_dir, "remove_cols_all.xlsx")
        
        print(self.predict)
        print(self.data_csv)
        print(self.predict_column)


        
   
        
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
            DecisionTreeClassifier(random_state = self.random_state)
            
            ]
        
        classification = defaultdict(list)
        classification["MODEL"].extend(classification_models)
        classification["ALIAS"].extend(["BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC"])
        classification["NAME"].extend([
            "BaggingClassifier",
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "BalancedRandomForestClassifier",
            "BalancedBaggingClassifier",
            "DecisionTreeClassifier"])
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
        
        #
        cm_list = ["ACC"]

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)

        if not os.path.isdir(self.exp_result_dir):
            os.mkdir(self.exp_result_dir)
        
        if not os.path.isfile(self.result_excel):
            open(self.result_excel,"w+")
        
        if not os.path.isdir(self.exp_models_dir):
            os.mkdir(self.exp_models_dir)
        
    

        # init models
        self.models_init()
        
        # read data and config
        data_df = self.df
        # config = read_data(self.config)
        
        #Ignore this
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
        print(data_split)
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


        prev_acc=0
        for i, data in enumerate(zip(x_train_list, y_train_list, x_test_list, y_test_list)):

                X_train, y_train, X_test, y_test = data
        
                imputer_alias = self.imputation['ALIAS'][0]
                imputer_name = self.imputation['NAME'][0]
                
                        
                print("\nFOLD: {}, IMPUTATION: {}".format(i+1, imputer_name),imputer_name)
                
                

                imputer_dump_name = os.path.join(self.exp_models_dir, "{}_{}_imputer.pkl".format(i, imputer_alias))
                if os.path.isfile(imputer_dump_name) and not self.overwrite_models:
                    print("Loading previously saved imputation model at = {}".format(imputer_dump_name))
                    imputer = load(open(imputer_dump_name, 'rb'))
                    X_train_ = imputer.transform(X_train)
                    X_test_ = imputer.transform(X_test)
                else:
                    print("Performing fresh imputation.")
                    base_estimator = CLONE(self.imputation["MODEL"][0], safe=True)   
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
                    
                    print(metrics.classification_report(y_test, pred,output_dict=False))
                    report=metrics.classification_report(y_test, pred,output_dict=True)
                    
                    RESULTS[key]['ACC'] +=report["accuracy"]
                    
                    
                    
                    #FINDING MOST ACCURATE MODEL AND IMPUTER
                    if(prev_acc<report["accuracy"]):
                         max_acc_model=clf_dump_name
                         max_acc_imputer=imputer_dump_name
                    prev_acc=report["accuracy"]
                
                
                for t,val in PREDICTIONS.items():
                    if t == str(i) + "_" + imputer_alias:
                        path = os.path.join(self.temp_dir, "{}_predictions.csv".format(t))
                        val_df = pd.DataFrame(val,columns=self.pred_class_list)
                        val_df.to_csv(path, index=False)
                
        
        #Storing Result
        for key in (RESULTS.keys()):
              RESULTS[key]["ACC"]/=self.k_fold
        R = np.zeros([len(RESULTS.keys()), 1])
        
        for t, val in enumerate(RESULTS.items()):
                     p, performance = val
                     R[t,:] = list(performance.values())
        

        R = pd.DataFrame(R, index=RESULTS.keys(), columns=["ACC"])
        
        R.loc[:,"NAME"] = NAMES
                
        R.to_excel(self.result_excel)
        
        
        #Storing predictions
        
        imputer = load(open(max_acc_imputer, 'rb'))
        X = imputer.transform(X)
        if not os.path.isfile(self.pred_excel):
            open(self.pred_excel,"w+")
        clf = load(open(max_acc_model, 'rb'))
        pred_final = clf.predict(X)
        df=pd.DataFrame()
        df.loc[:,self.predict_column]=list(pred_final)
        df.to_excel(self.pred_excel)
        
        