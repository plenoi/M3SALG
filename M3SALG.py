import numpy as np
import pandas as pd

#!pip install jpype1
#!pip install lightgbm=3.3.2
#!pip install scikit-learn==1.0.2
#!pip install xgboost==1.6.2
#!pip install joblib==1.2.0

file = open('PLS.py','w')
file.write('import numpy as np'+"\n")
file.write('from sklearn.cross_decomposition import PLSRegression'+"\n")
file.write('from sklearn.base import BaseEstimator, ClassifierMixin'+"\n")
file.write('class PLS(BaseEstimator, ClassifierMixin):'+"\n")
file.write('    def __init__(self):'+"\n")
file.write('        self.clf = PLSRegression(n_components=2)'+"\n")
file.write('    def fit(self, X, y):'+"\n")
file.write('        self.clf.fit(X,y)'+"\n")
file.write('        return self'+"\n")
file.write('    def predict(self, X):'+"\n")
file.write('        pr = [np.round(min(max(np.round(item[0]),0.000001),0.999999)) for item in self.clf.predict(X)]'+"\n")
file.write('        return np.array(pr)'+"\n")
file.write('    def predict_proba(self, X):'+"\n")
file.write('        p_all = []'+"\n")
file.write('        ptmp = np.array([min(max(item[0],0.000001),0.999999) for item in self.clf.predict(X)],dtype=float)'+"\n")
file.write('        p_all.append(1-ptmp)'+"\n")
file.write('        p_all.append(ptmp)'+"\n")
file.write('        return np.transpose(np.array(p_all))'+"\n")
file.close()


from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
if not isJVMStarted():
    cdk_path = './model/cdk-2.7.1.jar'
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path)
cdk =  JPackage('org').openscience.cdk

def featsmi(fp_type, smis, size=1024, depth=6):
    fg = {
            "AP2D" : cdk.fingerprint.AtomPairs2DFingerprinter(),
            "CDK":cdk.fingerprint.Fingerprinter(size, depth),
            "CDKExt":cdk.fingerprint.ExtendedFingerprinter(size, depth),
            "CDKGraph":cdk.fingerprint.GraphOnlyFingerprinter(size, depth),
            "MACCS":cdk.fingerprint.MACCSFingerprinter(),
            "PubChem":cdk.fingerprint.PubchemFingerprinter(cdk.silent.SilentChemObjectBuilder.getInstance()),
            "Estate":cdk.fingerprint.EStateFingerprinter(),
            "KR":cdk.fingerprint.KlekotaRothFingerprinter(),
            "FP4" : cdk.fingerprint.SubstructureFingerprinter(),
            "FP4C" : cdk.fingerprint.SubstructureFingerprinter(),
            "Circle" : cdk.fingerprint.CircularFingerprinter(),
            "Hybrid" : cdk.fingerprint.HybridizationFingerprinter(),
         }
    sp = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())
    for i,smi in enumerate(smis):
        mol = sp.parseSmiles(smi)
        if fp_type == "FP4C":
            fingerprinter = fg[fp_type]
            nbit = fingerprinter.getSize()
            fp = fingerprinter.getCountFingerprint(mol)
            feat = np.array([int(fp.getCount(i)) for i in range(nbit)])
        else:
            fingerprinter = fg[fp_type]
            nbit = fingerprinter.getSize()
            fp = fingerprinter.getFingerprint(mol)
            feat = np.array([int(fp.get(i)) for i in range(nbit)])
        if i == 0:
            featx = feat.reshape(1,-1)
        else:
            featx = np.vstack((featx, feat.reshape(1,-1)))
    return featx

df = pd.read_csv('./input/smiles.csv', names=['Smiles'], header=None)
data = df['Smiles'].values


import joblib
fname = "AP2D,CDK,CDKExt,CDKGraph,MACCS,PubChem,Estate,KR,FP4,FP4C,Circle,Hybrid".split(',')
cname = "SVM,RF,ET,XGB,LGBM,ADA,MLP,NB,KNN,DT,LR,PLS".split(',')
used_clf_name = ['ET-CDK-0', 'KNN-Circle-0', 'KNN-Circle-2', 'MLP-KR-3',
       'ET-Circle-3', 'ET-Circle-4', 'PLS-Circle-4', 'SVM-Hybrid-4',
       'LR-AP2D-5', 'XGB-Circle-5', 'SVM-CDKExt-6', 'KNN-CDKExt-6',
       'PLS-KR-6', 'SVM-Circle-6', 'ET-Circle-7', 'LGBM-AP2D-8',
       'LGBM-CDK-8', 'ADA-CDK-8', 'XGB-CDKGraph-8', 'RF-MACCS-8',
       'KNN-Circle-8', 'RF-CDK-9', 'ET-CDK-9', 'RF-Estate-9', 'SVM-KR-9',
       'MLP-Hybrid-9']
used_clf_feat = np.unique([item.split('-')[1] for item in used_clf_name])
used_clfs = joblib.load('./model/used_clfs.sav')
finalclf = joblib.load('./model/final_clf.sav')
feat_all = {item:featsmi(item,data) for item in used_clf_feat}
for i,item in enumerate(used_clf_name):
    efn = item.split('-')[1]
    probf = used_clfs[i].predict_proba(feat_all[efn])[:,0].reshape(-1,1)
    if i == 0:
        allprob = probf
    else:
        allprob = np.hstack((allprob,probf))

pr = finalclf.predict_proba(allprob)[:,0]

label = ['Positive', 'Negative'] 
file = open("./output/predict_result.csv","w")
for i, head, in enumerate(data):
    file.write(head+","+label[int(pr[i]+0.5)]+","+str(1-pr[i])+"\n")
file.close()