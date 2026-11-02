import os
import json,warnings,collections
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,mean_absolute_error,confusion_matrix
warnings.filterwarnings('ignore');P=Path('analysis');d=json.load(open(os.environ.get('CLINICAL_DATASET', 'data/anchored_with_phq4_only_aug.json')));m=pd.read_csv(P/'record_index.csv');fs=json.load(open(P/'summary.json'))['features'];raw=[];cov=[];pct=[];ref=[]
for r in d:
 a={};b={};c={};e={}
 for f in fs:
  x=np.array([v for v in (r.get(f) or {}).values() if v is not None]);b[f]=len(x);c[f]=r.get(f+'__pctile');e[f]=r.get(f+'__pctile_n')
  for stat,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:a[f+'_'+stat]=fn(x) if len(x) else np.nan
 raw.append(a);cov.append(b);pct.append(c);ref.append(e)
raw=pd.DataFrame(raw);cov=pd.DataFrame(cov);pct=pd.DataFrame(pct);ref=pd.DataFrame(ref);good=m.tier.eq(1)&~m.duplicated(['source','subject_id','window'],False)&cov.sum(axis=1).gt(0)
frames={'raw':raw,'raw_coverage':pd.concat([raw,cov.add_suffix('_n')],axis=1),'coverage':cov,'provided_percentiles':pct,'reference_counts':ref,'train_only_percentiles':raw};frames['passive_raw_coverage']=frames['raw_coverage'].drop(columns=[c for c in frames['raw_coverage'] if c.startswith(('stress_','social_level_'))])
res=[];predictions=[]
def metric(y,p):
 return dict(accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),macro_f1=f1_score(y,p,labels=[0,1,2,3],average='macro',zero_division=0),mae=mean_absolute_error(y,p),severe_recall=np.mean(p[y==3]==3),critical_miss=np.mean(p[y==3]<=1),severe_prediction_rate=np.mean(p==3),confusion=confusion_matrix(y,p,labels=[0,1,2,3]).tolist())
for scope in ['tier1_clean','globem_tier1_clean','ced_clean']:
 ids=np.where(good&(m.source.eq(scope.split('_')[0]) if scope!='tier1_clean' else True))[0];y=m.iloc[ids]['class'].to_numpy();g=m.iloc[ids].group.to_numpy();folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ids,y,g))
 for feat,X0 in frames.items():
  X=X0.iloc[ids];X=X.loc[:,X.notna().any()]
  for model in (['LR','RF'] if feat in ['raw_coverage','passive_raw_coverage'] else ['LR']):
   p=np.zeros(len(ids),int)
   for fi,(tr,te) in enumerate(folds):
    assert not set(g[tr])&set(g[te]);clf=LogisticRegression(class_weight='balanced',max_iter=2000) if model=='LR' else RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1)
    steps=[SimpleImputer(strategy='median',keep_empty_features=True)];steps += [QuantileTransformer(n_quantiles=min(100,len(tr)),random_state=42)] if feat=='train_only_percentiles' else [];steps += [StandardScaler(),clf];pipe=make_pipeline(*steps);pipe.fit(X.iloc[tr],y[tr]);p[te]=pipe.predict(X.iloc[te])
    for ix in te:predictions.append(dict(scope=scope,features=feat,model=model,index=int(ids[ix]),fold=fi,group=g[ix],true=int(y[ix]),pred=int(p[ix])))
   out=dict(scope=scope,features=feat,model=model,n=len(ids),subjects=len(set(g)),**metric(y,p));res.append(out);print(out,flush=True)
 p=np.zeros(len(ids),int)
 for tr,te in folds:p[te]=collections.Counter(y[tr]).most_common(1)[0][0]
 res.append(dict(scope=scope,features='none',model='majority',n=len(ids),subjects=len(set(g)),**metric(y,p)))
pd.DataFrame(res).to_csv(P/'clean_baselines.csv',index=False);pd.DataFrame(predictions).to_csv(P/'clean_predictions.csv',index=False)
# Common physical aggregates: duration in minutes, home time, total screen time, no supplied percentiles.
common=[]
for r in d:
 a={}
 for f in ['sleep','home_time_min','screen_total_min']:
  field=('sleep_duration_min' if r['source']=='globem' else 'sleep_duration') if f=='sleep' else f;scale=60 if field=='sleep_duration' else 1;x=np.array([v*scale for v in (r.get(field) or {}).values() if v is not None]);a[f+'_n']=len(x)
  for stat,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:a[f+'_'+stat]=fn(x) if len(x) else np.nan
 common.append(a)
X=pd.DataFrame(common);transfer=[]
for train,test in [('globem','ced'),('ced','globem')]:
 tr=np.where(good&m.source.eq(train))[0];te=np.where(good&m.source.eq(test))[0]
 for mod in ['LR','RF']:
  clf=LogisticRegression(class_weight='balanced',max_iter=2000) if mod=='LR' else RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1)
  pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),clf);pipe.fit(X.iloc[tr],m.iloc[tr]['class']);p=pipe.predict(X.iloc[te]);transfer.append(dict(train=train,test=test,model=mod,n=len(te),**metric(m.iloc[te]['class'].to_numpy(),p)))
pd.DataFrame(transfer).to_csv(P/'transfer_baselines.csv',index=False)
# Source predictability from coverage and reference counts only, grouped by subject.
ix=np.where(good)[0];y=m.iloc[ix].source.to_numpy();g=m.iloc[ix].group.to_numpy();diag=[]
for name,X0 in [('coverage',cov),('reference_counts',ref)]:
 X=X0.iloc[ix];X=X.loc[:,X.notna().any()];p=np.empty(len(ix),object)
 for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,g):
  pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(max_iter=2000));pipe.fit(X.iloc[tr],y[tr]);p[te]=pipe.predict(X.iloc[te])
 diag.append(dict(features=name,source_accuracy=accuracy_score(y,p)))
json.dump(diag,open(P/'source_diagnostic.json','w'),indent=2)
