import os
import json,warnings,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
warnings.filterwarnings('ignore')
P=Path('analysis');src=Path(os.environ.get('CLINICAL_DATASET', 'data/anchored_with_phq4_only_aug.json'));d=json.load(open(src));assert hashlib.sha256(src.read_bytes()).hexdigest()==json.load(open(P/'summary.json'))['sha256'];m=pd.read_csv(P/'record_index.csv');fs=json.load(open(P/'summary.json'))['features']
nonempty=np.array([any(v is not None for f in fs for v in (r.get(f) or {}).values()) for r in d]);good=m.tier.eq(1)&~m.duplicated(['source','subject_id','window'],False)&nonempty
features=['sleep_hours','sleep_efficiency','wake_after_sleep_min','bedtime_min','home_time_min','bt_unique_devices','screen_night_min','screen_total_min','screen_unlocks','walking_min','calls_out','conversation_min']
names={'sleep_hours':'sleep duration hours','sleep_efficiency':'sleep efficiency percent','wake_after_sleep_min':'wake after sleep minutes','bedtime_min':'unwrapped bedtime minutes','home_time_min':'home time minutes','bt_unique_devices':'Bluetooth device count','screen_night_min':'night screen minutes','screen_total_min':'total screen minutes','screen_unlocks':'screen unlock count','walking_min':'walking minutes','calls_out':'outgoing call count','conversation_min':'conversation minutes'}
rows=[]
for r in d:
 row={}
 for f in features:
  key=('sleep_duration_min' if r['source']=='globem' else 'sleep_duration') if f=='sleep_hours' else f;scale=1/60 if f=='sleep_hours' and r['source']=='globem' else 1
  vals=np.array([v*scale for v in (r.get(key) or {}).values() if v is not None]);row[f+'__count']=len(vals)
  for stat,fn in [('mean',np.mean),('variation',np.std),('minimum',np.min),('maximum',np.max)]:row[f+'__'+stat]=float(fn(vals)) if len(vals) else np.nan
 rows.append(row)
X=pd.DataFrame(rows);predictions=[];results=[];examples=[]
def metrics(y,p):
 return dict(accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),macro_f1=f1_score(y,p,labels=[0,1,2,3],average='macro',zero_division=0),severe_recall=np.mean(p[y==3]==3),critical_miss=np.mean(p[y==3]<=1),confusion=confusion_matrix(y,p,labels=[0,1,2,3]).tolist())
def classify(train,test,y,mode):
 clf=LogisticRegression(C=1,class_weight='balanced',max_iter=2000)
 pre=[SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()] if mode=='raw' else [OneHotEncoder(handle_unknown='ignore')] if mode=='concepts' else [TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)]
 pipe=make_pipeline(*pre,clf);pipe.fit(train,y);return pipe.predict(test)
def represent(A,B,fl):
 # Fitted reference cutoffs: values from training rows only; no labels.
 cuts={c:np.nanquantile(A[c],[1/3,2/3]) if A[c].notna().any() else np.array([0.,0.]) for c in A if not c.endswith('__count')}
 def convert(T):
  texts=[];literal=[];concepts=[]
  for _,r in T.iterrows():
   sentences=[];numbers=[];codes=[]
   for f in fl:
    n=int(r[f+'__count']);name=names[f];codes.append(str(n));sentences.append(f'{name} observed {n} days of seven.');numbers.append(f'{name} observed {n} days of seven.')
    for st in ['mean','variation','minimum','maximum']:
     c=f+'__'+st;v=r[c];level='unavailable' if np.isnan(v) else ['lower','middle','upper'][int(np.searchsorted(cuts[c],v,side='right'))]
     codes.append(level);sentences.append(f'{name} {st} {level}.');numbers.append(f'{name} {st} '+('unavailable.' if np.isnan(v) else f'{v:.2f}.'))
   texts.append(' '.join(sentences));literal.append(' '.join(numbers));concepts.append(codes)
  return {'qualitative_text':texts,'numeric_text':literal,'concepts':np.array(concepts),'raw':T}
 return convert(A),convert(B)
def run_experiment():
    for scope in ['tier1_clean','globem_tier1_clean','ced_clean']:
     ids=np.where(good&(m.source.eq(scope.split('_')[0]) if scope!='tier1_clean' else True))[0];y=m.iloc[ids]['class'].to_numpy();groups=m.iloc[ids].group.to_numpy();folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ids,y,groups));ps={k:np.zeros(len(ids),int) for k in ['raw','numeric_text','qualitative_text','concepts']}
     # Match previous saved folds exactly.
     old=pd.read_csv(P/'clean_predictions.csv');old=old[(old.scope==scope)&(old.features=='raw_coverage')&(old.model=='LR')].set_index('index')
     for fold,(tr,te) in enumerate(folds):
      assert not set(groups[tr])&set(groups[te]);assert all(old.loc[ids[te],'fold']==fold)
      A,B=represent(X.iloc[ids[tr]],X.iloc[ids[te]],features)
      for mode in ps:
       ps[mode][te]=classify(A[mode],B[mode],y[tr],mode)
       for j in te:predictions.append(dict(scope=scope,mode=mode,index=int(ids[j]),fold=fold,group=groups[j],true=int(y[j]),pred=int(ps[mode][j])))
      if scope=='tier1_clean' and fold==0:
       examples=[dict(index=int(ids[j]),numeric_summary=B['numeric_text'][k],qualitative_summary=B['qualitative_text'][k]) for k,j in enumerate(te[:3])]
     for mode,p in ps.items():
      out=dict(scope=scope,mode=mode,n=len(ids),subjects=len(set(groups)),**metrics(y,p));results.append(out);print(out,flush=True)
    pd.DataFrame(results).to_csv(P/'bottleneck_results.csv',index=False);pd.DataFrame(predictions).to_csv(P/'bottleneck_predictions.csv',index=False);json.dump(examples,open(P/'bottleneck_examples.json','w'),indent=2)
    # Matched transfer on only shared measurements; reference cutoffs fitted on source training data.
    common=['sleep_hours','home_time_min','screen_total_min'];transfer=[]
    for train,test in [('globem','ced'),('ced','globem')]:
     tr=np.where(good&m.source.eq(train))[0];te=np.where(good&m.source.eq(test))[0];cols=[c for c in X if c.split('__')[0] in common];A,B=represent(X.iloc[tr][cols],X.iloc[te][cols],common)
     for mode in A:
      p=classify(A[mode],B[mode],m.iloc[tr]['class'],mode);transfer.append(dict(train=train,test=test,mode=mode,**metrics(m.iloc[te]['class'].to_numpy(),p)))
    pd.DataFrame(transfer).to_csv(P/'bottleneck_transfer.csv',index=False)
    # Paired subject bootstrap on fixed OOF predictions. Compare confusion matrices without refitting.
    a=pd.DataFrame(predictions);rng=np.random.default_rng(42);ci=[]
    def score(cm):
     recall=np.divide(np.diag(cm),cm.sum(axis=1),out=np.zeros(4),where=cm.sum(axis=1)>0);f1=np.divide(2*np.diag(cm),cm.sum(axis=0)+cm.sum(axis=1),out=np.zeros(4),where=(cm.sum(axis=0)+cm.sum(axis=1))>0);return np.array([recall.mean(),f1.mean()])
    for scope in a.scope.unique():
     z=a[a.scope==scope];gs=sorted(z.group.unique());cms={mode:np.array([confusion_matrix(v.true,v.pred,labels=[0,1,2,3]) for _,v in z[z['mode']==mode].groupby('group',sort=True)]) for mode in z['mode'].unique()}
     for mode in ['numeric_text','qualitative_text','concepts']:
      diffs=[]
      for _ in range(2000):
       ix=rng.integers(0,len(gs),len(gs));diffs.append(score(cms[mode][ix].sum(axis=0))-score(cms['raw'][ix].sum(axis=0)))
      q=np.quantile(diffs,[.025,.975],axis=0);delta=score(cms[mode].sum(axis=0))-score(cms['raw'].sum(axis=0));ci.append(dict(scope=scope,mode=mode,ba_difference=delta[0],ba_low=q[0,0],ba_high=q[1,0],f1_difference=delta[1],f1_low=q[0,1],f1_high=q[1,1]))
    pd.DataFrame(ci).to_csv(P/'bottleneck_paired_intervals.csv',index=False)


if __name__ == "__main__":
    run_experiment()
