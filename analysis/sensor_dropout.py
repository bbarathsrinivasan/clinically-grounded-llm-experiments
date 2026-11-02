"""Nested grouped expected sensor-dropout training versus matched zero-dropout control."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,re,warnings,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import semantic_experiment as base
warnings.filterwarnings('error',category=ConvergenceWarning)
P=Path('analysis');torch.set_num_threads(4)
MASKS={'none':[],'sleep':['sleep_hours','sleep_efficiency','wake_after_sleep_min','bedtime_min'],'mobility_environment':['home_time_min','walking_min','bt_unique_devices'],'phone_communication':['screen_night_min','screen_total_min','screen_unlocks','calls_out','conversation_min']}
CS=[.0001,.001,.01,.1,1.,10.];PS=[0.,.25,.5];MODES=['raw','numeric_hybrid'];CONDITIONS=list(MASKS);ids=np.where(base.good)[0];X=base.X.iloc[ids].reset_index(drop=True);meta=base.m.iloc[ids].reset_index(drop=True);variants={};sections=set()
# Numeric summaries use no reference distribution and therefore can be rendered once.
for mask,fields in MASKS.items():
 B=X.copy()
 for c in B:
  if c.split('__')[0] in fields:B[c]=0 if c.endswith('__count') else np.nan
 sentences=[]
 for _,r in B.iterrows():
  cs=[]
  for f in base.features:
   name=base.names[f];parts=[f'{name} observed {int(r[f+"__count"])} days of seven.']
   for st in ['mean','variation','minimum','maximum']:
    v=r[f+'__'+st];parts.append(f'{name} {st} '+('unavailable.' if np.isnan(v) else f'{v:.2f}.'))
   cs.append(' '.join(parts))
  sentences.append(cs);sections.update(cs)
 variants[mask]={'raw':B.to_numpy(),'sections':sentences}
texts=sorted(sections);print('Encoding',len(texts),'unique numeric sections',flush=True)
encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',local_files_only=True);encoder.eval()
for p in encoder.parameters():p.requires_grad_(False)
lengths=[len(t) for t in encoder.tokenizer(texts,truncation=False)['input_ids']];assert max(lengths)<=encoder.max_seq_length
with torch.inference_mode():v=encoder.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False)
lookup={t:i for i,t in enumerate(texts)}
for mask in CONDITIONS:
 z=variants[mask];emb=np.array([v[[lookup[t] for t in cs]].mean(0) for cs in z.pop('sections')]);emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12);assert np.isfinite(emb).all();z['numeric_hybrid']=np.concatenate([z['raw'],emb],axis=1)
y=meta['class'].to_numpy();g=meta.group.to_numpy();old=pd.read_csv(P/'nested_predictions.csv');rows=[];search=[];choices=[]
def prepare(tr,te,mode):
 assert not set(g[tr])&set(g[te]);imp=SimpleImputer(strategy='median',keep_empty_features=True);imp.fit(variants['none'][mode][tr]);sc=StandardScaler();sc.fit(imp.transform(variants['none'][mode][tr]))
 A={c:sc.transform(imp.transform(variants[c][mode][tr])) for c in CONDITIONS};B={c:sc.transform(imp.transform(variants[c][mode][te])) for c in CONDITIONS};return A,B

def fit(A,labels,C,p):
 clf=LogisticRegression(C=C,class_weight='balanced',max_iter=4000)
 if p==0:clf.fit(A['none'],labels)
 else:
  augmented=np.concatenate([A[c] for c in CONDITIONS]);targets=np.tile(labels,4);weights=np.concatenate([np.full(len(labels),1-p)]+[np.full(len(labels),p/3)]*3);assert np.isclose(weights.sum(),len(labels));clf.fit(augmented,targets,sample_weight=weights)
 return clf

for scope in ['tier1_clean','globem_tier1_clean','ced_clean']:
 ix=np.arange(len(ids)) if scope=='tier1_clean' else np.where(meta.source.eq(scope.split('_')[0]))[0]
 folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ix,y[ix],g[ix]))
 for fold,(tr0,te0) in enumerate(folds):
  tr,te=ix[tr0],ix[te0];assert not set(g[tr])&set(g[te]);prev=old[(old.scope==scope)&(old['mode']=='raw')].set_index('index');assert all(prev.loc[ids[te],'fold']==fold)
  inner=[(tr[a],tr[b]) for a,b in StratifiedGroupKFold(3,shuffle=True,random_state=117).split(tr,y[tr],g[tr])]
  for mode in MODES:
   prepared=[(a,b,*prepare(a,b,mode)) for a,b in inner];trials=[]
   for p in PS:
    for C in CS:
     scores=[]
     for it,iv,A,B in prepared:
      clf=fit(A,y[it],C,p)
      scores.append(np.mean([f1_score(y[iv],clf.predict(B[c]),labels=[0,1,2,3],average='macro',zero_division=0) for c in CONDITIONS]))
     value=float(np.mean(scores));trials.append((value,p,C));search.append(dict(scope=scope,fold=fold,mode=mode,p=p,C=C,inner_robust_macro_f1=value))
   # Stable ties prefer p=0 then stronger regularization; both arms use same validation objective.
   best=max(trials,key=lambda z:(z[0],-z[1],-z[2]));zero=max([z for z in trials if z[1]==0],key=lambda z:(z[0],-z[2]));A,B=prepare(tr,te,mode)
   for policy,(value,p,C) in [('zero_dropout',zero),('selected_dropout',best)]:
    choices.append(dict(scope=scope,fold=fold,mode=mode,policy=policy,p=p,C=C,inner_robust_macro_f1=value));clf=fit(A,y[tr],C,p)
    for mask in CONDITIONS:
     pred=clf.predict(B[mask])
     for j,q in zip(te,pred):rows.append(dict(scope=scope,fold=fold,mode=mode,policy=policy,mask=mask,index=int(ids[j]),group=g[j],true=int(y[j]),pred=int(q)))
  print('Finished',scope,'fold',fold,flush=True)
  pd.DataFrame(rows).to_csv(P/'dropout_predictions.csv',index=False);pd.DataFrame(search).to_csv(P/'dropout_search.csv',index=False);pd.DataFrame(choices).to_csv(P/'dropout_selected.csv',index=False)
a=pd.DataFrame(rows);results=[]
for (scope,mode,policy,mask),z in a.groupby(['scope','mode','policy','mask']):results.append(dict(scope=scope,mode=mode,policy=policy,mask=mask,n=len(z),**base.metrics(z.true.to_numpy(),z.pred.to_numpy())))
r=pd.DataFrame(results);r.to_csv(P/'dropout_results.csv',index=False);r.groupby(['scope','mode','policy'])[['balanced_accuracy','macro_f1']].mean().to_csv(P/'dropout_robustness.csv')
# Cluster-bootstrap paired mean metrics across all four conditions; does not include refitting uncertainty.
def score(cm):
 return np.array([np.mean(np.divide(np.diag(cm),cm.sum(1),out=np.zeros(4),where=cm.sum(1)>0)),np.mean(np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(4),where=(cm.sum(0)+cm.sum(1))>0))])
rng=np.random.default_rng(42);cis=[]
for scope in a.scope.unique():
 z=a[a.scope==scope];cms={}
 for mode in MODES:
  for policy in ['zero_dropout','selected_dropout']:
   cms[(mode,policy)]=np.array([[confusion_matrix(q.true,q.pred,labels=[0,1,2,3]) for _,q in z[(z['mode']==mode)&(z.policy==policy)&(z['mask']==mask)].groupby('group',sort=True)] for mask in CONDITIONS])
 pairs=[(('raw','selected_dropout'),('raw','zero_dropout')),(('numeric_hybrid','selected_dropout'),('numeric_hybrid','zero_dropout')),(('numeric_hybrid','selected_dropout'),('raw','selected_dropout'))]
 for left,right in pairs:
  L,R=cms[left],cms[right];n=L.shape[1]
  def diff(ix):return np.mean([score(c[:,ix].sum(1)[k]) for k in range(4) for c in [L]],axis=0)-np.mean([score(R[:,ix].sum(1)[k]) for k in range(4)],axis=0)
  samples=np.array([diff(rng.integers(0,n,n)) for _ in range(1000)]);ci=np.quantile(samples,[.025,.975],axis=0);d=diff(np.arange(n));cis.append(dict(scope=scope,left='/'.join(left),right='/'.join(right),ba_delta=d[0],ba_lo=ci[0,0],ba_hi=ci[1,0],f1_delta=d[1],f1_lo=ci[0,1],f1_hi=ci[1,1]))
pd.DataFrame(cis).to_csv(P/'dropout_paired_intervals.csv',index=False)
json.dump(dict(C_grid=CS,dropout_probabilities=PS,method='Exact weighted expectation of one-group random dropout: clean weight 1-p; each of three masked copies p/3; total weight per original row 1',selection='inner 3-fold mean macro-F1 over clean plus three single-group outages, equally weighted; outer fixed 5-fold subject splits',controls='p=0 retuned with same robustness objective; selected arm may choose p=0',preprocessing='fit imputer/scaler on clean training rows only, identical across p; encoder frozen',masks=MASKS,max_section_tokens=max(lengths),limitations='Exploratory adaptive reuse of outer folds; simulated outages; selected dropout search has greater search budget than fixed p=0 control; no untouched final test'),open(P/'dropout_config.json','w'),indent=2)
print(pd.read_csv(P/'dropout_robustness.csv').round(4).to_string(index=False),flush=True)
