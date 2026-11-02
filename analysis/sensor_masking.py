"""Evaluation-only modality removal; reuse nested-selected C, no mask-specific tuning."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import re,json,warnings
from pathlib import Path
import numpy as np,pandas as pd,torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import semantic_experiment as base
warnings.filterwarnings('error',category=ConvergenceWarning)
P=Path('analysis');torch.set_num_threads(4)
MASKS={'none':[],'sleep':['sleep_hours','sleep_efficiency','wake_after_sleep_min','bedtime_min'],'mobility_environment':['home_time_min','walking_min','bt_unique_devices'],'phone_communication':['screen_night_min','screen_total_min','screen_unlocks','calls_out','conversation_min']}
MODES=['raw','numeric_text','qualitative_text'];selected=pd.read_csv(P/'nested_selected.csv').set_index(['scope','fold','mode']);old=pd.read_csv(P/'nested_predictions.csv');jobs=[];sections=set()
def chunk(t):
 s=re.split(r'(?<=\.)\s+',t.strip());assert len(s)==60;return [' '.join(s[k:k+5]) for k in range(0,len(s),5)]
def represent(A,B):
 a,b=base.represent(A,B,base.features)
 for z in [a,b]:
  for mode in MODES[1:]:
   z[mode]=[chunk(t) for t in z[mode]]
   for cs in z[mode]:sections.update(cs)
 return a,b
for scope in ['tier1_clean','globem_tier1_clean','ced_clean']:
 ids=np.where(base.good&(base.m.source.eq(scope.split('_')[0]) if scope!='tier1_clean' else True))[0];y=base.m.iloc[ids]['class'];g=base.m.iloc[ids].group
 for fold,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ids,y,g)):
  tr,te=ids[tr],ids[te];assert not set(base.m.iloc[tr].group)&set(base.m.iloc[te].group)
  prev=old[(old.scope==scope)&(old['mode']=='raw')].set_index('index');assert all(prev.loc[te,'fold']==fold)
  A=base.X.iloc[tr];original=base.X.iloc[te];training,clean=represent(A,original);conditions={};affected={};empty={}
  for mask,fields in MASKS.items():
   B=original.copy();cols=[c for c in B if c.split('__')[0] in fields];counts=[f+'__count' for f in fields];affected[mask]=(original[counts].sum(axis=1)>0).to_numpy() if fields else np.ones(len(te),bool)
   for c in cols:B[c]=0 if c.endswith('__count') else np.nan
   untouched=[c for c in B if c not in cols];assert B[untouched].equals(original[untouched])
   for c in cols:assert (B[c].eq(0).all() if c.endswith('__count') else B[c].isna().all())
   empty[mask]=(B[[c for c in B if c.endswith('__count')]].sum(1)==0).to_numpy()
   conditions[mask]=clean if mask=='none' else represent(A,B)[1]
  jobs.append(dict(scope=scope,fold=fold,tr=tr,te=te,training=training,conditions=conditions,affected=affected,empty=empty))
print('Prepared all masking conditions; unique sections:',len(sections),flush=True)
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',local_files_only=True);model.eval()
for p in model.parameters():p.requires_grad_(False)
# Recreate the original encoder batch inventory to keep floating-point results identical.
# Frozen encoders can vary slightly with batch padding; preserve the prior encoding order.
original_sections=set()
def inventory(tr,te,features):
 cols=[c for c in base.X if c.split('__')[0] in features];a,b=base.represent(base.X.iloc[tr][cols],base.X.iloc[te][cols],features)
 for rep in [a,b]:
  for mode in MODES[1:]:
   for text in rep[mode]:
    ss=re.split(r'(?<=\.)\s+',text.strip())
    original_sections.update(' '.join(ss[k:k+5]) for k in range(0,len(ss),5))
def original_task(tr,te,features):
 inventory(tr,te,features)
 for it,iv in StratifiedGroupKFold(3,shuffle=True,random_state=117).split(tr,base.m.iloc[tr]['class'],base.m.iloc[tr].group):inventory(tr[it],tr[iv],features)
for job in jobs:original_task(job['tr'],job['te'],base.features)
for src,dst in [('globem','ced'),('ced','globem')]:original_task(np.where(base.good&base.m.source.eq(src))[0],np.where(base.good&base.m.source.eq(dst))[0],['sleep_hours','home_time_min','screen_total_min'])
original_texts=sorted(original_sections);extra_texts=sorted(sections-original_sections);texts=original_texts+extra_texts
print('Reproducing prior encoder batches:',len(original_texts),'plus new sections:',len(extra_texts),flush=True)
lens=[len(x) for x in model.tokenizer(texts,truncation=False)['input_ids']];assert max(lens)<=model.max_seq_length
with torch.inference_mode():
 vectors=model.encode(original_texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False)
 if extra_texts:vectors=np.concatenate([vectors,model.encode(extra_texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False)])
assert np.isfinite(vectors).all();lookup={t:i for i,t in enumerate(texts)}
def matrix(rep,mode):
 raw=rep['raw'].to_numpy()
 if mode=='raw':return raw
 v=np.array([vectors[[lookup[t] for t in cs]].mean(0) for cs in rep[mode]]);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12);return np.concatenate([raw,v],axis=1)
rows=[]
for j in jobs:
 for mode in MODES:
  A=matrix(j['training'],mode);imp=SimpleImputer(strategy='median',keep_empty_features=True);A=imp.fit_transform(A);sc=StandardScaler();A=sc.fit_transform(A);C=float(selected.loc[(j['scope'],j['fold'],mode),'C']);clf=LogisticRegression(C=C,class_weight='balanced',max_iter=3000);clf.fit(A,base.m.iloc[j['tr']]['class'])
  for mask,B in j['conditions'].items():
   pred=clf.predict(sc.transform(imp.transform(matrix(B,mode))))
   if mask=='none':
    expected=old[(old.scope==j['scope'])&(old['mode']==mode)].set_index('index').loc[j['te'],'pred'].to_numpy();assert np.array_equal(pred,expected),'Unmasked baseline failed to reproduce'
   for k,(i,p) in enumerate(zip(j['te'],pred)):rows.append(dict(scope=j['scope'],fold=j['fold'],mode=mode,mask=mask,index=int(i),group=base.m.iloc[i].group,true=int(base.m.iloc[i]['class']),pred=int(p),affected=bool(j['affected'][mask][k]),all_missing=bool(j['empty'][mask][k])))
 print('Completed',j['scope'],'fold',j['fold'],flush=True)
a=pd.DataFrame(rows);a.to_csv(P/'masking_predictions.csv',index=False)
results=[]
for (scope,mode,mask),z in a.groupby(['scope','mode','mask']):
 clean=a[(a.scope==scope)&(a['mode']==mode)&(a['mask']=='none')];before=base.metrics(clean.true.to_numpy(),clean.pred.to_numpy());met=base.metrics(z.true.to_numpy(),z.pred.to_numpy());results.append(dict(scope=scope,mode=mode,mask=mask,n=len(z),affected=int(z.affected.sum()),all_missing=int(z.all_missing.sum()),ba_change=met['balanced_accuracy']-before['balanced_accuracy'],f1_change=met['macro_f1']-before['macro_f1'],**met))
pd.DataFrame(results).to_csv(P/'masking_results.csv',index=False)
# Paired cluster bootstrap: masked hybrid vs masked raw, and difference in performance change.
def score(cm):
 return np.array([np.mean(np.divide(np.diag(cm),cm.sum(1),out=np.zeros(4),where=cm.sum(1)>0)),np.mean(np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(4),where=(cm.sum(0)+cm.sum(1))>0))])
rng=np.random.default_rng(42);cis=[]
for scope in a.scope.unique():
 z=a[a.scope==scope];cms={(mode,mask):np.array([confusion_matrix(v.true,v.pred,labels=[0,1,2,3]) for _,v in q.groupby('group',sort=True)]) for (mode,mask),q in z.groupby(['mode','mask'])}
 for mask in list(MASKS)[1:]:
  for mode in MODES[1:]:
   h=cms[(mode,mask)];r=cms[('raw',mask)];hc=cms[(mode,'none')];rc=cms[('raw','none')];samples=[]
   for _ in range(2000):
    ix=rng.integers(0,len(h),len(h));delta=score(h[ix].sum(0))-score(r[ix].sum(0));change=delta-(score(hc[ix].sum(0))-score(rc[ix].sum(0)));samples.append(np.r_[delta,change])
   q=np.quantile(samples,[.025,.975],axis=0);delta=score(h.sum(0))-score(r.sum(0));change=delta-(score(hc.sum(0))-score(rc.sum(0)));cis.append(dict(scope=scope,mode=mode,mask=mask,masked_ba_delta=delta[0],masked_ba_lo=q[0,0],masked_ba_hi=q[1,0],masked_f1_delta=delta[1],masked_f1_lo=q[0,1],masked_f1_hi=q[1,1],ba_change_advantage=change[0],ba_change_lo=q[0,2],ba_change_hi=q[1,2],f1_change_advantage=change[1],f1_change_lo=q[0,3],f1_change_hi=q[1,3]))
pd.DataFrame(cis).to_csv(P/'masking_paired_intervals.csv',index=False)
json.dump(dict(masks=MASKS,protocol='Train unmasked, test one absent feature family; count=0 and aggregates=NaN; regenerate summaries using training-only thresholds',C='reuse nested_selected.csv, no retuning',encoder='frozen all-MiniLM-L6-v2, mean feature-section pooling',max_tokens=max(lens),unmasked_prediction_check='exact match to nested_predictions.csv for every row/model/scope',limitations='Controlled simultaneous group removal, not natural missingness or unseen-dataset validation. Repeated exploratory use of outer folds. Fixed-prediction bootstrap does not include refit uncertainty.'),open(P/'masking_config.json','w'),indent=2)
print(pd.DataFrame(results).query("scope=='tier1_clean'").drop(columns='confusion').round(4).to_string(index=False),flush=True)
