"""Frozen MiniLM, feature-section mean pooling; no target-informed encoder updates."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import re,json,hashlib,warnings
from pathlib import Path
import numpy as np,pandas as pd,torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import semantic_experiment as base
warnings.filterwarnings('ignore');P=Path('analysis');torch.set_num_threads(4);jobs=[];allchunks=set()
def chunks(text):
 s=re.split(r'(?<=\.)\s+',text.strip());assert len(s)%5==0
 return [' '.join(s[i:i+5]) for i in range(0,len(s),5)]
def addjob(scope,fold,tr,te,features,transfer=False):
 cols=[c for c in base.X if c.split('__')[0] in features];A,B=base.represent(base.X.iloc[tr][cols],base.X.iloc[te][cols],features)
 for mode in ['numeric_text','qualitative_text']:
  a=[chunks(t) for t in A[mode]];b=[chunks(t) for t in B[mode]]
  for cs in a+b:allchunks.update(cs)
  jobs.append(dict(scope=scope,fold=fold,tr=tr,te=te,mode=mode,a=a,b=b,transfer=transfer))
old=pd.read_csv(P/'bottleneck_predictions.csv')
for scope in ['tier1_clean','globem_tier1_clean','ced_clean']:
 ids=np.where(base.good&(base.m.source.eq(scope.split('_')[0]) if scope!='tier1_clean' else True))[0];y=base.m.iloc[ids]['class'].to_numpy();g=base.m.iloc[ids].group.to_numpy()
 for fold,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ids,y,g)):
  assert not set(g[tr])&set(g[te]);prev=old[(old.scope==scope)&(old['mode']=='raw')].set_index('index');assert all(prev.loc[ids[te],'fold']==fold)
  addjob(scope,fold,ids[tr],ids[te],base.features)
for train,test in [('globem','ced'),('ced','globem')]:
 tr=np.where(base.good&base.m.source.eq(train))[0];te=np.where(base.good&base.m.source.eq(test))[0];assert not set(base.m.iloc[tr].group)&set(base.m.iloc[te].group)
 addjob(train+'_to_'+test,-1,tr,te,['sleep_hours','home_time_min','screen_total_min'],True)
texts=sorted(allchunks);print('Unique feature sections:',len(texts),flush=True)
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',local_files_only=True,device='cpu');model.eval()
for p in model.parameters():p.requires_grad_(False)
lens=[len(t) for t in model.tokenizer(texts,add_special_tokens=True,truncation=False)['input_ids']];assert max(lens)<=model.max_seq_length
fingerprint=hashlib.sha256(('miniLM-feature-average-v1\n'+'\n'.join(texts)).encode()).hexdigest();cache=P/'frozen_section_vectors.npz'
if cache.exists() and str(np.load(cache)['fingerprint'])==fingerprint:vec=np.load(cache)['vectors']
else:
 with torch.inference_mode():vec=model.encode(texts,batch_size=64,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
 np.savez_compressed(cache,vectors=vec,fingerprint=fingerprint)
assert np.isfinite(vec).all();lookup={t:i for i,t in enumerate(texts)}
def embed(sections):
 z=np.array([vec[[lookup[t] for t in cs]].mean(axis=0) for cs in sections]);return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
rows=[];transfer=[]
for j in jobs:
 A=embed(j['a']);B=embed(j['b']);clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000));clf.fit(A,base.m.iloc[j['tr']]['class']);pred=clf.predict(B)
 for i,p in zip(j['te'],pred):
  record=dict(scope=j['scope'],fold=j['fold'],mode=j['mode'],index=int(i),group=base.m.iloc[i].group,true=int(base.m.iloc[i]['class']),pred=int(p));(transfer if j['transfer'] else rows).append(record)
 print('completed',j['scope'],j['fold'],j['mode'],flush=True)
a=pd.DataFrame(rows);a.to_csv(P/'frozen_encoder_predictions.csv',index=False);t=pd.DataFrame(transfer);t.to_csv(P/'frozen_encoder_transfer_predictions.csv',index=False)
results=[dict(scope=scope,mode=mode,n=len(z),subjects=z.group.nunique(),**base.metrics(z.true.to_numpy(),z.pred.to_numpy())) for (scope,mode),z in a.groupby(['scope','mode'])];pd.DataFrame(results).to_csv(P/'frozen_encoder_results.csv',index=False)
pd.DataFrame([dict(scope=scope,mode=mode,**base.metrics(z.true.to_numpy(),z.pred.to_numpy())) for (scope,mode),z in t.groupby(['scope','mode'])]).to_csv(P/'frozen_encoder_transfer.csv',index=False)
def score(cm):
 rec=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(4),where=cm.sum(1)>0);f=np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(4),where=(cm.sum(0)+cm.sum(1))>0);return np.array([rec.mean(),f.mean()])
rng=np.random.default_rng(42);cis=[]
for (scope,mode),z in a.groupby(['scope','mode']):
 for comparator in ['raw',mode]:
  o=old[(old.scope==scope)&(old['mode']==comparator)].sort_values('index');z=z.sort_values('index');assert np.array_equal(z['index'],o['index']);assert np.array_equal(z.true,o.true)
  cz=np.array([confusion_matrix(q.true,q.pred,labels=[0,1,2,3]) for _,q in z.groupby('group',sort=True)]);co=np.array([confusion_matrix(q.true,q.pred,labels=[0,1,2,3]) for _,q in o.groupby('group',sort=True)]);ds=[]
  for _ in range(2000):
   ix=rng.integers(0,len(cz),len(cz));ds.append(score(cz[ix].sum(0))-score(co[ix].sum(0)))
  q=np.quantile(ds,[.025,.975],axis=0);delta=score(cz.sum(0))-score(co.sum(0));cis.append(dict(scope=scope,mode=mode,comparator=comparator,ba_delta=delta[0],ba_low=q[0,0],ba_high=q[1,0],f1_delta=delta[1],f1_low=q[0,1],f1_high=q[1,1]))
pd.DataFrame(cis).to_csv(P/'frozen_encoder_paired_intervals.csv',index=False)
json.dump(dict(encoder='sentence-transformers/all-MiniLM-L6-v2',config_commit=getattr(model[0].auto_model.config,'_commit_hash',None),frozen=True,dimension=int(vec.shape[1]),max_section_tokens=max(lens),encoder_token_limit=model.max_seq_length,pooling='Mean normalized feature-section vectors; normalize final vector',classifier='train-fold StandardScaler + balanced LogisticRegression C=1',folds='identical to previous experiment',unique_sections=len(texts),cache_fingerprint=fingerprint),open(P/'frozen_encoder_run_config.json','w'),indent=2)
print(pd.DataFrame(results).drop(columns='confusion').to_string(index=False),flush=True)
