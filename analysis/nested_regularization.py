"""Nested grouped regularization comparison; qualitative cutoffs refit inside inner folds."""
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
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import semantic_experiment as base
warnings.filterwarnings('error',category=ConvergenceWarning)
P=Path('analysis');GRID=[.0001,.001,.01,.1,1.,10.];MODES=['raw','numeric_text','qualitative_text'];tasks=[];pairs=[];sections=set();torch.set_num_threads(4)
def chunks(t):
 s=re.split(r'(?<=\.)\s+',t.strip());assert len(s)%5==0;return [' '.join(s[k:k+5]) for k in range(0,len(s),5)]
def pair(tr,te,features):
 assert not set(base.m.iloc[tr].group)&set(base.m.iloc[te].group)
 cols=[c for c in base.X if c.split('__')[0] in features];A,B=base.represent(base.X.iloc[tr][cols],base.X.iloc[te][cols],features)
 out={'tr':tr,'te':te,'raw':(A['raw'].to_numpy(),B['raw'].to_numpy())}
 for mode in MODES[1:]:
  a=[chunks(t) for t in A[mode]];b=[chunks(t) for t in B[mode]]
  for cs in a+b:sections.update(cs)
  out[mode]=(a,b)
 pairs.append(out);return len(pairs)-1
def task(scope,fold,tr,te,features,transfer=False):
 outer=pair(tr,te,features);inner=[];y=base.m.iloc[tr]['class'].to_numpy();g=base.m.iloc[tr].group.to_numpy()
 for it,iv in StratifiedGroupKFold(3,shuffle=True,random_state=117).split(tr,y,g):inner.append(pair(tr[it],tr[iv],features))
 tasks.append(dict(scope=scope,fold=fold,outer=outer,inner=inner,transfer=transfer))
old=pd.read_csv(P/'bottleneck_predictions.csv')
for scope in ['tier1_clean','globem_tier1_clean','ced_clean']:
 ix=np.where(base.good&(base.m.source.eq(scope.split('_')[0]) if scope!='tier1_clean' else True))[0];y=base.m.iloc[ix]['class'];g=base.m.iloc[ix].group
 for fold,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ix,y,g)):
  prev=old[(old.scope==scope)&(old['mode']=='raw')].set_index('index');assert all(prev.loc[ix[te],'fold']==fold)
  task(scope,fold,ix[tr],ix[te],base.features)
for a,b in [('globem','ced'),('ced','globem')]:task(a+'_to_'+b,-1,np.where(base.good&base.m.source.eq(a))[0],np.where(base.good&base.m.source.eq(b))[0],['sleep_hours','home_time_min','screen_total_min'],True)
texts=sorted(sections);print('Prepared nested splits; unique sections:',len(texts),flush=True)
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',local_files_only=True,device='cpu');model.eval()
for p in model.parameters():p.requires_grad_(False)
lengths=[len(x) for x in model.tokenizer(texts,truncation=False)['input_ids']];assert max(lengths)<=model.max_seq_length
with torch.inference_mode():vectors=model.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False)
assert np.isfinite(vectors).all();lookup={t:i for i,t in enumerate(texts)}
def embed(cs):
 z=np.array([vectors[[lookup[t] for t in c]].mean(0) for c in cs]);return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
# Only frozen embeddings are shared across splits; imputation/scaling fit on each split's training subset.
for p in pairs:
 rawA,rawB=p['raw']
 for mode in MODES:
  A,B=(rawA,rawB) if mode=='raw' else (np.concatenate([rawA,embed(p[mode][0])],axis=1),np.concatenate([rawB,embed(p[mode][1])],axis=1))
  imp=SimpleImputer(strategy='median',keep_empty_features=True);A=imp.fit_transform(A);B=imp.transform(B);sc=StandardScaler();p[mode]=(sc.fit_transform(A),sc.transform(B))
print('Encoding and split-specific preprocessing complete.',flush=True)
rows=[];trows=[];search=[];selected=[]
for t in tasks:
 outer=pairs[t['outer']]
 for mode in MODES:
  scores=[]
  for C in GRID:
   fold_scores=[]
   for k in t['inner']:
    p=pairs[k];A,B=p[mode];clf=LogisticRegression(C=C,class_weight='balanced',max_iter=3000);clf.fit(A,base.m.iloc[p['tr']]['class']);pred=clf.predict(B);fold_scores.append(f1_score(base.m.iloc[p['te']]['class'],pred,labels=[0,1,2,3],average='macro',zero_division=0))
   val=float(np.mean(fold_scores));scores.append(val);search.append(dict(scope=t['scope'],fold=t['fold'],mode=mode,C=C,mean_inner_macro_f1=val))
  C=GRID[int(np.argmax(scores))];selected.append(dict(scope=t['scope'],fold=t['fold'],mode=mode,C=C,inner_macro_f1=max(scores)))
  A,B=outer[mode];clf=LogisticRegression(C=C,class_weight='balanced',max_iter=3000);clf.fit(A,base.m.iloc[outer['tr']]['class']);pred=clf.predict(B)
  for i,q in zip(outer['te'],pred):(trows if t['transfer'] else rows).append(dict(scope=t['scope'],fold=t['fold'],mode=mode,C=C,index=int(i),group=base.m.iloc[i].group,true=int(base.m.iloc[i]['class']),pred=int(q)))
 print('Finished',t['scope'],'fold',t['fold'],flush=True)
a=pd.DataFrame(rows);b=pd.DataFrame(trows)
a.to_csv(P/'nested_predictions.csv',index=False);b.to_csv(P/'nested_transfer_predictions.csv',index=False);pd.DataFrame(search).to_csv(P/'nested_search.csv',index=False);pd.DataFrame(selected).to_csv(P/'nested_selected.csv',index=False)
for data,name in [(a,'nested_results.csv'),(b,'nested_transfer.csv')]:
 res=[dict(scope=s,mode=m,n=len(z),subjects=z.group.nunique(),**base.metrics(z.true.to_numpy(),z.pred.to_numpy())) for (s,m),z in data.groupby(['scope','mode'])];pd.DataFrame(res).to_csv(P/name,index=False)
def score(cm):
 return np.array([np.mean(np.divide(np.diag(cm),cm.sum(1),out=np.zeros(4),where=cm.sum(1)>0)),np.mean(np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(4),where=(cm.sum(0)+cm.sum(1))>0))])
rng=np.random.default_rng(42);cis=[]
for s in a.scope.unique():
 raw=a[(a.scope==s)&(a['mode']=='raw')].sort_values('index');cr=np.array([confusion_matrix(z.true,z.pred,labels=[0,1,2,3]) for _,z in raw.groupby('group',sort=True)])
 for mode in MODES[1:]:
  q=a[(a.scope==s)&(a['mode']==mode)].sort_values('index');assert list(q['index'])==list(raw['index']);cm=np.array([confusion_matrix(z.true,z.pred,labels=[0,1,2,3]) for _,z in q.groupby('group',sort=True)]);d=[]
  for _ in range(2000):
   ix=rng.integers(0,len(cr),len(cr));d.append(score(cm[ix].sum(0))-score(cr[ix].sum(0)))
  ci=np.quantile(d,[.025,.975],axis=0);delta=score(cm.sum(0))-score(cr.sum(0));cis.append(dict(scope=s,mode=mode,ba_delta=delta[0],ba_low=ci[0,0],ba_high=ci[1,0],f1_delta=delta[1],f1_low=ci[0,1],f1_high=ci[1,1]))
pd.DataFrame(cis).to_csv(P/'nested_paired_intervals.csv',index=False)
json.dump(dict(C_grid=GRID,selection='mean inner 3-fold subject-grouped macro-F1; ties prefer smaller C',outer='same five subject-grouped folds, seed42',inner_seed=117,encoder='frozen all-MiniLM-L6-v2',preprocessing='cutoffs, medians and scaling fitted within each inner/outer training split',max_section_tokens=max(lengths),note='Adaptive exploratory comparison on repeatedly used outer folds; not a final untouched test'),open(P/'nested_config.json','w'),indent=2)
print(pd.read_csv(P/'nested_results.csv').drop(columns='confusion').round(4).to_string(index=False),flush=True)
