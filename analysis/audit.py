import json,collections,hashlib,warnings,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,mean_absolute_error,confusion_matrix
warnings.filterwarnings('ignore')
P=Path(__file__).parent
src=Path(sys.argv[1]); data=json.load(open(src)); fs=sorted({k for r in data for k,v in r.items() if isinstance(v,dict) and 'D1' in v})
m=pd.DataFrame([{k:r.get(k) for k in ['sample_id','source','tier','subject_id','window','phq4_score','covid_era','quality_tier','granularity']} for r in data]);m['class']=np.minimum(m.phq4_score//3,3).astype(int);m['group']=m.source+':'+m.subject_id;m['aug']=m.sample_id.str.contains('aug')
m.to_csv(P/'record_index.csv',index=False)
summary={'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'n':len(m),'features':fs,'composition':m.groupby(['source','tier','quality_tier'],dropna=False).size().to_dict().__str__(),'scores':m.phq4_score.value_counts().sort_index().to_dict(),'classes':pd.crosstab(m.source,m['class']).to_dict(),'covid':pd.crosstab(m.source,m.covid_era.fillna('missing')).to_dict()}
rows=[]; raw=[];cov=[];pct=[];meta=[]; flags=[]
for r in data:
 a={};b={};c={};e={}
 for f in fs:
  v=r.get(f); vs=[v.get('D'+str(i)) for i in range(1,8)] if isinstance(v,dict) else [None]*7;x=np.array([z for z in vs if isinstance(z,(int,float)) and np.isfinite(z)])
  for stat,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:a[f+'_'+stat]=float(fn(x)) if len(x) else np.nan
  b[f+'_n']=len(x);c[f+'_pct']=r.get(f+'__pctile',np.nan);e[f+'_refn']=r.get(f+'__pctile_n',np.nan)
  rd=r.get(f+'__range_days') or {}
  for k in ['within','below','above','unknown']:e[f+'_'+k]=rd.get(k,np.nan)
  if isinstance(v,dict):
   rows.append({'source':r['source'],'feature':f,'n':len(x),'values':x.tolist(),'flag':r.get(f+'__flag'),'range':str(r.get(f+'__range')),'pct':r.get(f+'__pctile'),'refn':r.get(f+'__pctile_n')})
 raw.append(a);cov.append(b);pct.append(c);meta.append(e)
coverage=[]
for s in sorted(m.source.unique()):
 for f in fs:
  rr=[r for r in rows if r['source']==s and r['feature']==f];n=int((m.source==s).sum());x=np.array([v for r in rr for v in r['values']]);ns=[r['n'] for r in rr]
  coverage.append(dict(source=s,feature=f,rows=n,structured_rows=len(rr),observed_days=len(x),daily_coverage=len(x)/(n*7),no_observed_rows=n-sum(z>0 for z in ns),sparse_1_2_rows=sum(0<z<3 for z in ns),minimum=float(x.min()) if len(x) else None,p01=float(np.quantile(x,.01)) if len(x) else None,median=float(np.median(x)) if len(x) else None,p99=float(np.quantile(x,.99)) if len(x) else None,maximum=float(x.max()) if len(x) else None,zeros=int(sum(x==0)),negative=int(sum(x<0)),ranges=str(collections.Counter(r['range'] for r in rr)),flags=str(collections.Counter(str(r['flag']) for r in rr))))
pd.DataFrame(coverage).to_csv(P/'feature_source_audit.csv',index=False)
sub=m.groupby('group').agg(source=('source','first'),windows=('window','size'),classes=('class','nunique'),scores=('phq4_score','nunique'));sub.to_csv(P/'subjects.csv');summary['subjects']=sub.groupby('source').agg(subjects=('windows','size'),repeated=('windows',lambda x:sum(x>1)),min=('windows','min'),median=('windows','median'),max=('windows','max'),changing_class=('classes',lambda x:sum(x>1))).to_dict()
summary['duplicates']={}
for name,keys in [('subject_window',['source','subject_id','window']),('sample_id',['sample_id'])]:summary['duplicates'][name]={'extra':int(m.duplicated(keys).sum()),'involved':int(m.duplicated(keys,keep=False).sum())}
for name,fn in [('full',lambda r:r),('without_sampleid',lambda r:{k:v for k,v in r.items() if k!='sample_id'}),('daily_features',lambda r:{f:r.get(f) for f in fs})]:
 hs=[json.dumps(fn(r),sort_keys=True) for r in data];co=collections.Counter(hs);summary['duplicates'][name]={'extra':sum(n-1 for n in co.values()),'involved':sum(n for n in co.values() if n>1)}
summary['conflicting_window_labels']=int((m.groupby(['source','subject_id','window']).phq4_score.nunique()>1).sum())
summary['augmentation']=pd.crosstab([m.source,m.aug],m['class']).to_string()
tr,te=train_test_split(np.arange(len(m)),test_size=.2,random_state=42,stratify=m['class']);summary['random_split_test_seen_subject_fraction']=float(m.iloc[te].group.isin(m.iloc[tr].group).mean())
summary['all_features_unobserved']=int((pd.DataFrame(cov).sum(axis=1)==0).sum())
json.dump(summary,open(P/'summary.json','w'),indent=2,default=str)
print(json.dumps(summary,indent=2),flush=True)
# Deduplicate subject-window, first; all occurrences have same group regardless.
keep=~m.duplicated(['source','subject_id','window']); frames={k:pd.DataFrame(v).astype(float) for k,v in [('raw',raw),('coverage',cov),('percentile',pct),('metadata',meta)]}
frames['raw_coverage']=pd.concat([frames['raw'],frames['coverage']],axis=1);frames['pct_coverage']=pd.concat([frames['percentile'],frames['coverage']],axis=1);frames['raw_no_selfreport']=frames['raw_coverage'].drop(columns=[c for c in frames['raw_coverage'] if c.startswith(('stress_','social_level_'))]);frames['source_era']=pd.get_dummies(m[['source','covid_era']].fillna('missing')).astype(float)
results=[];preds=[]
def metrics(y,p):
 sev=y==3
 return dict(accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),macro_f1=f1_score(y,p,labels=[0,1,2,3],average='macro',zero_division=0),ordinal_mae=mean_absolute_error(y,p),severe_recall=float(np.mean(p[sev]==3)) if sev.any() else None,critical_miss=float(np.mean(p[sev]<=1)) if sev.any() else None,confusion=confusion_matrix(y,p,labels=[0,1,2,3]).tolist())
for scope in ['tier1','globem','ced','all_exploratory']:
 idx=np.where(keep & (m.tier.eq(1) if scope=='tier1' else m.source.eq(scope) if scope in ['globem','ced'] else True))[0];y=m.iloc[idx]['class'].to_numpy();g=m.iloc[idx].group.to_numpy();splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(idx,y,g))
 for feat in ['raw','coverage','percentile','raw_coverage','pct_coverage','metadata','raw_no_selfreport','source_era']:
  X=frames[feat].iloc[idx]; X=X.loc[:,X.notna().any()]
  for model in ['logistic','forest'] if feat in ['raw','raw_coverage','pct_coverage','raw_no_selfreport'] else ['logistic']:
   p=np.zeros(len(y),int)
   for fold,(tr,te) in enumerate(splits):
    assert not set(g[tr])&set(g[te]);clf=LogisticRegression(C=1,max_iter=2000,class_weight='balanced') if model=='logistic' else RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=42)
    pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),clf);pipe.fit(X.iloc[tr],y[tr]);p[te]=pipe.predict(X.iloc[te])
   out=dict(scope=scope,features=feat,model=model,n=len(y),subjects=len(set(g)),**metrics(y,p));results.append(out);print(json.dumps(out),flush=True)
   preds.extend(dict(scope=scope,features=feat,model=model,index=int(i),true=int(t),pred=int(q)) for i,t,q in zip(idx,y,p))
  # no tuning; OOF for all five folds
 p=np.zeros(len(y),int)
 for tr,te in splits:p[te]=collections.Counter(y[tr]).most_common(1)[0][0]
 results.append(dict(scope=scope,features='none',model='majority',n=len(y),subjects=len(set(g)),**metrics(y,p)))
pd.DataFrame(results).to_csv(P/'baselines.csv',index=False);pd.DataFrame(preds).to_csv(P/'predictions.csv',index=False)
