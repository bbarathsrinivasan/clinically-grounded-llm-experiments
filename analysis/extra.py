import os
import json,collections,datetime
import pandas as pd,numpy as np
from pathlib import Path
P=Path('analysis');d=json.load(open(os.environ.get('CLINICAL_DATASET', 'data/anchored_with_phq4_only_aug.json')));fs=json.load(open(P/'summary.json'))['features'];out={};issues=[];ident=collections.defaultdict(list);stats=collections.Counter();refs=[]
for r in d:
 ident[json.dumps({f:r.get(f) for f in fs},sort_keys=True)].append(r['sample_id'])
 for f in fs:
  v=r.get(f)
  if not isinstance(v,dict):continue
  x=[z for z in v.values() if z is not None];n=len(x);flag=r.get(f+'__flag');rd=r.get(f+'__range_days');bounds=r.get(f+'__range');pct=r.get(f+'__pctile');ref=r.get(f+'__pctile_n')
  if n in [1,2] and not str(flag).startswith('sparse'):stats['sparse_without_sparse_flag_'+r['source']]+=1
  if not n and flag not in ['no_observed_days','all_missing']:stats['empty_without_flag']+=1
  if rd and bounds:
   lo,hi=bounds;expected={'within':sum(lo<=z<=hi for z in x),'below':sum(z<lo for z in x),'above':sum(z>hi for z in x),'unknown':7-n}
   if rd!=expected:issues.append([r['sample_id'],f,rd,expected])
  if pct is not None and not 0<=pct<=100:stats['bad_pct']+=1
  if ref is not None:refs.append(dict(source=r['source'],window=r['window'],feature=f,n=ref))
out['flags']=dict(stats);out['range_day_mismatches']=issues;out['identical_daily_feature_groups']=[v for v in ident.values() if len(v)>1];ref=pd.DataFrame(refs);out['ref_n']=ref.groupby('source').n.agg(['min','median','max','nunique']).to_dict();out['ref_n_within_source_window_feature_varies']=int((ref.groupby(['source','window','feature']).n.nunique()>1).sum());out['range_constants']={f:sorted(set(str(r.get(f+'__range')) for r in d)) for f in fs}
pd.crosstab([pd.DataFrame(d).source,pd.DataFrame(d).tier],pd.DataFrame(d).covid_era).to_csv(P/'era_tier.csv');json.dump(out,open(P/'extra.json','w'),indent=2);print(json.dumps(out,indent=2))
