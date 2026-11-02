from pathlib import Path
import json,collections
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score,f1_score
P=Path('analysis');b=pd.read_csv(P/'clean_baselines.csv');p=pd.read_csv(P/'clean_predictions.csv');rng=np.random.default_rng(42);out=[]
for (scope,feat,model),a in p.groupby(['scope','features','model']):
 if feat not in ['raw_coverage','passive_raw_coverage','provided_percentiles']:continue
 gs=[v for _,v in a.groupby('group')];stats=[]
 for _ in range(500):
  z=pd.concat([gs[i] for i in rng.integers(0,len(gs),len(gs))]);stats.append([balanced_accuracy_score(z.true,z.pred),f1_score(z.true,z.pred,labels=[0,1,2,3],average='macro',zero_division=0)])
 ci=np.quantile(stats,[.025,.975],axis=0);out.append(dict(scope=scope,features=feat,model=model,balanced_accuracy_lo=ci[0,0],balanced_accuracy_hi=ci[1,0],macro_f1_lo=ci[0,1],macro_f1_hi=ci[1,1]))
pd.DataFrame(out).to_csv(P/'cluster_bootstrap_intervals.csv',index=False)
m=pd.read_csv(P/'record_index.csv');fig,ax=plt.subplots(1,2,figsize=(13,4.7));ct=pd.crosstab(m.source,m['class']).reindex(['globem','ced','hrv','lifesnaps']);ct.columns=['Normal','Mild','Moderate','Severe'];ct.plot.bar(stacked=True,ax=ax[0],color=['#77bba5','#f0cf73','#e8a064','#bb646c']);ax[0].set_title('Records by source and supplied severity');ax[0].set_ylabel('Records');ax[0].tick_params(axis='x',rotation=0)
z=b[(b.scope=='tier1_clean')&((b.features.isin(['raw','provided_percentiles','coverage']) )|((b.features=='raw_coverage'))| (b.model=='majority'))];labels=[f'{r.features.replace("_"," ")} ({r.model})' for _,r in z.iterrows()];ax[1].barh(labels,z.balanced_accuracy,color='#477c9d');ax[1].axvline(.25,color='gray',linestyle='--');ax[1].set_xlim(0,.5);ax[1].set_title('Clean Tier-1: grouped balanced accuracy');ax[1].set_xlabel('Five-fold out-of-fold estimate');fig.tight_layout();fig.savefig(P/'audit_overview.png',dpi=180)
