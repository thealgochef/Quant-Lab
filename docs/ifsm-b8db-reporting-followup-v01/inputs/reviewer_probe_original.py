from __future__ import annotations
import ast,copy,csv,json,math,sys,types
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from dataclasses import dataclass
from collections import Counter,defaultdict
from collections.abc import Mapping,Sequence
from typing import Any
import pandas as pd
ROOT=Path('/mnt/data/b8db_review_work')
S=ROOT/'source/source/quant_lab/src'
P=S/'alpha_lab/agents/data_infra/ifvg/presentation/lab'
V=Path('/mnt/data/IFSM_b8db_Independent_Review_v01/verification')
# Execute supplied function bodies in an isolated namespace; remove project
# imports unavailable without pyarrow. Accounting primitives are independent.
def cents(r,s):
 v=r.get(s+'_cents')
 if v is not None:return int(v)
 v=r.get(s+'_usd')
 return None if v is None else int(Decimal(str(v))*100)
def stamp(v):return pd.Timestamp(v)
def fill_cost_cents(q,m):return int((Decimal(q)*Decimal(m)/10).quantize(Decimal(1),rounding=ROUND_HALF_UP))
ns=dict(__name__='isolated_b8db_reporting_probe',pd=pd,json=json,math=math,
 Mapping=Mapping,Sequence=Sequence,Any=Any,Counter=Counter,defaultdict=defaultdict,
 dataclass=dataclass,Decimal=Decimal,fill_cost_cents=fill_cost_cents,VERSION='ifsm_mffu_reporting_v5',cents=cents,stamp=stamp)
sys.modules[ns['__name__']]=types.ModuleType(ns['__name__'])
for p in [S/'alpha_lab/propsim/funded/reporting_legs.py',P/'mffu_gamma.py']:
 tree=ast.parse(p.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
 exec(compile(tree,str(p),'exec'),ns)
# Real saved whole-target trade, same target receipt emitted by the source's
# funded-to-Core bridge, as witnessed independently in target_decision.
st=ROOT/'result/standard/funded_comparison/funded_comparison_b8db8427cb226061_export_v1'
def typed(r):
 out={}
 for k,v in r.items():
  if v=='':out[k]=None
  elif v in ('true','false'):out[k]=v=='true'
  elif v[0:1] in '{[':out[k]=json.loads(v)
  elif k.endswith(('_cents','_ticks','_ns')) or k in ['quantity','final_exit_quantity','scale_out_quantity','account_number','seq']:out[k]=int(v)
  else:out[k]=v
 return out
r=next(typed(r) for r in csv.DictReader((st/'trades.csv').open()) if r['configuration']=='MCB003')
td=r['target_decision'];snap=td['context']['asof']
receipt={'event':'first_target','configuration':'MCB003','stream':'funded','trade_id':r['trade_ref'],'setup_id':'probe_matching_setup','policy':'gamma_conditional_1r_v1','action':td['action'],'context':td['context']['receipt'],'reasons':[]}
class Index:
 _level_times=[]
 bundle_sha256=snap['bundle_sha256'];table_sha256=snap['table_sha256']
 def snapshot(self,at):
  out=copy.deepcopy(snap);out['decision_time_utc']=at.isoformat();return out
class Study:
 result_id='isolated_saved_trade_probe';configurations=['MCB003'];calendar=['2025-06-16']
 result={'tables':{'trades':[r]},'mffu_batch':{'decision_context':[receipt]}}
g=ns['build_gamma'](Study(),Index())
t=g['trades'][0]
assert t['first_target_policy_receipt'] is not None
row=ns['selected_rows'](g,'MCB003','funded','first_1R_checkpoint')[0]
answer={'scope':'Exact supplied reporting bodies on one actual saved trade with the matching first-target receipt shape; not full application or economic replay',
 'trade_id':r['trade_ref'],'actual_fill_ns':td['decision_ns'],'receipt_timestamp':receipt['context']['decision_ts_utc'],
 'checkpoint_timestamp':t['first_checkpoint_utc'],'difference_ns':stamp(t['first_checkpoint_utc']).value-stamp(receipt['context']['decision_ts_utc']).value,
 'matching_receipt_retained':True,'computed_checkpoint_role':t['first_checkpoint_context_role'],'selected_row_context_role':row['context_role'],
 'coverage':ns['coverage']([row]),'entry_context_role':t['context_role'],
 'checkpoint_gamma_matches_saved_target_gamma':t['first_checkpoint_snapshot']['gamma']==snap['gamma']}
# Changing neither event identity nor values, a microsecond-exact synthetic
# timestamp gives the contrasting executed label (precision sensitivity).
r2=copy.deepcopy(r);r2['exit_utc']=receipt['context']['decision_ts_utc']
Study.result={'tables':{'trades':[r2]},'mffu_batch':{'decision_context':[receipt]}}
t2=ns['build_gamma'](Study(),Index())['trades'][0]
answer['synthetic_microsecond_exact_role']=t2['first_checkpoint_context_role']
assert answer['computed_checkpoint_role']=='reporting_annotation_at_actual_funded_fill'
assert answer['synthetic_microsecond_exact_role']=='executed_saved_decision_receipt'
# Population-level saved output discrepancy (no extrapolation of probe).
rows=list(csv.DictReader((ROOT/'result/integrated/reporting/gamma_checkpoints.csv').open()))
target_trades=[typed(x) for x in csv.DictReader((st/'trades.csv').open()) if x['target_decision']]
cp={(x['configuration_id'],x['population'],x['trade_ref']):x for x in rows if x['basis']=='first_1R_checkpoint'}
roles=Counter(cp[(x['configuration'],'funded',x['trade_ref'])]['context_role'] for x in target_trades)
precision=Counter(stamp(x['exit_utc'] if not x['scale_out_quantity'] else x['scale_out_utc']).value-x['target_decision']['decision_ns'] for x in target_trades)
answer['funded_trades_with_saved_target_decision']=len(target_trades)
answer['their_exported_checkpoint_roles']=dict(roles)
answer['exact_target_observation_and_checkpoint_ns_difference_counts']=dict(precision)
(V/'CHECKPOINT_PROVENANCE_PROBE.json').write_text(json.dumps(answer,indent=2,default=str))
print(json.dumps(answer,indent=2,default=str))
