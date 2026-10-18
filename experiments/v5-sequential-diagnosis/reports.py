"""Learning curves, complementary capacity, and fixed routing comparisons."""
from common import *
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from sklearn.linear_model import Ridge
from spec_research.common import tokenizer


def tables_for(pool,step,stage,splits,binding):
    methods=[pool] if pool in ('pooled','weighted') else [f'{pool}_{i}' for i in range(5)]
    tables=[]
    for method in methods:
        tables.append(matrix(splits[stage],f'{method}_s{step}',stage,EXP/'adapters'/method/f's{step}',binding))
    return tables


def describe(rows,tables,old,fixed=None):
    info=pool_stats(rows,tables,fixed)
    keys,a=values(rows,tables);_,b=values(rows,[old])
    info['per_expert_changed_acceptance_windows']=[int(sum(a[:,i]!=b[:,0])) for i in range(a.shape[1])]
    info['per_expert_better_windows']=[int(sum(a[:,i]>b[:,0])) for i in range(a.shape[1])]
    info['per_expert_worse_windows']=[int(sum(a[:,i]<b[:,0])) for i in range(a.shape[1])]
    if a.shape[1]>1:
        info['removal_losses']=[mean(aggregate(keys,a.max(axis=1)-np.delete(a,i,axis=1).max(axis=1)).values()) for i in range(a.shape[1])]
    info['suffix_limit']='Proposals stop after first rejection; full rejected suffix diversity is not measured.'
    return info


def stage1_curves(splits,binding):
    curves={};selection={}
    old=load_matrix(splits['val'],'old_mix16','val',binding)
    for pool in (*POOLS,'pooled','weighted'):
        steps=(10,80,160) if pool in ('pooled','weighted') else (2,16,32)
        curves[pool]={}
        for step in steps:
            table=tables_for(pool,step,'val',splits,binding)
            info=describe(splits['val'],table,old)
            curves[pool][str(step)]=info
            atomic(OUT/'stage1_curves.partial.json',curves)
            status('stage1_curve',pool=pool,step=step,oracle=info['prefix_oracle'],different=info['different_acceptance_windows'],windows=info['windows'])
        chosen=max(steps,key=lambda s:(curves[pool][str(s)]['prefix_oracle'],-s))
        selection[pool]={'step':chosen,'fixed':curves[pool][str(chosen)]['fixed_index'],
                         'reason':'validation prefix Oracle, earlier checkpoint on ties'}
    atomic(OUT/'stage1_curves.json',curves);atomic(OUT/'selection.json',selection)
    return selection


def paired_field(first,second,field1='prefix_oracle',field2='prefix_oracle'):
    ids=sorted(first['per_request']);assert ids==sorted(second['per_request'])
    return paired([first['per_request'][r][field1] for r in ids],[second['per_request'][r][field2] for r in ids])


def stage2(splits,binding):
    selection=read(OUT/'selection.json');summaries={}
    old=load_matrix(splits['eval'],'old_mix16','eval',binding)
    for pool in (*POOLS,'pooled','weighted'):
        tables=tables_for(pool,selection[pool]['step'],'eval',splits,binding)
        summaries[pool]=describe(splits['eval'],tables,old,selection[pool]['fixed'])
    for m in ('mix16','mix80'):
        table=load_matrix(splits['eval'],'old_'+m,'eval',binding)
        summaries['old_'+m]=describe(splits['eval'],[table],old,0)
    comparisons={f'mismatch_oracle_minus_{m}':paired_field(summaries['mismatch'],summaries[m]) for m in ('semantic','random','pooled','weighted','old_mix80','old_mix16')}
    result={'stage':2,'selection':selection,'pools':summaries,'comparisons':comparisons,
            'complementarity_gate':all(x['ci95'][0]>0 for x in comparisons.values()),
            'scope':'120 reused development eval requests; Oracle is retrospective, not routed speed'}
    atomic(OUT/'stage2.json',result)
    status('stage2_complete',gate=result['complementarity_gate'],mismatch_oracle=summaries['mismatch']['prefix_oracle'])
    return result


def route_items(rows,tables):
    keys,a=values(rows,tables);byid={r['id']:r for r in rows}
    return keys,a,[(byid[rid],p) for rid,p in keys]


def stage3(splits,binding):
    selection=read(OUT/'selection.json');stage2_report=read(OUT/'stage2.json')
    tok=tokenizer();output={};predictions={}
    for pool in POOLS:
        step=selection[pool]['step']
        train_tables=tables_for(pool,step,'route',splits,binding)
        eval_tables=[load_matrix(splits['eval'],f'{pool}_{i}_s{step}','eval',binding) for i in range(5)]
        keys,y,items=route_items(splits['route'],train_tables)
        ekeys,ey,eitems=route_items(splits['eval'],eval_tables)
        def tokens(item):
            row,p=item
            return 'p'+str(p)+' '+' '.join('t'+str(t) for t in row['prompt_ids']+row['response_ids'][:p])
        def text(item):
            row,p=item
            return 'position='+str(p)+' '+tok.decode(row['prompt_ids']+row['response_ids'][:p],skip_special_tokens=False)
        informative=np.ptp(y,axis=1)>0
        prior=y.mean(axis=0)
        results={};details={}
        for name in ('hash','char','char_weighted'):
            if name=='hash':
                vec=HashingVectorizer(n_features=4096,alternate_sign=False,ngram_range=(1,2),lowercase=False)
                x=vec.transform([tokens(t) for t in items]);ex=vec.transform([tokens(t) for t in eitems]);alpha=10.
            else:
                vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=12000,lowercase=False)
                x=vec.fit_transform([text(t) for t in items]);ex=vec.transform([text(t) for t in eitems]);alpha=1.
            weights=np.where(informative,5.,1.) if name=='char_weighted' else None
            model=Ridge(alpha=alpha,solver='lsqr').fit(x,y,sample_weight=weights)
            start=time.perf_counter();pred=model.predict(ex);ms=(time.perf_counter()-start)*1000
            choice=pred.argmax(axis=1);actual=ey[np.arange(len(ey)),choice]
            scores=aggregate(ekeys,actual)
            results[name]={'mean':mean(scores.values()),'per_request':scores,
                           'selected_counts':dict(collections.Counter(map(int,choice))),
                           'prediction_ms_excluding_feature_extraction':ms,
                           'oracle_gap':paired([stage2_report['pools'][pool]['per_request'][r]['prefix_oracle'] for r in sorted(scores)],
                                               [scores[r] for r in sorted(scores)]),
                           'gain_over_val_selected_fixed':paired([scores[r] for r in sorted(scores)],
                                    [stage2_report['pools'][pool]['per_request'][r]['fixed'] for r in sorted(scores)])}
            details[name]=[{'id':rid,'position':p,'choice':int(c),'scores':s.tolist(),'accepted':float(v)}
                           for (rid,p),c,s,v in zip(ekeys,choice,pred,actual)]
        output[pool]={'training_requests':len({k[0] for k in keys}),'training_windows':len(keys),
                      'informative_windows':int(informative.sum()),
                      'informative_requests':len({keys[i][0] for i in range(len(keys)) if informative[i]}),
                      'mean_acceptance_prior':prior.tolist(),'routers':results}
        predictions[pool]=details
        atomic(OUT/'stage3.partial.json',output)
        status('stage3_pool',pool=pool,informative=int(informative.sum()),windows=len(keys),weighted_mean=results['char_weighted']['mean'])
    candidate=output['mismatch']['routers']['char_weighted']['per_request'];ids=sorted(candidate)
    comparisons={}
    for p in POOLS:
        ref=output[p]['routers']['char_weighted']['per_request']
        comparisons['mismatch_char_weighted_minus_'+p]=paired([candidate[r] for r in ids],[ref[r] for r in ids])
    for m in ('old_mix80','old_mix16','pooled','weighted'):
        ref=stage2_report['pools'][m]['per_request']
        comparisons['mismatch_char_weighted_minus_'+m]=paired([candidate[r] for r in ids],[ref[r]['fixed'] for r in ids])
    result={'stage':3,'pools':output,'comparisons':comparisons,
            'primary':'mismatch char_weighted, specified before evaluation',
            'routing_gate':all(v['ci95'][0]>0 for k,v in comparisons.items() if not k.endswith('_mismatch')),
            'scope':'public-prefix replay; no online switching or net-speed claim'}
    atomic(OUT/'stage3_predictions.json',predictions);atomic(OUT/'stage3.json',result)
    status('stage3_complete',gate=result['routing_gate'],metric=comparisons['mismatch_char_weighted_minus_old_mix80']['mean'])
    return result


def stage4(splits,binding):
    oldval=load_matrix(splits['val'],'old_mix16','val',binding)
    oldeval=load_matrix(splits['eval'],'old_mix16','eval',binding)
    output={}
    for pool in ('mismatch','rank32','wide'):
        val=tables_for(pool,32,'val',splits,binding)
        cv=describe(splits['val'],val,oldval)
        evaluation=tables_for(pool,32,'eval',splits,binding)
        ev=describe(splits['eval'],evaluation,oldeval,cv['fixed_index'])
        training=[read(EXP/'adapters'/f'{pool}_{i}'/'complete.json') for i in range(5)]
        output[pool]={'val':cv,'eval':ev,'active_adapter_parameters':training[0]['trainable_parameters'],
                      'pool_adapter_parameters':sum(t['trainable_parameters'] for t in training),
                      'training_seconds':sum(t['seconds'] for t in training),'training_peak_gib':max(t['peak_gib'] for t in training),
                      'initialization_proposal_matches':[t['initialization']['initial_proposal_matches_mix16'] for t in training]}
        status('stage4_pool',pool=pool,oracle=ev['prefix_oracle'],fixed=ev['fixed'],different=ev['different_acceptance_windows'])
    comparisons={p+'_oracle_minus_rank16_qv':paired_field(output[p]['eval'],output['mismatch']['eval']) for p in ('rank32','wide')}
    result={'stage':4,'pools':output,'comparisons':comparisons,
            'capacity_gate':{p:comparisons[p+'_oracle_minus_rank16_qv']['ci95'][0]>0 for p in ('rank32','wide')},
            'scope':'matched 32-step/640-request capacity diagnostic; no router trained for new structures'}
    atomic(OUT/'stage4.json',result)
    return result
