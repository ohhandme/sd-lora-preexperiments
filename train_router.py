"""Frozen CPU TF-IDF routers; labels must be router_train acceptance evidence."""
import argparse,pickle,time
from spec_research.common import *
from spec_research.data import partition
if __name__=='__main__':
    assert_spec();p=argparse.ArgumentParser();p.add_argument('--kind',choices=['task','acceptance'],required=True);p.add_argument('--labels');p.add_argument('--output',required=True);a=p.parse_args()
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train=[r for d in DOMAINS for r in partition(d,'router_train')];val=[r for d in DOMAINS for r in partition(d,'router_val')]
    labels={}
    if a.kind=='acceptance':
        assert a.labels,'Real per-request expert acceptance labels required'
        for r in rows(a.labels):
            assert r['partition'] in ('router_train','router_val') and r['L']==8
            scores=r['expert_mean_a'];best=max(scores.values());tied=[d for d in DOMAINS if scores[d]==best]
            labels[r['id']]=r['domain'] if r['domain'] in tied else tied[0]
        assert all(r['id'] in labels for r in train+val)
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=20000,min_df=2,sublinear_tf=True),LogisticRegression(C=1.,max_iter=1000,class_weight='balanced',random_state=42))
    y=[r['domain'] if a.kind=='task' else labels[r['id']] for r in train];model.fit([r['prompt'] for r in train],y)
    results=[]
    for r in val:
        t=time.perf_counter();pred=model.predict([r['prompt']])[0];ms=(time.perf_counter()-t)*1000
        entry={'id':r['id'],'domain':r['domain'],'predicted':pred,'feature_and_classification_ms':ms,'expert_activation_included':False,'domain_correct':pred==r['domain']}
        if a.kind=='acceptance':entry['best_expert_correct']=pred==labels[r['id']]
        results.append(entry)
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'router.pkl').open('xb') as f:pickle.dump(model,f)
    atomic(out/'validation.json',{'kind':a.kind,'requests':results,'training_ids':[r['id'] for r in train],'provenance':provenance(),'note':'CPU classifier timing only; online routing must add real resident adapter activation.'})
