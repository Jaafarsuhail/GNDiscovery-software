"""GNMIC Prediction Support.

Rebuilt for GNMIC-EC1792 at the fixed P(active) threshold 0.510.
The primary GNMIC probability is never modified by this module.
"""
from __future__ import annotations
import math
import numpy as np
import joblib

BASE_FEATURES = [
    "predicted_class_confidence", "directed_threshold_margin", "max_similarity",
    "directed_class_similarity_difference", "weighted_agreement_k10_g4",
    "effective_neighbours_k10_g4", "predicted_class",
]

def _logit(x, eps=1e-5):
    x=float(np.clip(x,eps,1-eps))
    return math.log(x/(1-x))

def support_basis(x7):
    X=np.asarray(x7,dtype=float).reshape(-1,7)
    c,m,s,d,a,e,pr=[X[:,i] for i in range(7)]
    eps=1e-5
    return np.column_stack([
        c,m,s,d,a,e,pr,
        c*c,m*m,s*s,d*d,a*a,e*e,
        c*a,c*s,c*d,c*e,m*a,m*s,a*s,a*d,a*e,s*d,s*e,d*e,
        pr*c,pr*m,pr*s,pr*d,pr*a,pr*e,
        np.log(np.clip(c,eps,1-eps)/(1-np.clip(c,eps,1-eps))),
        np.log(np.clip(a,eps,1-eps)/(1-np.clip(a,eps,1-eps))),
        np.sqrt(np.clip(m,0,None)),np.sqrt(np.clip(s,0,None)),np.log1p(e)
    ])

def neighbour_metrics(similarities, labels, predicted_class, k=10, gamma=4.0):
    s=np.asarray(similarities,dtype=float)
    y=np.asarray(labels,dtype=int)
    if s.size != y.size:
        raise ValueError("similarities and labels must have the same length")
    if s.size==0:
        return dict(max_similarity=0.0, nearest_active_similarity=0.0, nearest_inactive_similarity=0.0,
                    directed_class_similarity_difference=0.0, weighted_active_fraction=0.5,
                    weighted_agreement_k10_g4=0.5, effective_neighbours_k10_g4=0.0)
    order=np.argsort(-s)
    s=s[order]; y=y[order]
    max_similarity=float(s[0])
    sa=float(s[y==1].max()) if np.any(y==1) else 0.0
    si=float(s[y==0].max()) if np.any(y==0) else 0.0
    raw=sa-si
    directed=float(raw if int(predicted_class)==1 else -raw)
    kk=min(int(k),len(s))
    sk=s[:kk]; yk=y[:kk]
    w=np.power(sk,float(gamma))
    sw=float(w.sum())
    if sw<=0:
        weighted_active=0.5; eff=0.0
    else:
        weighted_active=float(np.dot(w,yk)/sw)
        sw2=float(np.dot(w,w))
        eff=float(sw*sw/sw2) if sw2>0 else 0.0
    agreement=float(weighted_active if int(predicted_class)==1 else 1.0-weighted_active)
    return dict(max_similarity=max_similarity, nearest_active_similarity=sa, nearest_inactive_similarity=si,
                directed_class_similarity_difference=directed, weighted_active_fraction=weighted_active,
                weighted_agreement_k10_g4=agreement, effective_neighbours_k10_g4=eff)

def predict_support(bundle, p_active, max_similarity, directed_class_similarity_difference,
                    weighted_agreement_k10_g4, effective_neighbours_k10_g4):
    t=float(bundle['decision_threshold'])
    p=float(np.clip(p_active,1e-6,1-1e-6))
    pred=int(p>=t)
    z=np.array([[_logit(p)]],dtype=float)
    calibrated_active=float(bundle['active_probability_calibrator'].predict_proba(z)[0,1])
    confidence=calibrated_active if pred==1 else 1.0-calibrated_active
    margin=(p-t) if pred==1 else (t-p)
    x7=np.array([[confidence,margin,max_similarity,directed_class_similarity_difference,
                  weighted_agreement_k10_g4,effective_neighbours_k10_g4,pred]],dtype=float)
    reliability=float(bundle['correctness_model'].predict_proba(support_basis(x7))[0,1])
    pol=bundle['category_policy']
    if max_similarity < pol['indeterminate_if_max_similarity_below'] or effective_neighbours_k10_g4 < pol['indeterminate_if_effective_neighbours_below']:
        cat='Indeterminate'
    elif reliability >= pol['high_reliability_threshold'] and weighted_agreement_k10_g4 >= pol['high_agreement_threshold']:
        cat='High'
    elif reliability >= pol['moderate_reliability_threshold'] and weighted_agreement_k10_g4 >= pol['moderate_agreement_threshold']:
        cat='Moderate'
    else:
        cat='Low'
    return {
        'P_active':p, 'predicted_class':pred, 'predicted_class_confidence':confidence,
        'distance_from_threshold':abs(p-t), 'directed_threshold_margin':margin,
        'max_similarity':float(max_similarity),
        'directed_class_similarity_difference':float(directed_class_similarity_difference),
        'weighted_agreement':float(weighted_agreement_k10_g4),
        'effective_neighbours':float(effective_neighbours_k10_g4),
        'estimated_accuracy_for_support_profile':reliability,
        'Prediction_Support':cat,
    }

def load_bundle(path):
    return joblib.load(path)
