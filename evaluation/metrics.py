import re
from typing import List, Set, Dict, Tuple
from bert_score import score as bert_score_fn


def compute_n_fcr(generated_text: str, negative_attrs: Set[str], synonym_map: Dict[str, List[str]] = None) -> float:
    if not negative_attrs:
        return 1.0
    synonym_map = synonym_map or {}
    text_lower = generated_text.lower()
    covered = 0
    for attr in negative_attrs:
        candidates = [attr] + synonym_map.get(attr, [])
        if any(c.lower() in text_lower for c in candidates):
            covered += 1
    return covered / len(negative_attrs)


def compute_f_ehr(generated_text: str, kg_facts: Dict[str, str]) -> float:
    if not kg_facts:
        return 1.0
    text_lower = generated_text.lower()
    hit = sum(1 for v in kg_facts.values() if str(v).lower() in text_lower)
    return hit / len(kg_facts)


def compute_p_ehr(generated_text: str, user_pref_attrs: Set[str], synonym_map: Dict[str, List[str]] = None) -> float:
    if not user_pref_attrs:
        return 1.0
    synonym_map = synonym_map or {}
    text_lower = generated_text.lower()
    hit = 0
    for attr in user_pref_attrs:
        candidates = [attr] + synonym_map.get(attr, [])
        if any(c.lower() in text_lower for c in candidates):
            hit += 1
    return hit / len(user_pref_attrs)


def compute_bert_score(generated_texts: List[str], reference_texts: List[str]) -> Tuple[float, float, float]:
    P, R, F1 = bert_score_fn(generated_texts, reference_texts, model_type="roberta-large", verbose=False)
    return float(P.mean()), float(R.mean()), float(F1.mean())


def compute_bleu(generated_text: str, reference_text: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    hypothesis = generated_text.lower().split()
    reference = [reference_text.lower().split()]
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)


def evaluate_batch(
    generated_texts: List[str],
    reference_texts: List[str],
    negative_attrs_list: List[Set[str]],
    kg_facts_list: List[Dict[str, str]],
    user_pref_attrs_list: List[Set[str]],
    synonym_map: Dict[str, List[str]] = None,
) -> Dict[str, float]:
    n_fcr_scores = [
        compute_n_fcr(gen, neg_attrs, synonym_map)
        for gen, neg_attrs in zip(generated_texts, negative_attrs_list)
    ]
    f_ehr_scores = [
        compute_f_ehr(gen, kg_facts)
        for gen, kg_facts in zip(generated_texts, kg_facts_list)
    ]
    p_ehr_scores = [
        compute_p_ehr(gen, pref_attrs, synonym_map)
        for gen, pref_attrs in zip(generated_texts, user_pref_attrs_list)
    ]
    bleu_scores = [
        compute_bleu(gen, ref)
        for gen, ref in zip(generated_texts, reference_texts)
    ]
    _, _, bert_f1 = compute_bert_score(generated_texts, reference_texts)

    return {
        "N-FCR": sum(n_fcr_scores) / len(n_fcr_scores),
        "F-EHR": sum(f_ehr_scores) / len(f_ehr_scores),
        "P-EHR": sum(p_ehr_scores) / len(p_ehr_scores),
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "BERTScore-F1": bert_f1,
    }
