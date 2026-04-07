from typing import Dict, List, Set


AMPLIFY_PHRASES: Set[str] = {
    "excellent", "outstanding", "perfect", "superb", "exceptional",
    "impressive", "remarkable", "fantastic", "great", "wonderful",
    "highly recommend", "best", "top-tier", "superior", "ideal",
    "love", "enjoy", "appreciate", "satisfied", "pleased",
}

WARNING_PHRASES: Set[str] = {
    "however", "note that", "unfortunately", "be aware", "warning",
    "but", "although", "despite", "drawback", "downside",
    "disappointing", "poor", "weak", "lacking", "limited",
    "issue", "problem", "concern", "caution", "careful",
    "not ideal", "may not", "could be better", "falls short",
}

STANCE_LEXICON: Dict[str, Set[str]] = {
    "Amplify": AMPLIFY_PHRASES,
    "Warning": WARNING_PHRASES,
}


class RuleAnchoredVerifier:

    def __init__(self, synonym_map: Dict[str, List[str]] = None):
        self.synonym_map = synonym_map or {}
        self.stance_lexicon = STANCE_LEXICON

    def key_anchor(self, text: str, attribute: str) -> bool:
        text_lower = text.lower()
        if attribute.lower() in text_lower:
            return True
        for syn in self.synonym_map.get(attribute, []):
            if syn.lower() in text_lower:
                return True
        return False

    def tone_match(self, text: str, attribute: str, stance: str) -> bool:
        text_lower = text.lower()
        target_phrases = self.stance_lexicon.get(stance, set())
        sentences = text_lower.split(".")
        context_sentences = []
        for i, sent in enumerate(sentences):
            if attribute.lower() in sent or any(syn.lower() in sent for syn in self.synonym_map.get(attribute, [])):
                start = max(0, i - 2)
                end = min(len(sentences), i + 3)
                context_sentences.extend(sentences[start:end])
        if not context_sentences:
            context_sentences = sentences
        context = " ".join(context_sentences)
        return any(phrase in context for phrase in target_phrases)

    def compute_follow_reward(self, generated_text: str, action: List[tuple]) -> float:
        if not action:
            return 1.0
        total = 0.0
        for attr, stance in action:
            ka = float(self.key_anchor(generated_text, attr))
            tm = float(self.tone_match(generated_text, attr, stance))
            total += ka * tm
        return total / len(action)
