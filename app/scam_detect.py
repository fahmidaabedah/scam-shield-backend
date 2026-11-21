import json
import re
from pathlib import Path

import joblib
import numpy as np

URL_RE = re.compile(r'https?://\S+|\b\w+\.(?:com|io|xyz|top|click|online)\S*|\b(?:bit\.ly|t\.me|wa\.me|cutt\.ly)/\S+')
PHONE_RE = re.compile(r'01[3-9]\d{8}')
BN = 'ঀ-৿'
BN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

LABELS_BN = {
    'legit': 'স্বাভাবিক বার্তা',
    'bank_phishing': 'ব্যাংক ফিশিং',
    'mobile_money_fraud': 'মোবাইল মানি জালিয়াতি',
    'prize_lottery': 'পুরস্কার/লটারি স্কাম',
    'job_scam': 'চাকরির স্কাম',
    'loan_investment_fraud': 'ঋণ/বিনিয়োগ জালিয়াতি',
    'govt_impersonation': 'সরকারি কর্মকর্তা পরিচয়ে প্রতারণা',
    'parcel_delivery_scam': 'পার্সেল ডেলিভারি স্কাম',
    'romance_sextortion': 'রোমান্স/সেক্সটরশন',
}


def clean(t: str) -> str:
    t = str(t).translate(BN_DIGITS).lower()
    t = URL_RE.sub(' urltoken ', t)
    t = PHONE_RE.sub(' phonetoken ', t)
    t = re.sub(r'\d+', ' num ', t)
    t = re.sub(rf'[^\w\s{BN}]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


class Detector:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)
        self.labels = list(self.model.classes_)
        self.legit_idx = self.labels.index('legit')

    def predict_proba(self, texts):
        cleaned = [clean(t) for t in texts]
        return self.model.predict_proba(cleaned)

    def classify(self, msg: str) -> dict:
        probs = self.predict_proba([msg])[0]
        p = dict(zip(self.labels, probs))
        scam_score = 1 - p['legit']
        is_scam = scam_score >= 0.5
        label = max((l for l in p if l != 'legit'), key=p.get) if is_scam else 'legit'
        return {
            'verdict': 'SCAM' if is_scam else 'NOT SCAM',
            'scam_score': round(float(scam_score), 4),
            'type': label,
            'type_bn': LABELS_BN.get(label, label),
            'probability': round(float(p[label]), 4),
            'all': {k: round(float(v), 4) for k, v in sorted(p.items(), key=lambda x: -x[1])},
        }

    def explain(self, msg: str, top_k: int = 4) -> list[tuple[str, float]]:
        words = msg.split()
        if len(words) < 2:
            return [(w, 0.0) for w in words]
        variants = [msg] + [' '.join(words[:i] + words[i + 1:]) for i in range(len(words))]
        probs = self.predict_proba(variants)
        scam = np.clip(1 - probs[:, self.legit_idx], 1e-6, 1 - 1e-6)
        logit = np.log(scam / (1 - scam))
        drop = np.maximum(logit[0] - logit[1:], 0)
        weights = drop / drop.max() if drop.max() > 0 else drop
        ranked = sorted(zip(words, weights), key=lambda x: -x[1])
        return [(w, round(float(s), 4)) for w, s in ranked if s >= 0.35][:top_k]
