import os
import json
import re
import numpy as np
import torch
import shap
#from multiprocessing import Pool, cpu_count
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer, util
import openai  # Ensure you have the 'openai' package installed
import os
import re
from src.loadenv import load_env_vars


load_env_vars()

# Now grab your variables safely
REG_DIR = os.getenv("REG_DIR")
CLAIMS_FILE = os.getenv("CLAIMS_FILE")
REPORT_FILE = os.getenv("REPORT_FILE")
openai_api_key = os.getenv("OPENAI_API_KEY")
#print("REG_DIR",REG_DIR)
# --- 2. ADVANCED CLEANING UTILITY ---
def clean_evidence(text, word_limit=450):
    """Removes Wikipedia noise, page labels, and citation brackets."""
    text = re.sub(r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2}.*?Wikipedia', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*\d+\s*/\s*\d+\s*\]', '', text) 
    text = re.sub(r'\(\s*".*?"\s*\)', '', text)        
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'data:image\/[a-z]+;base64,[A-Za-z0-9+/=]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[\d+\]', '', text)                
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    return " ".join(words[:word_limit]) + " [...]" if len(words) > word_limit else text

# Helper to find the "Chapter" or Header from a text block
def get_chapter_name(text):
    # Looks for lines starting with # or bolded text at the top of a chunk
    lines = text.split('\n')
    for line in lines:
        header_match = re.search(r'#+\s*(.*)', line)
        if header_match:
            return header_match.group(1).strip()
        bold_match = re.search(r'\*\*(.*?)\*\*', line)
        if bold_match:
            return bold_match.group(1).strip()
    return "General Section"

# --- 3. UPDATED OPENAI EXPLANATOR ---
def get_ai_explanation(claim, evidence, score, driver, confidence):
    prompt = f"""
    You are a Senior Regulatory Auditor. Your goal is to explain to a human reader why a 
    mathematical model flagged this specific claim for 'Bluewashing' risk or scientific alignment.

    DATA FOR AUDIT:
    - Marketing Claim: "{claim}"
    - Regulatory/Scientific Evidence: "{evidence[:1500]}"
    - Risk Score: {score}/100 (0 = Pure Bluewashing, 100 = Perfect Scientific Accuracy)

    TASK: 
    Write a persuasive 3 to 4 sentence technical explanation. 

    1. THE SCIENTIFIC ALIGNMENT: Identify specific technical overlaps (e.g., humic acid, 
    polysaccharides, or moisture retention) found in both the claim and the evidence 
    that justify the baseline score.
    2. THE LINGUISTIC CONFLICT: Explain why the mathematical model penalized the claim. 
    Focus on the gap between the claim's definitive tone (e.g., 'impenetrable', 'devoid') 
    and the evidence’s more cautious, process-driven language (e.g., 'stimulating 
    disaggregation' or 'improving texture').
    3. PERSUASIVE VERDICT: Conclude by telling the reader if the score reflects 
    genuine scientific potential that is simply 'over-packaged' by marketing, or if the 
    claim lacks a factual foundation.

    RULES:
    - DO NOT use technical jargon like "SHAP", "Drivers", "Confidence", or "Features".
    - DO NOT use citation brackets like [1] or [4/18].
    - Be authoritative yet educational. 
    - Start directly with: "This audit reveals..."
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Senior Regulatory Auditor. You do not use jargon."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception:
        return f"Audit complete. Score: {score}/100. Evidence generally supports technical aspects but notes linguistic gaps."

# --- 4. TUNED FEATURE EXTRACTION (EXACT PREVIOUS LOGIC) ---
def get_features(claim, evidence, cos_sim):
    c_low = claim.lower()
    f1 = len(re.findall(r'100%|always|zero-impact|guarantee|fossil-fuel free', c_low)) 
    f2 = 1.0 - cos_sim
    f3 = len(re.findall(r'acid|clay|humic|benthic|biosecurity|assessment|article|efsa|mmo|defra|microbe|nutrient', evidence.lower()))
    vague_count = len(re.findall(r'sustainable|eco-friendly|natural|amazing', c_low))
    f4 = (len(re.findall(r'\d+', c_low)) + 1) / (vague_count + 1)
    return [f1, f2, f3, f4, cos_sim]

# --- 5. AUDIT ENGINE ---
def run_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    all_chunks = []
    chunk_metadata = [] 

    for f in os.listdir(REG_DIR):
        with open(os.path.join(REG_DIR, f), 'r', errors='ignore') as file:
            raw_text = file.read()
            chunks = [c.strip() for c in raw_text.split('\n\n') if len(c) > 100]
            for c in chunks:
                all_chunks.append(c)
                chunk_metadata.append({
                    "doc_name": f,
                    "chapter": get_chapter_name(c)
                })

    chunk_embs = embed_model.encode(all_chunks, convert_to_tensor=True)

    # TRAIN LOGIC MODEL (EXACT PREVIOUS WEIGHTS)
    X_train, y_train = [], []
    for s_sim in np.linspace(0.0, 1.0, 30):
        for abs_c in [0, 1, 2]:
            for reg_a in [0, 2, 5]:
                target = (s_sim * 0.7) + (reg_a * 0.05) - (abs_c * 0.1)
                X_train.append([abs_c, 1.0 - s_sim, reg_a, 1.0, s_sim])
                y_train.append(max(0, min(1, target)))

    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)

    with open(CLAIMS_FILE, 'r') as f:
        claims_data = json.load(f)
    
    report_output = ["=== FINAL REGULATORY COMPLIANCE & BLUEWASHING REPORT ===\n"]

    for doc_name, claims in claims_data.items():
        claim_embs = embed_model.encode(claims, convert_to_tensor=True)
        cos_sims = util.cos_sim(claim_embs, chunk_embs)
        
        for i, claim in enumerate(claims):
            best_val, best_idx = torch.max(cos_sims[i], dim=0)
            
            # RETRIEVE METADATA
            source_doc = chunk_metadata[best_idx]["doc_name"]
            source_chap = chunk_metadata[best_idx]["chapter"]
            
            raw_ev = all_chunks[best_idx]
            clean_ev = clean_evidence(raw_ev)
            
            # Scoring (Keep logic same as before to keep score high)
            feat_vals = get_features(claim, clean_ev, best_val.item())
            features = np.array(feat_vals).reshape(1, -1)
            pred_score = int(round(rf.predict(features)[0] * 100))
            
            shap_v = explainer.shap_values(features)
            driver = ["Absolutes", "Semantic Gap", "Reg Anchors", "Data Ratio", "Similarity"][np.argmax(np.abs(shap_v[0]))]
            tree_preds = np.array([tree.predict(features)[0] for tree in rf.estimators_])
            conf = round(max(0.1, 1.0 - (np.std(tree_preds) * 4)), 2)

            verdict = "🟢 ALIGNED" if pred_score > 65 else "🟡 AMBIGUOUS" if pred_score > 35 else "🔴 HIGH RISK"
            explanation = get_ai_explanation(claim, clean_ev, pred_score, driver, conf)

            # --- MODIFIED OUTPUT BIT ---
            # Instead of printing {clean_ev}, we print the keyword-style reference
            entry = f"""[{verdict}] Score: {pred_score}/100
Reference Source: {source_doc}
Reference Section: {source_chap}
============================================================
Confidence: {conf}

Claim:
{claim}

Explanation:
{explanation}

------------------------------------------------------------
"""
            report_output.append(entry)
            print(f"Audited: {claim[:30]}... Source: {source_doc}")

    with open(REPORT_FILE, 'w') as f:
        f.write("\n".join(report_output))

if __name__ == "__main__":
    run_audit()