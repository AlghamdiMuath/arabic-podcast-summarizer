import os
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load mT5 summarization pipeline (fine-tuned for multilingual use)
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize(text: str, max_chunk_len=1000) -> str:
    # Preprocess: Break long text into smaller chunks
    chunks = [text[i:i+max_chunk_len] for i in range(0, len(text), max_chunk_len)]
    summaries = []
    
    for chunk in chunks:
        output = summarizer(chunk, max_length=200, min_length=30, do_sample=False)
        summaries.append(output[0]["summary_text"])
    
    # Format into Arabic bullet points
    bullet_points = "\n".join([f"• {line.strip()}" for line in summaries])
    return f"🔹 أهم النقاط:\n{bullet_points}"

def save_summary(text: str, id: str, output_dir="data/summaries"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{id}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path

# Optional: Whisper-to-English + translate back using Argos Translate
def summarize_via_english(text: str) -> Optional[str]:
    try:
        import argostranslate.package, argostranslate.translate
        installed_languages = argostranslate.translate.get_installed_languages()
        en = next((l for l in installed_languages if l.code == "en"), None)
        ar = next((l for l in installed_languages if l.code == "ar"), None)
        if not (en and ar):
            raise RuntimeError("Arabic/English Argos Translate packages not installed")

        from transformers import pipeline
        eng_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Translate to English
        eng_text = ar.translate(text, en)
        eng_summary = eng_summarizer(eng_text[:1024], max_length=180, min_length=30, do_sample=False)[0]["summary_text"]

        # Back-translate to Arabic
        return en.translate(eng_summary, ar)

    except Exception as e:
        print("Fallback summarization failed:", e)
        return None
