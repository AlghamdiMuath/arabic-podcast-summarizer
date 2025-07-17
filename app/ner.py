import json
from typing import List, Dict
from transformers import pipeline
import os

# Try to load AraBERT NER model
try:
    ner_pipeline = pipeline(
        "ner",
        model="aubmindlab/bert-base-arabertv02-ner",
        tokenizer="aubmindlab/bert-base-arabertv02-ner",
        grouped_entities=True
    )
    MODEL = "arabert"
except Exception as e:
    print(" Failed to load AraBERT. Trying CAMeL Tools fallback.")
    from camel_tools.ner import NERecognizer
    ner_pipeline = NERecognizer.pretrained()
    MODEL = "camel"

def extract_entities(text: str) -> List[Dict[str, str]]:
    if MODEL == "arabert":
        results = ner_pipeline(text)
        entities = []
        for ent in results:
            entity = {
                "entity": ent['word'].strip(),
                "type": ent['entity_group']
            }
            entities.append(entity)
        return entities

    elif MODEL == "camel":
        results = ner_pipeline.predict_sentence(text)
        entities = []
        current_entity = ""
        current_type = ""
        for word, tag in results:
            if tag.startswith("B-"):
                if current_entity:
                    entities.append({"entity": current_entity.strip(), "type": current_type})
                current_entity = word
                current_type = tag[2:]
            elif tag.startswith("I-") and current_type:
                current_entity += " " + word
            else:
                if current_entity:
                    entities.append({"entity": current_entity.strip(), "type": current_type})
                    current_entity = ""
                    current_type = ""
        if current_entity:
            entities.append({"entity": current_entity.strip(), "type": current_type})
        return entities

    else:
        raise RuntimeError("No valid NER model loaded")

def save_entities(text: str, id_: str, output_dir="data/transcripts"):
    entities = extract_entities(text)
    output_path = os.path.join(output_dir, f"entities_{id_}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    print(f" Named entities saved to {output_path}")
    return entities
