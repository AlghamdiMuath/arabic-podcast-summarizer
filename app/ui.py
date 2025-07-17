import gradio as gr
from app.downloader import download_audio
from app.transcriber import transcribe
from app.diarizer import diarize
from app.segmenter import structure_transcript
from app.ner import extract_entities
from app.summarizer import summarize

import os
import json
from datetime import datetime

def pipeline(youtube_url, status=gr.Progress(track_tqdm=True)):
    try:
        status(0, desc="تحميل الفيديو من يوتيوب...")
        meta = download_audio(youtube_url)
        audio_path = meta["audio_path"]
        video_title = meta["title"]
        episode_id = meta["id"]

        status(0.1, desc="تفريغ النص ...")
        transcript = transcribe(audio_path)
        transcript_text = "\n".join([seg["text"] for seg in transcript["segments"]])
        transcript_path = f"data/transcripts/transcript_{episode_id}.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False)

        status(0.3, desc=" تمييز المتحدثين باستخدام ...")
        diarization = diarize(audio_path)
        diarization_path = f"data/transcripts/diarization_{episode_id}.json"
        with open(diarization_path, "w", encoding="utf-8") as f:
            json.dump(diarization, f, ensure_ascii=False)

        status(0.5, desc=" تنظيم النص حسب المتحدث...")
        structured = structure_transcript(transcript, diarization)
        speaker_blocks = "\n\n".join([f"{spk}:\n{text}" for spk, text in structured.items()])
        structured_path = f"data/transcripts/structured_{episode_id}.json"
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False)

        status(0.7, desc="استخراج الكيانات العربية...")
        ner_results = extract_entities(transcript_text)
        ner_path = f"data/transcripts/entities_{episode_id}.json"
        with open(ner_path, "w", encoding="utf-8") as f:
            json.dump(ner_results, f, ensure_ascii=False)

        status(0.85, desc="إنشاء ملخص نصي بالعربية...")
        summary = summarize(transcript_text)
        summary_path = f"data/summaries/{episode_id}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        status(1.0, desc=" المعالجة اكتملت")

        final_txt_path = f"data/summaries/full_{episode_id}.txt"
        with open(final_txt_path, "w", encoding="utf-8") as f:
            f.write(f"العنوان: {video_title}\n\n")
            f.write("النص الكامل:\n")
            f.write(transcript_text + "\n\n")
            f.write("النص حسب المتحدث:\n")
            f.write(speaker_blocks + "\n\n")
            f.write("الكيانات المستخرجة:\n")
            for ent in ner_results:
                f.write(f"{ent['type']} : {ent['value']}\n")
            f.write("\nالملخص:\n")
            f.write(summary)

        return (
            transcript_text,
            speaker_blocks,
            ner_results,
            summary,
            final_txt_path,
            " تم التلخيص بنجاح!"
        )

    except Exception as e:
        return f" خطأ: {str(e)}", "", [], "", None, " حدث خطأ أثناء التنفيذ."

with gr.Blocks(title="ملخص بودكاست عربي") as demo:
    gr.Markdown("# أداة تلخيص البودكاست العربي")
    gr.Markdown("أدخل رابط يوتيوب لحلقة بودكاست، وستقوم الأداة تلقائيًا بتحميل وتفريغ وتلخيص المحتوى.")

    with gr.Row():
        url_input = gr.Textbox(label="رابط يوتيوب ", placeholder="https://www.youtube.com/watch?v=...")
        summarize_btn = gr.Button("ابدأ المعالجة")

    status_box = gr.Markdown("في انتظار الإدخال...")

    transcript_output = gr.Textbox(label=" النص الكامل", lines=8, show_copy_button=True)
    speakers_output = gr.Textbox(label=" النص حسب المتحدث", lines=8, show_copy_button=True)
    ner_output = gr.Dataframe(headers=["النوع", "القيمة"], label=" الكيانات المستخرجة")
    summary_output = gr.Textbox(label=" الملخص", lines=6, show_copy_button=True)
    download_output = gr.File(label=" تحميل النتيجة بصيغة txt")

    summarize_btn.click(
        fn=pipeline,
        inputs=[url_input],
        outputs=[
            transcript_output,
            speakers_output,
            ner_output,
            summary_output,
            download_output,
            status_box
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
