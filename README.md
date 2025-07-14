\# Arabic Podcast Summarizer | مُلخّص بودكاستات عربية



A professional-quality tool that takes a YouTube podcast link and returns a structured Arabic summary using local, open-source ML models.



أداة بجودة احترافية تأخذ رابط بودكاست يوتيوب وتُنتج ملخّصًا عربيًا منظمًا باستخدام نماذج ذكاء اصطناعي محليّة ومفتوحة المصدر.



---



\##  Features | الميزات



\- Download \& transcribe Arabic audio | تحميل وتفريغ صوتي باللغة العربية

\- Speaker diarization (host/guest) | التعرف على المتحدثين (مُضيف/ضيف)

\-  Arabic Named Entity Recognition | استخراج الكيانات (أشخاص، شركات...)

\-  Arabic Summarization | تلخيص نص عربي مُنظم

\- Simple UI (Streamlit or Gradio) | واجهة استخدام سهلة



---



\## Tech Stack | التقنيات



\- `yt-dlp`, `Whisper`, `pyannote-audio`, `CAMeL Tools` or `AraBERT`, `mT5`

\- Python 3.10+, Local inference preferred (no cloud APIs)



---



\## Usage | طريقة الاستخدام



1\. Install dependencies:

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt

2\. Run the app:
   ```bash
   streamlit run app/ui.py

3\. Paste a YouTube link and get your summary!






&nbsp;In Progress | المشروع قيد التطوير

