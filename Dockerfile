FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python src/generate_data.py --out data/segments.csv && python src/train_evaluate.py
EXPOSE 8501
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.address=0.0.0.0"]
