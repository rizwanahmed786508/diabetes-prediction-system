FROM python:3.14-slim

WORKDIR /app

COPY app.py .

RUN pip install --no-cache-dir streamlit requests numpy matplotlib reportlab

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]