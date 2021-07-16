FROM python:3.7-slim-stretch
EXPOSE 1410
WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
CMD python main.py

