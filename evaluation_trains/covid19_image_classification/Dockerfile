FROM python:3.8

WORKDIR /home

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python",  "covid19_image_classification_main.py" ]
