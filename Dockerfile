#FROM pytorch/torchserve:0.5.2-cpu
FROM pytorch/torchserve:0.5.2-gpu

USER root

# Setting ENV value you want to deploy.
ENV PRETRAINED_MODEL_NAME_OR_PATH="gpt2"
ENV REVISION="main"

ENV LRU_CACHE_CAPACITY=1

# Install transformers
RUN pip install transformers==4.15.0

RUN echo default_workers_per_model=1 >> /home/model-server/config.properties
RUN echo default_response_timeout=3600 >> /home/model-server/config.properties
RUN echo unregister_model_timeout=3600 >> /home/model-server/config.properties

# Copy Code
COPY ./download_model.py ./donwload_model.py
COPY ./handler.py ./handler.py

# Run
EXPOSE 8080
COPY ./run.sh ./run.sh
CMD ./run.sh