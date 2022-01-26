#FROM pytorch/torchserve:0.5.2-cpu
FROM pytorch/torchserve:0.5.2-gpu

USER root

ENV PRETRAINED_MODEL_NAME_OR_PATH="gpt2"
ENV REVISION="main"
ENV LRU_CACHE_CAPACITY=1

# Install transformers
RUN pip install transformers==4.15.0 loguru==0.5.3

# Download Model
COPY ./download_model.py ./donwload_model.py
RUN python donwload_model.py --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" --revision "${REVISION}" && rm donwload_model.py

# Archieve Model
COPY ./handler.py ./handler.py
RUN torch-model-archiver --model-name text-generation --version 1.0 --serialized-file ./model/pytorch_model.bin --handler ./handler.py --extra-files ./model


RUN echo default_workers_per_model=1 >> /home/model-server/config.properties
RUN echo default_response_timeout=3600 >> /home/model-server/config.properties
RUN echo unregister_model_timeout=3600 >> /home/model-server/config.properties

EXPOSE 8080

CMD ["torchserve", "--start", "--ncs", "--model-store=./", "--models=./text-generation.mar"]