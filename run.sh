#!/bin/bash
echo "Download Model"
python donwload_model.py --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" --revision "${REVISION}" && rm donwload_model.py

if [ $? -ne 0 ]
then
  echo "Fail to Download Model"
  exit 1
fi

echo "Archive Model"
torch-model-archiver --model-name text-generation --version 1.0 --serialized-file ./model/pytorch_model.bin --handler ./handler.py --extra-files ./model

if [ $? -ne 0 ]
then
  echo "Fail to Archive Model"
  exit 1
fi

echo "Run API Server"
torchserve --start --ncs --model-store=./ --models=./text-generation.mar
