# TextGenerationAPI
Text Generation API using Huggingface Transformers

You can use the model from the link below.
* https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads

## How to Use
You have to modify the Dockerfile according to the procedure below to suit the model you are going to deploy.

#### 1. Choose the appropriate torchserve base image.

Choose the torchserve base image suitable for the model you are distributing. At the time of writing this article, the latest version is the `0.5.2` version.
```dockerfile
FROM pytorch/torchserve:0.5.2-gpu
```

#### 2. Modify model name and revision you want to deploy.

Find the model in the hugging face and modify it accordingly in the Dockerfile. If there is no revision of the model you found, you can enter `main`.
```dockerfile
ENV PRETRAINED_MODEL_NAME_OR_PATH="gpt2"
ENV REVISION="main"
```

#### 3. Install transformers
Install a `transformer` that fits the model you will use. At the time of writing this article, the latest version is `4.15.0`.
```dockerfile
RUN pip install transformers==4.15.0
```

#### 4. Build and run Dockerfile.
If the modification is finished, please build and run the docker file.

* In order to use gpu on the docker, you must install nvidia-docker.
* https://github.com/NVIDIA/nvidia-docker

```shell
docker build -t text-generation .
```

```shell
docker run -d --gpus all -p 8080:8080 text-generation
```

#### 5. Test Your Model
```shell
curl --request POST 'http://127.0.0.1:8080/predictions/text-generation' \
 --header 'Content-Type: application/json' \
 --data-raw '{
    "text_inputs": "My name is Teven and I am"
  }'

["My name is Teven and I am a student at the University of California, Berkeley. I am"]
```
