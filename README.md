## Build a Dockerized Inference API using Cog
This repository contains the code and instructions to build a Dockerized Inference API for an LLM using Cog. For detailed tutorial of building the docker image and deploying to it to AWS EC2, please refer to [our blog](https://blog.neuralwork.ai/).
The LLM is the mistral-7b finetuned on the style instruct dataset and named mistral-7b-style-instruct. Training code and instructions of the model can be found in the [instruct-finetune-mistral](https://github.com/neuralwork/instruct-finetune-mistral) repository, its detailed tutotial can be found in [our blog post](https://blog.neuralwork.ai/deploying-llms-on-aws-ec2-using-cog-a-complete-guide/).

## Pre-requisites
- Nvidia GPU with CUDA support.
- [Docker](https://www.docker.com/) installed.
- [Cog](https://github.com/replicate/cog) installed.
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

## Build the Docker Image
To build the Docker image, run the following in the cloned directory:
```bash
cog build -t mistral-7b-style-instruct
```
This will build the Docker image with the name mistral-7b-style-instruct.

## Run the Docker Image
To run the Docker image, run the following in the cloned directory:
```bash
docker run -p 5000:5000 mistral-7b-style-instruct
```
## Test the Inference API
To test the Inference API, you can use the following curl command:

```bash
curl http://localhost:5000/predictions -X POST -H "Content-Type: application/json" -d '{"input": {"prompt":"I am an athletic and 180cm tall man in my mid twenties, I have a rectangle shaped body with slightly broad shoulders and have a sleek,casual style. I usually prefer darker colors.", "event": "I am going to a wedding."}}'
```
Or you can use the following python code:
```python
import requests

url = 'http://localhost:5000/predictions'
data = {"input": {"prompt":"I am an athletic and 180cm tall man in my mid twenties, I have a rectangle shaped body with slightly broad shoulders and have a sleek,casual style. I usually prefer darker colors.", "event": "I am going to a wedding."}}
response = requests.post(url, json=data)
print(response.json())
```

From [neuralwork](https://neuralwork.ai/) with :heart:
