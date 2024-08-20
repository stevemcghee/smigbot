import google.generativeai as genai
from typing import Iterable

from data_model import ChatMessage, State
import mesop as me
from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-west1",
    api_endpoint: str = "us-west1-aiplatform.googleapis.com",):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", prediction)

predict_custom_trained_model_sample(
    project="1049076937274",
    endpoint_id="3619033726731681792", # gemma-serve-vllm_20240731_074330-endpoint
    location="us-west1",
    instances={ "instance_key_1": "value", } # still gotta figure this out
)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

def configure_gemma():
    state = me.state(State)

def send_prompt(prompt: str, history: list[ChatMessage]) -> Iterable[str]:
    configure_gemma()
    # model = genai.GenerativeModel(
    #     model_name="gemma2-local",
    #     generation_config=generation_config,
    # )
    chat_session = model.start_chat(
        history=[
            {"role": message.role, "parts": [message.content]} for message in history
        ]
    )
    for chunk in chat_session.send_message(prompt, stream=True):
        yield chunk.text



# ---------- from notebook ------- #
# Chat templates.
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

# endpoint_name = endpoint_without_peft.name
# aip_endpoint_name = (
#     f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{endpoint_name}"
# )
# endpoint = aiplatform.Endpoint(aip_endpoint_name)
# aiplatform.Endpoint(aip_endpoint_name)

# projects/1049076937274/locations/us-west1/endpoints/4501739253696299008

# Sample formatted prompt.
prompt = (
    USER_CHAT_TEMPLATE.format(prompt="What is a good place for travel in the US?")
    + MODEL_CHAT_TEMPLATE.format(prompt="California.")
    + USER_CHAT_TEMPLATE.format(prompt="What can I do in California?")
    + "<start_of_turn>model\n"
)
print("Chat prompt:\n", prompt)

instances = [
    {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 1,
    },
]
response = endpoints["hexllm_tpu"].predict(instances=instances)

prediction = response.predictions[0]
print(prediction)