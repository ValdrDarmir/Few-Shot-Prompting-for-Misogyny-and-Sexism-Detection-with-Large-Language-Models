# This script is used for all zero shot prompting methods with Mixtral 8x7B
# For users: change path in main and change model in generate_api_call, also
# add own api key in config
# Author: Niklas Donhauser
# Date: September 10, 2024

# import libraries
import requests
import json
from config import API_KEY_FIREWORKS
from pydantic import BaseModel, Field
import sys
import os
import datetime
import time

# setup API key and link to the fireworks API
API_KEY = API_KEY_FIREWORKS
url = "https://api.fireworks.ai/inference/v1/chat/completions"


# Model for the response so every output looks the same
class Annotation(BaseModel):
    user: str = Field(..., description="Ein Kennzeichner für den Annotator.")
    label: str = Field(
        ...,
        description="Das vergebene Label des Annotators für den Text.",
        enum=["0-Kein", "1-Gering", "2-Vorhanden", "3-Stark", "4-Extrem"]
    )


# json schema for the response
schema_json = json.dumps({
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "minItems": 4,
            "items": Annotation.model_json_schema()
        }
    },
    "required": ["annotations"]
})

# header for the call; Key needed
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


# loading the prompt for the api call
def loadPrompt(promptPath):
    prompt = None
    if os.path.exists(promptPath):
        print("Load")
        file = open(promptPath, "r", encoding="utf-8")
        prompt = file.read()
    else:
        print("Error Reading Prompt")
    return prompt


# load the data to classify
def loadCorpus(dataPath):
    corpus_dict = {}
    if os.path.exists(dataPath):
        with open(dataPath, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line.strip())
                identifier = entry["id"]
                user_text = entry["text"]
                # Handle different key names for annotators
                if "annotations" in entry:
                    element = entry["annotations"]
                    annotators = [annotation["user"] for annotation in element]
                elif "annotators" in entry:
                    annotators = entry["annotators"]
                else:
                    annotators = []

                annotator_count = len(annotators)
                annotator_names = ", ".join(annotators)
                corpus_dict[identifier] = (user_text, annotator_names,
                                           annotator_count)
    else:
        print("Path didn't exist (loadCorpus)")
    return corpus_dict


# save the model response in a jsonl file
def saveResponse(answer, key, text, resultFile):
    if isinstance(answer, str):
        answer = json.loads(answer)
    with open(resultFile, 'a',
              encoding='utf-8') as file:
        data = {
            "id": key,
            "text": text,
            "annotations": answer["annotations"]
        }
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')


# save the used tokens in a jsonl file
def saveTokens(promptTokens, totalTokens, completionTokens,
               key, text, resultPath):
    with open(resultPath, 'a',
              encoding='utf-8') as file:
        data = {
            "id": key,
            "text": text,
            "totalTokens": totalTokens,
            "promptTokens": promptTokens,
            "completionTokens": completionTokens
        }
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')


#  write error messages in a txt file, if model produces errors or the schema
def writeError(message, key, errorFile):
    current_timestamp = datetime.datetime.now()
    error_message = "Error at " + \
        str(key) + " at Time " + str(current_timestamp) + \
        " with error " + message + "\n"
    with open(errorFile, "a") as file:
        file.write(error_message)


# setup the api call; change the model here
def generate_api_call(prompt, singleEntryWithData):
    text = singleEntryWithData[0]
    # 1 = names; 2 = count
    extended_prompt = prompt.format(singleEntryWithData[2],
                                    singleEntryWithData[1])
    payload = {
        "model": "accounts/fireworks/models/mixtral-8x7b-instruct",
        "max_tokens": 1024,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.0,
        "response_format": {"type": "json_object",
                            "schema": json.loads(schema_json)},
        "messages": [
            {
                "role": "system",
                "content": extended_prompt
            },
            {
                "role": "user",
                "content": text
            }

        ]
    }
    print(payload)
    return payload


# setup for the call; getting the text to predict and prepare the response for
# the save
def modelCall(promptPath, dataPath, errorPath,
              resultPath, resultTokensPath):
    prompt = loadPrompt(promptPath)
    corpus = loadCorpus(dataPath)

    for key in corpus:
        # handle too much requests with a wait time
        time.sleep(1)
        singleEntryWithData = corpus[key]
        inputForModel = generate_api_call(prompt, singleEntryWithData)

        try:
            response = requests.request(
                "POST", url, headers=headers, data=json.dumps(inputForModel))

        except Exception as e:
            print(f"ERROR: {e}")
            writeError("Exception", key, errorPath)
            continue

        response_data = response.json()

        print(response_data)

        if 'choices' in response_data and len(response_data['choices']) > 0:
            check = response_data['choices'][0]
            if 'message' in check and 'content' in check['message']:
                try:
                    text = singleEntryWithData[0]
                    answer = check['message']['content']
                    json_answer = json.loads(answer)

                    fin_reason = check["finish_reason"]

                    usage = response_data['usage']
                    promptTokens = usage['prompt_tokens']
                    totalTokens = usage['total_tokens']
                    completionTokens = usage['completion_tokens']

                    if fin_reason not in ["function_call", "stop"]:
                        writeError("Finish reason error", key, errorPath)
                        print(f"Error at {key}: Finish reason error")

                    print(json_answer)
                    saveResponse(json_answer, key, text, resultPath)
                    saveTokens(promptTokens, totalTokens, completionTokens,
                               key, text, resultTokensPath)
                except json.JSONDecodeError:
                    writeError("JSON decode error", key, errorPath)
                    print(f"Error at {key}: JSON decode error")
            else:
                print(f"Error at {key}: Invalid response format")
                writeError("Invalid response format", key, errorPath)
        else:
            print(f"Error at {key}: No response")
            writeError("Request didn't work, no JSON as return",
                       key, errorPath)


def main():
    # change the path
    dataPath = "[path]/[dataset_name].jsonl"
    resultPath = "[path]/result.jsonl"  # noqa: E501
    resultTokensPath = "[path]/result_token.jsonl"  # noqa: E501
    errorPath = "[path]/error_messages.txt"  # noqa: E501
    promptPath = "[path]/basic_prompt.txt"  # noqa: E501

    modelCall(promptPath, dataPath, errorPath,
              resultPath, resultTokensPath)


if __name__ == '__main__':
    sys.exit(main())
