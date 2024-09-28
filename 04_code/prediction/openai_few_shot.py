# This script is used for all few shot prompting methods with
# openais gpt-4o mini and gpt-3.5 turbo
# For users: change path in main and change model in generate_api_call, also
# add own api key in config
# Author: Niklas Donhauser
# Date: September 10, 2024

# import libraries
from config import API_KEY_OPENAI
from openai import OpenAI
import json
from pydantic import BaseModel, Field
import sys
import os
import datetime

# set openai key for api calls
client = OpenAI(api_key=API_KEY_OPENAI)


# Model for the response so every output looks the same
class Annotation(BaseModel):
    user: str = Field(..., description="Ein Kennzeichner für den Annotator.")
    label: str = Field(
        ...,
        description="Die Anmerkung des Annotators für den Aspekt.",
        enum=["0-Kein", "1-Gering", "2-Vorhanden", "3-Stark", "4-Extrem"]
    )


# json schema for the response
schema_json = json.dumps({
    "type": "object",
    "properties": {
        "annotation": {
            "type": "array",
            "maxItems": 1,
            "minItems": 1,
            "uniqueItems": True,
            "items": Annotation.model_json_schema()
        }
    },
    "required": ["annotation"]
})


# header for the call; Key needed
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY_OPENAI}",
}


# loading the prompt for the api call
def loadPrompt(promptPath):
    prompt = None
    if os.path.exists(promptPath):
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
def saveResponse(answerList, key, text, resultFile):
    transformed_list = []

    for entry in answerList:
        if isinstance(entry, list):
            transformed_list.extend(entry)

    with open(resultFile, 'a', encoding='utf-8') as file:
        data = {
            "id": key,
            "text": text,
            "annotations": transformed_list
        }
        print(data)
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


# combine different messages (Basic prompt, summary guidelines, examples) to a
# prompt
def generateMessage(promptPath, annotator, text):
    n_shot = 5
    prompt = loadPrompt(promptPath)
    message = []

    systemMessage = {'role': 'system', 'content': prompt}
    message.append(systemMessage)

    for n in range(n_shot):
        examplesJson = f'data/examples/{n_shot}_examples/{annotator}.jsonl'
        with open(examplesJson, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                if line_number == n + 1:
                    data = json.loads(line)

        exampleText = data["text"]
        exampleText = exampleText + "\n Klassifiziere diesen Text auf Sexismus und Frauenfeindlichkeit. Gib *genau* ein Label als Antwort"  # noqa: E501
        exampleResponse = data["annotations"]
        exampleUser = {'role': 'user', 'content': exampleText}
        message.append(exampleUser)

        exampleAssistant = {'role': 'assistant', 'content': str(exampleResponse)}  # noqa: E501
        message.append(exampleAssistant)

    text = text + "\n Klassifiziere diesen Text auf Sexismus und Frauenfeindlichkeit. Gib *genau* ein Label als Antwort"  # noqa: E501
    textToPredict = {'role': 'user', 'content': text}
    message.append(textToPredict)

    # print(message)
    # print("\n----------------------------\n")
    return message


# setup the api call; change the model here
# responses have to be processed different than with fireworks
def modelCall(promptPath, dataPath, errorPath,
              resultPath, resultTokensPath):
    corpus = loadCorpus(dataPath)

    for key in corpus:
        singleEntryWithData = corpus[key]

        # get Annotators
        allAnnotators = singleEntryWithData[1]
        annotatorList = allAnnotators.split(", ")

        # get text to predict
        text = singleEntryWithData[0]

        answerList = []
        for annotator in annotatorList:

            messagesUser = generateMessage(promptPath, annotator, text)
            # print("Meine Message:", messagesUser)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=messagesUser,
                    function_call="auto",
                    functions=[{
                        "name": "annotate",
                        "parameters": json.loads(schema_json)
                    }]
                )

            except Exception as e:
                print(f"ERROR: {e}")
                writeError("Exception", key, errorPath)
                continue
            # print(annotator, key)
            # print(response_data['choices'][0]['message']['content'])
            # print("------------")
            if hasattr(response, 'choices') and len(response.choices) > 0:  # noqa: E501

                check = response.choices[0]

                if hasattr(check, 'message') and hasattr(check.message,
                                                         'content'):
                    try:

                        text = singleEntryWithData[0]
                        answer = check.message.content

                        if answer is None and check.message.function_call:
                            answer = check.message.function_call.arguments
                            answer_json = json.loads(answer)
                            if "annotation" in answer_json:
                                answer_json = answer_json["annotation"]

                            # Convert the extracted list back to a JSON string
                            answer = json.dumps(answer_json)
                        if isinstance(answer, str):
                            answer = answer.replace("'", '"')

                        json_answer = json.loads(answer)

                        fin_reason = check.finish_reason
                        if fin_reason not in ["function_call", "stop"]:
                            writeError("Finish reason error", key, errorPath)
                            print(f"Error at {key}: Finish reason error")

                        usage = response.usage
                        promptTokens = usage.prompt_tokens
                        totalTokens = usage.total_tokens
                        completionTokens = usage.completion_tokens

                        answerList.append(json_answer)
                        saveTokens(promptTokens, totalTokens,
                                   completionTokens, key, text,
                                   resultTokensPath)

                    except Exception as e:
                        print(f"ERROR in Ex: {e}")
                        writeError("Exception in Response code",
                                   key,
                                   errorPath)
                        continue

                else:
                    print(f"Error at {key}: Invalid response format")
                    writeError("Invalid response format", key, errorPath)
            else:
                print(f"Error at {key}: No response")
                writeError("Request didn't work, no JSON as return",
                           key, errorPath)

        saveResponse(answerList, key, text, resultPath)


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
