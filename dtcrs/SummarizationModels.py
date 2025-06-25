import logging
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
import requests
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT4SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4o-mini"):
        load_dotenv()
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )
            print(response)

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return ""


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):
        load_dotenv()
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return ""


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):
        load_dotenv()
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return ""


class LocalSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="deepseek"):
        """
        Initializes the model with the specified model version.

        Args:
            model (str, optional): The model name to use for summarizing. Defaults to "Qwen".
        """
        load_dotenv()
        self.model = model
        self.api_url = os.getenv("LOCAL_MODEL_API_URL")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        """
        Sends the request to the FastAPI endpoint to get the summary for the given context.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens for the generated summary.
            stop_sequence (str, optional): The sequence at which to stop the summary generation.

        Returns:
            str: The generated summary.
        """
        prompt = (f"Write a summary of the following, including as many key details as possible: {context}\n"
            "Summary:"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Define the request payload
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }

        try:
            # Send the POST request to FastAPI endpoint
            response_data = requests.post(self.api_url, json=data)
            response_data.raise_for_status()
            result = response_data.json()
            # Get the summary from the response
            full_summary = str(result['choices'][0]['message']['content']).strip()
            print(full_summary)

            # 计算 prompt 的长度并从该位置截取答案
            # summary_start_index = len(prompt)
            # summary = full_summary[summary_start_index:].strip()
            return full_summary
        except Exception as e:
            print(f"Error occurred while fetching summary: {e}")
            return ""


