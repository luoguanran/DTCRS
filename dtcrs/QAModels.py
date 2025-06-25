import os

import requests
from openai import OpenAI

from dotenv import load_dotenv
import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question, answer_type):
        pass


class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        load_dotenv()
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.completions.create(
                prompt=f"using the following information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()

        except Exception as e:
            print(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        load_dotenv()
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, answer_type, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        if answer_type == "generate":
            prompt = f"using the following information {context}. Answer the following question in less than 5-7 words, if possible: {question}"
        else:
            prompt = f"Given Context: {context} Choose the best full answer amongst the option to question: {question}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, answer_type, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, answer_type, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return ""


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4o-mini"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        load_dotenv()
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, answer_type, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        if answer_type == "generate":
            prompt = f"using the following information {context}. Answer the following question in less than 5-7 words, if possible: {question}"
        else:
            prompt = f"Given Context: {context} Choose the best full answer amongst the option to question: {question}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, answer_type, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, answer_type, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return ""


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]


class LocalQAModel(BaseQAModel):
    def __init__(self, model="deepseek"):
        """
        Initializes the model with the specified model version.

        Args:
            model (str, optional): The model name to use for generating answers. Defaults to "Qwen".
        """
        load_dotenv()
        self.model = model
        self.api_url = os.getenv("LOCAL_MODEL_API_URL")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, answer_type, max_tokens=150, stop_sequence=None):
        """
        Sends the request to the FastAPI endpoint to get the answer for the given question.

        Args:
            context (str): The context to help answer the question.
            question (str): The question to be answered.
            max_tokens (int, optional): The maximum number of tokens for the generated answer.
            stop_sequence (str, optional): The sequence at which to stop the answer generation.

        Returns:
            str: The generated answer.
        """
        if answer_type == "generate":
            prompt = f"using the following information {context}. Answer the following question in less than 5-7 words, if possible: {question}"
        else:
            prompt = f"Given Context: {context} Choose the best full answer amongst the option to question: {question}"

        messages = [
            {"role": "system", "content": "You are Question Answering Portal"},
            {
                "role": "user",
                "content": prompt,
            },
        ]
        # Define the request payload
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1000,
        }

        response_data = requests.post(self.api_url, json=data)
        response_data.raise_for_status()  # Raises exception for non-200 status codes

        # 解析响应
        result = response_data.json()
        # 获取完整的回答内容
        answer = str(result['choices'][0]['message']['content']).strip()
        print("answer"+answer)

        return answer

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, answer_type, max_tokens=150, stop_sequence=None):
        """
        Public method to answer the question based on context and the provided model.

        Args:
            context (str): The context to help answer the question.
            question (str): The question to be answered.
            max_tokens (int, optional): The maximum number of tokens for the generated answer.
            stop_sequence (str, optional): The sequence at which to stop the answer generation.

        Returns:
            str: The generated answer.
        """
        try:
            return self._attempt_answer_question(context, question, answer_type, max_tokens=max_tokens, stop_sequence=stop_sequence)
        except Exception as e:
            print(f"Error in answering question: {e}")
            return ""
