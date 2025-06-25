import os
from openai import OpenAI
from dotenv import load_dotenv
import requests


class GPTModule:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the ChatGPT module.

        Parameters:
        - model (str): The GPT model to use, default is gpt-4o-mini.
        """
        # Set OpenAI API key
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.model = model

    def generate_table_of_contents(self, document_text):
        """
        Generate a table of contents for a document using ChatGPT.

        Parameters:
        - document_text (str): The original text of the document.

        Returns:
        - table_of_contents (str): The generated table of contents.
        """
        if len(document_text) > 125000:
            document_text = document_text[:125000]

        prompt = (
            "Please generate a detailed table of contents for the following document. "
            "The table of contents should include chapter and subchapter titles.\n\n"
            f"Document text: \n{document_text}\n\n"
            "Table of contents:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an assistant helping the user generate a table of contents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0,
            )
            table_of_contents = response.choices[0].message.content
            return table_of_contents
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None

    def decompose_question(self, question, table_of_contents):
        """
        Decompose a question into sub-questions based on the table of contents.

        Parameters:
        - question (str): The user’s question.
        - table_of_contents (str): The generated table of contents.

        Returns:
        - subquestions (list): A list of decomposed sub-questions.
        """
        prompt = (
            "Based on the following table of contents, decompose the user's question into multiple "
            "relevant sub-questions that can be answered step by step. The sub-questions should not exceed the scope "
            "of the original question.\n\n"
            f"Table of contents:\n{table_of_contents}\n\n"
            f"User question: {question}\n\n"
            "Please break down the question into sub-questions that correspond to sections in the table of contents. "
            "Do not extend the scope beyond the original question. Format the output as follows:\n"
            "1. Sub-question 1\n"
            "2. Sub-question 2\n"
            "..."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an assistant helping the user decompose questions into sub-questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0,
            )

            subquestions = response.choices[0].message.content.strip()
            subquestion_list = [subq.strip() for subq in subquestions.split('\n') if subq.strip()]
            return subquestion_list
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None

    def decompose_question_no_content(self, question):
        """
        Decompose a question into sub-questions without using a table of contents.

        Parameters:
        - question (str): The user’s question.

        Returns:
        - subquestions (list): A list of decomposed sub-questions.
        """
        prompt = (
            f"User question: {question}\n\n"
            "Please break down the question into sub-questions that correspond to sections in the table of contents. "
            "Do not extend the scope beyond the original question. Format the output as follows:\n"
            "1. Sub-question 1\n"
            "2. Sub-question 2\n"
            "..."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an assistant helping the user decompose questions into sub-questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0,
            )

            subquestions = response.choices[0].message.content.strip()
            subquestion_list = [subq.strip() for subq in subquestions.split('\n') if subq.strip()]
            return subquestion_list
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None

    def is_complex_question(self, question, table_of_contents):
        """
        Determine if a question is complex, meaning it requires referencing multiple sections in the table of contents to answer.

        Parameters:
        - question (str): The user’s question.
        - table_of_contents (str): The generated table of contents.

        Returns:
        - bool: Returns True if the question requires multiple sections for an answer, otherwise False.
        """
        prompt = (
            "Given the table of contents listed below, determine if the user's question is complex and "
            "requires information from multiple sections to be fully answered. Answer with 'Yes' if the question is complex, "
            "or 'No' if it is not.\n\n"
            f"Table of contents:\n{table_of_contents}\n\n"
            f"User question: {question}\n"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an assistant tasked with evaluating the complexity of questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0,
            )

            answer = response.choices[0].message.content.strip()
            # print("Need to summarize? → ", answer)
            return True if "yes" in answer.lower() else False
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return False


class LocalModule:
    def __init__(self, model="deepseek", api_key=None, base_url=None):
        """
        Initialize LlamaModule.

        Parameters:
        - model (str): The Llama model to be used (e.g., "llama-2").
        - api_key (str): Your Llama API key.
        - base_url (str): The base URL of the Llama API.
        """
        # Load environment variables from a .env file
        load_dotenv()

        # Set up Llama API client
        self.model = model
        self.api_url = os.getenv("LOCAL_MODEL_API_URL")

    def generate_table_of_contents(self, document_text):
        """
        Generate a table of contents for the given document using Llama.

        Parameters:
        - document_text (str): The original content of the document.

        Returns:
        - table_of_contents (str): The generated table of contents.
        """
        prompt = (
            "Please generate a table of contents for the following document. The table of contents should include chapter and subchapter titles:\n"
            f"Document text: \n{document_text}\n\n"
            "Table of contents:"
        )
        messages = [
            {"role": "system",
             "content": "You are an assistant helping the user generate a table of contents."},
            {"role": "user", "content": prompt}
        ]
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2000,
        }

        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            result = response.json()
            # Get the complete answer content
            table_of_contents = str(result['choices'][0]['message']['content']).strip()
            return table_of_contents
        except Exception as e:
            print(f"Llama API Error: {e}")
            return None

    def decompose_question(self, question, table_of_contents, max_retries=5):
        """
        Decompose a user question into sub-questions based on a generated table of contents.

        Parameters:
        - question (str): The user question.
        - table_of_contents (str): The table of contents generated by Llama.

        Returns:
        - subquestions (list): A list of sub-questions derived from the original question.
        """
        prompt = (
            f"Table of contents:\n{table_of_contents}\n\n"
            f"Original question: {question}\n\n"
            "Based on the table of contents, break down the original question into **multiple sub-questions**. "
            "Ensure that you generate **more than one sub-question**, and each sub-question should be directly related to the original question without extending beyond its scope. "
            "Format the output as follows:\n"
            "1. Sub-question1\n"
            "2. Sub-question2\n"
            "...\n"
            "Sub-questions:"
        )
        messages = [
            {"role": "system",
             "content": "You are an assistant helping the user decompose questions into sub-questions."},
            {"role": "user", "content": prompt}
        ]
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2000,
        }

        retries = 0
        while retries < max_retries:
            try:
                # Try generating sub-questions
                response = requests.post(self.api_url, json=data)
                response.raise_for_status()
                result = response.json()

                # Get the complete answer content
                subquestions = str(result['choices'][0]['message']['content']).strip()
                print(f"Attempt {retries + 1} - Generated subquestions:\n{subquestions}")

                # Split the sub-questions into a list, removing empty lines and numbers
                subquestion_list = [subq.strip() for subq in subquestions.split('\n') if subq.strip()]

                # Check if the number of sub-questions is greater than 1
                if len(subquestion_list) > 1:
                    return subquestion_list  # Return the list of sub-questions that meet the condition

                # If not, increase the retry count
                retries += 1
                print(f"Warning: Only {len(subquestion_list)} sub-question(s) generated. Retrying...")

            except Exception as e:
                print(f"Llama API Error: {e}")
                retries += 1

        # If the maximum number of retries is reached and still not meeting the condition, return None or a default value
        print(f"Failed to generate more than one sub-question after {max_retries} attempts.")
        return None

    def is_complex_question(self, question, table_of_contents):
        """
        Determine if the question is complex, i.e., if it requires information from multiple sections of the table of contents to be fully answered.

        Parameters:
        - question (str): The user's question.
        - table_of_contents (str): The generated table of contents.

        Returns:
        - bool: True if the question requires multiple sections, otherwise False.
        """
        prompt = (
            "Given the table of contents listed below, determine if the user's question is complex and "
            "requires information from multiple sections to be fully answered. Answer with 'Yes' if the question is complex, "
            "or 'No' if it is not.\n\n"
            f"Table of contents:\n{table_of_contents}\n\n"
            f"User question: {question}\n"
        )
        messages = [
            {"role": "system",
             "content": "You are an assistant tasked with evaluating the complexity of questions."},
            {"role": "user", "content": prompt}
        ]
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 50,
        }

        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            result = response.json()
            # Get the complete answer content
            answer = str(result['choices'][0]['message']['content']).strip()
            return True if "yes" in answer.lower() else False
        except Exception as e:
            print(f"Llama API Error: {e}")
            return False

    def decompose_question_no_content(self, question, max_retries=5):
        """
        Decompose a user question into sub-questions without using a table of contents.

        Parameters:
        - question (str): The user question.

        Returns:
        - subquestions (list): A list of sub-questions derived from the original question.
        """
        prompt = (
            f"Original question: {question}\n\n"
            "Break down the original question into **multiple sub-questions**. "
            "Ensure that you generate **more than one sub-question**, and each sub-question should be directly related to the original question without extending beyond its scope. "
            "Format the output as follows:\n"
            "1. Sub-question1\n"
            "2. Sub-question2\n"
            "...\n"
            "Sub-questions:"
        )
        messages = [
            {"role": "system",
             "content": "You are an assistant helping the user decompose questions into sub-questions."},
            {"role": "user", "content": prompt}
        ]
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2000,
        }

        retries = 0
        while retries < max_retries:
            try:
                # Try generating sub-questions
                response = requests.post(self.api_url, json=data)
                response.raise_for_status()
                result = response.json()

                # Get the complete answer content
                subquestions = str(result['choices'][0]['message']['content']).strip()
                print(f"Attempt {retries + 1} - Generated subquestions:\n{subquestions}")

                # Split the sub-questions into a list, removing empty lines and numbers
                subquestion_list = [subq.strip() for subq in subquestions.split('\n') if subq.strip()]

                # Check if the number of sub-questions is greater than 1
                if len(subquestion_list) > 1:
                    return subquestion_list  # Return the list of sub-questions that meet the condition

                # If not, increase the retry count
                retries += 1
                print(f"Warning: Only {len(subquestion_list)} sub-question(s) generated. Retrying...")

            except Exception as e:
                print(f"Llama API Error: {e}")
                retries += 1

        # If the maximum number of retries is reached and still not meeting the condition, return None or a default value
        print(f"Failed to generate more than one sub-question after {max_retries} attempts.")
        return None




