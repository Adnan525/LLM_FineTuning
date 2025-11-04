import json, re, os
from pathlib import Path

class FinetuneDataGenerator:
    def __init__(self, 
                 inference_filepath: str, 
                 user_prompt_path: str) -> None: # actual dataset for user prompts
        self.inference_filepath = inference_filepath
        self.user_prompt_path = user_prompt_path
        self.data = self._load_data()
        self.user_prompts = self._load_user_prompts()

    
    def generate_data(self, 
                      regex: str,
                      output_path: str = Path(__file__).parent / "finetune_data.json") -> None:
        extracted_data = self._extract_data(regex)
        ft_data = []
        for prompt, inference in zip(self.user_prompts, extracted_data):
            ft_data.append({"q": prompt, "a": inference})
        print(f"Saving {len(ft_data)} entries to {output_path}")
        self._save_as_json(ft_data, output_path)
    

    def _save_as_json(self, data: list, output_path: str) -> None:
        with open(output_path, "w") as file:
            json.dump(data, file, indent=4)


    def _load_user_prompts(self) -> list:
        prefix = ("Following is a function signature for a python program. "
        "Please complete the function according to the signature and docstring - \n")
        
        with open(self.user_prompt_path, "r") as file:
            data = [prefix + json.loads(line).get("prompt") for line in file]
        return data


    def _load_data(self) -> list:
        with open(self.inference_filepath, "r") as file:
            data = file.read()
        return data
    

    def _extract_data(self, regex: str) -> list:
        data = re.findall(regex, self.data, re.DOTALL)
        return [self._process_data(entry) for entry in data if entry.strip()]


    def _process_data(self, text: str) -> str:
        text = self._truncate_data(text)
        text = self._clean_data(text)
        return text


    def _clean_data(self, text: str) -> str:
        start = text.find("#")
        end_ticks = [match.end() for match in re.finditer("```", text)]
        end = end_ticks[-1]
        return text[start:end].strip()


    def _truncate_data(self, text: str) -> str:
        """
        Returns the first attempt since the subsequent attempts will have debuggin reasoning.

        Args:
            text (str): Input text containing multiple attempts.

        Returns:
            str: Truncated text containing only the first attempt.
        """
        return text.split("Container found")[0]


if __name__ == "__main__":
    data_generator = FinetuneDataGenerator("data/humaneval_qwen.txt", 
                                           "data/human-eval-v2-20210705_main.jsonl")
    regex = r"QWEN2\.5-CODER(.*?)(?=##################################################)"
    data_generator.generate_data(regex)