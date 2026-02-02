import re
from typing import Optional, Union, Tuple

from base import LLM

class GLNSSampler:
    def __init__(self, llm: LLM):
        self.llm = llm

    def get_code(self, prompt: str, return_raw: bool = False) -> Union[Optional[str], Tuple[Optional[str], str]]:
        response = self.llm.draw_sample(prompt)
        code = self._extract_code_block(response)
        if return_raw:
            return code, response
        return code

    def _extract_code_block(self, response: str) -> str | None:
        # Match ```python ... ``` or just ``` ... ```
        pattern = r"```(?:python)?\s*(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
