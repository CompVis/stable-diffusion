import torch
import numpy as np
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import TypeVar, List, Type, Optional
from ldm.models.diffusion.ddpm import LatentDiffusion

T = TypeVar('T')


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.


def get_step(weight: float, steps: int) -> int:
    step = int(weight) if weight >= 1. else int(weight * steps)
    return max(0, min(steps-1, step))


def parse_step(text: str, steps: int) -> int:
    return get_step(parse_float(text), steps)


@dataclass
class Guidance:
    scale: Optional[float]
    cond: Optional[object]
    uncond: Optional[object]
    prompt: str

    def apply(self, cond: object, uncond: object, scale: float):
        if self.cond is not None:
            cond = self.cond

        if self.scale is not None:
            scale = self.scale

        if self.uncond is not None:
            uncond = self.uncond

        return cond, uncond, scale


@dataclass
class Prompt:
    pos_text: str
    neg_text: str
    step: int


class Token(ABC):
    @staticmethod
    @abstractmethod
    def starts_with(char: str) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def create(text: str, steps: int):
        pass


class TextToken(Token):
    @staticmethod
    def create(text: str, steps: int):
        return text

    @staticmethod
    def starts_with(char: str) -> bool:
        return True


@dataclass
class CommandToken(Token):
    method: str
    args: List[str]
    step: int

    @staticmethod
    def create(text: str, steps: int):
        return None

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == '@'


@dataclass
class SwapToken(Token):
    word1: str = ""
    word2: str = ""
    step: int = 0

    @staticmethod
    def create(text: str, steps: int):
        value = text[1:-1]
        fields = str.split(value, ':')
        if len(fields) < 2:
            return SwapToken(word2=value)
        if len(fields) == 2:
            return SwapToken(word2=fields[0], step=parse_step(fields[1], steps))
        else:
            return SwapToken(word1=fields[0], word2=fields[1], step=parse_step(fields[2], steps))

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == '['


@dataclass
class ScaleToken(Token):
    scale: float = -1.
    step: int = 0

    @staticmethod
    def create(text: str, steps: int):
        fields = str.split(text[1:-1], ':')
        if len(fields) != 2:
            return ScaleToken()

        return ScaleToken(scale=parse_float(fields[0]), step=parse_step(fields[1], steps))

    @staticmethod
    def starts_with(char: str) -> bool:
        return char == '{'


def filter_type(array: list, dtype: Type[T]) -> List[T]:
    return [item for item in array if type(item) is dtype]


class PromptParser:
    def __init__(self, model):
        self.model = model
        self.regex = re.compile(r'\[.*?]|\{.*?}|.+?(?=[\[{])|.*')
        self.tokens = [SwapToken, ScaleToken, TextToken]

    # test regex for commands, not used yet
    # \[.*?]|\{.+?}|@[^\s(]+\(.*?\)|.+?(?=[\[{@])|.+

    def get_prompt_guidance(self, prompt, steps, batch_size) -> List[Guidance]:
        result: List[Guidance] = list()

        # initialize array
        for i in range(0, steps):
            result.append(Guidance(None, None, None, ""))

        cur_pos = ""
        cur_neg = ""
        # set prompts
        print("Used prompts:")
        for item in self.__parse_prompt(prompt, steps):
            if item.pos_text != cur_pos:
                print(f'step {item.step}: "{item.pos_text}"')
                result[item.step].cond = self.model.get_learned_conditioning(batch_size * item.pos_text)
                cur_pos = item.pos_text

            if item.neg_text != cur_neg:
                print(f'step {item.step}: [negative] "{item.neg_text}"')
                result[item.step].uncond = self.model.get_learned_conditioning(batch_size * item.neg_text)
                cur_neg = item.neg_text

            result[item.step].prompt = cur_pos

        # set scales
        for scale in self.__get_scales(prompt, steps):
            result[scale.step].scale = scale.scale

        return result

    def __get_scales(self, prompt: str, steps: int) -> List[ScaleToken]:
        tokens = self.__get_tokens(prompt, steps)
        scales = filter_type(tokens, ScaleToken)

        return scales

    def __get_word_info(self, word: str) -> (str, bool):
        if len(word) == 0:
            return word, False

        if word[0] == '-':
            return word[1:], False

        return word, True

    def __parse_prompt(self, prompt, steps) -> List[Prompt]:
        tokens = self.__get_tokens(prompt, steps)
        values = np.array([token.step for token in filter_type(tokens, SwapToken)])
        values = np.concatenate(([0], values))
        values = np.sort(np.unique(values))

        builders = [(value, list(), list()) for value in values]

        for token in tokens:
            if type(token) is SwapToken:
                word1, is_pos1 = self.__get_word_info(token.word1)
                word2, is_pos2 = self.__get_word_info(token.word2)

                if not len(word2):
                    is_pos2 = is_pos1

                for (value, pos_text, neg_text) in builders:
                    if value < token.step:
                        is_pos, word = is_pos1, word1
                    else:
                        is_pos, word = is_pos2, word2

                    builder = pos_text if is_pos else neg_text
                    builder.append(word)

            elif type(token) is str:
                for _, pos_text, _ in builders:
                    pos_text.append(token)

        return [Prompt(pos_text=''.join(pos_text), neg_text=''.join(neg_text), step=int(value))
                for value, pos_text, neg_text in builders]

    def __get_tokens(self, prompt: str, steps: int):
        parts = self.regex.findall(prompt)
        result = list()

        for part in parts:
            if len(part) == 0:
                continue

            for token in self.tokens:
                if token.starts_with(part[0]):
                    result.append(token.create(part, steps))
                    break

        return result


class PromptGuidanceModelWrapper:
    def __init__(self, model: LatentDiffusion):
        self.model = model

        self.__step: int = None
        self.prompt_guidance: List[Guidance] = None
        self.scale: float = 0.
        self.init_scale: float = 0.
        self.c = None
        self.uc = None
        self.parser = PromptParser(model)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if self.prompt_guidance is None:
            raise RuntimeError("Wrapper not prepared, make sure to call prepare before using the model")

        if self.__step < len(self.prompt_guidance):
            self.c, self.uc, self.scale = \
                self.prompt_guidance[self.__step].apply(self.c, self.uc, self.scale)

        has_unconditional = len(cond) == 2
        if has_unconditional:
            cond[0] = self.uc
            cond[1] = self.c
        else:
            cond = self.c

        result = self.model.apply_model(x_noisy, t, cond, return_ids)

        if has_unconditional and self.scale != self.init_scale:
            e_t_uncond, e_t = result.chunk(2)
            e_diff = e_t - e_t_uncond
            e_t = e_t_uncond + (self.scale / self.init_scale) * e_diff
            result = torch.cat([e_t_uncond, e_t])

        self.__step += 1

        return result

    def prepare_prompts(self, prompt: str, scale: float, steps: int, batch_size: int):
        self.__step = 0

        self.prompt_guidance = self.parser.get_prompt_guidance(prompt, steps, batch_size)

        uc = self.model.get_learned_conditioning(batch_size * [""])
        c, uc, scale = self.prompt_guidance[0].apply(uc, uc, scale)

        self.init_scale = scale
        self.scale = scale
        self.c = c
        self.uc = uc

