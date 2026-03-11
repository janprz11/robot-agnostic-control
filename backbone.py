# -*- coding: ascii -*-

import abc
import base64
from copy import copy, deepcopy
from io import BytesIO
import logging
from multiprocessing import Lock
import openai
import os
from PIL import Image
import torch
import transformers
from typing import Any

__all__ = ["Llama1B", "Llama3B", "Llama11B", "ChatGPT41", "ChatGPT41Mini"]

_pipeline_locks: dict[int, Lock] = dict()
_reference_counter: dict[str, int] = dict()

HF_TOKEN = os.environ.get("HF_API_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# alphabetical ordering within each inheritance level


class AbstractLargeModel(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self, system_prompt: str | None = None, use_history: bool = True, **kwargs: Any
    ):
        pass

    @property
    @abc.abstractmethod
    def MODEL_ID(self) -> str:
        pass

    @abc.abstractmethod
    def clone(self) -> "AbstractLargeModel":
        pass

    @abc.abstractmethod
    def reset(self, system_prompt: str | None, use_history: bool) -> None:
        pass


class AbstractLLM(AbstractLargeModel):

    @abc.abstractmethod
    def __call__(self, prompt: str, max_new_tokens: int = 256) -> str:
        pass


class AbstractVLM(AbstractLargeModel):

    @abc.abstractmethod
    def __call__(
        self,
        prompt: str,
        image: Image.Image | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        pass


class AbstractLocalLLM(AbstractLLM):

    def __init__(
        self,
        system_prompt: str | None = None,
        use_history: bool = True,
        device: Any = None,
        pipeline: transformers.Pipeline | None = None,
    ):
        global _pipeline_locks, _reference_counter

        # select the device
        if device is None and pipeline is not None:
            self._device = pipeline.device
        elif device is None:
            self._device = max(
                range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.get_device_properties(i).total_memory,
            )
        else:
            self._device = device
        if isinstance(self._device, int):
            self._device = torch.device(f"cuda:{self._device}")

        # setup the pipeline
        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = self._make_pipeline()

        # setup lock and reference counter
        self._pipeline_id = id(self._pipeline)
        if self._pipeline_id not in _pipeline_locks:
            _pipeline_locks[self._pipeline_id] = Lock()
        self._lock = _pipeline_locks[self._pipeline_id]
        if self._pipeline_id not in _reference_counter:
            _reference_counter[self._pipeline_id] = 0
        _reference_counter[self._pipeline_id] += 1

        self.reset(system_prompt, use_history)

    def __call__(self, prompt: str, max_new_tokens: int = 256) -> str:
        # build payload from history and append user prompt
        if hasattr(self, "history"):
            payload = self.history
        else:
            payload = copy(self._history)
        payload.append({"role": "user", "content": prompt})

        with self._lock:
            response = self._send_payload(payload, max_new_tokens=max_new_tokens)

        payload.append({"role": "assistant", "content": response})
        return response

    def __del__(self) -> None:
        global _pipeline_locks, _reference_counter

        # decrement reference count and free GPU memory if last reference
        with self._lock:
            _reference_counter[self._pipeline_id] -= 1
            to_delete = _reference_counter[self._pipeline_id] == 0

        if to_delete:
            del _pipeline_locks[self._pipeline_id]
            del _reference_counter[self._pipeline_id]
            self._pipeline.model.to("cpu")
            del self._pipeline
            torch.cuda.empty_cache()

    @abc.abstractmethod
    def _make_pipeline(self) -> None:
        pass

    @abc.abstractmethod
    def _send_payload(
        self, payload: list[dict[str, Any]], max_new_tokens: int = 256
    ) -> str:
        pass

    def clone(self) -> "AbstractLocalLLM":
        # create a new instance sharing the same pipeline, then copy history
        rv = type(self)(
            system_prompt=self._system_prompt,
            use_history=self._use_history,
            pipeline=self._pipeline,
        )

        if hasattr(self, "history"):
            assert not hasattr(rv, "_history")
            rv.history = deepcopy(self.history)
        else:
            assert not hasattr(rv, "history")
            rv._history = deepcopy(self._history)

        return rv

    def reset(self, system_prompt: str | None, use_history: bool) -> None:
        # set up history list and toggle between mutable and immutable modes
        history = list()
        self._system_prompt = system_prompt
        if self._system_prompt is not None:
            history.append({"role": "system", "content": self._system_prompt})
        self._use_history = use_history

        if self._use_history:
            self.history = history
            if hasattr(self, "_history"):
                del self._history
        else:
            self._history = history
            if hasattr(self, "history"):
                del self.history


class AbstractLocalVLM(AbstractVLM):

    def __init__(
        self,
        system_prompt: str | None = None,
        use_history: bool = True,
        device: Any = None,
        model: Any = None,
        processor: Any = None,
    ):
        global _pipeline_locks, _reference_counter

        # select the device
        if device is None and model is not None:
            self._device = model.device
        elif device is None:
            self._device = max(
                range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.get_device_properties(i).total_memory,
            )
        else:
            self._device = device
        if isinstance(self._device, int):
            self._device = torch.device(f"cuda:{self._device}")

        # setup the model and processor
        if model is not None:
            assert processor is not None
            self._model = model
            self._processor = processor
        else:
            assert processor is None
            self._model = self._make_model()
            self._processor = self._make_processor()

        # setup lock and reference counter
        self._pipeline_id = id(self._model)
        if self._pipeline_id not in _pipeline_locks:
            _pipeline_locks[self._pipeline_id] = Lock()
        self._lock = _pipeline_locks[self._pipeline_id]
        if self._pipeline_id not in _reference_counter:
            _reference_counter[self._pipeline_id] = 0
        _reference_counter[self._pipeline_id] += 1

        self.reset(system_prompt, use_history)

    def __call__(
        self,
        prompt: str,
        image: Image.Image | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        # build multimodal payload with optional image and text
        if hasattr(self, "history"):
            payload = self.history
        else:
            payload = copy(self._history)
        payload.append({"role": "user", "content": list()})
        if image is not None:
            payload[-1]["content"].append({"type": "image", "image": image})
        payload[-1]["content"].append({"type": "text", "text": prompt})

        with self._lock:
            response = self._send_payload(payload, max_new_tokens=max_new_tokens)

        payload.append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )
        return response

    def __del__(self) -> None:
        global _pipeline_locks, _reference_counter

        # decrement reference count and free GPU memory if last reference
        with self._lock:
            _reference_counter[self._pipeline_id] -= 1
            to_delete = _reference_counter[self._pipeline_id] == 0

        if to_delete:
            del _pipeline_locks[self._pipeline_id]
            del _reference_counter[self._pipeline_id]
            self._model.to("cpu")
            del self._model
            del self._processor
            torch.cuda.empty_cache()

    @abc.abstractmethod
    def _make_model(self) -> None:
        pass

    @abc.abstractmethod
    def _make_processor(self) -> None:
        pass

    @abc.abstractmethod
    def _send_payload(
        self, payload: list[dict[str, Any]], max_new_tokens: int = 256
    ) -> str:
        pass

    def clone(self) -> "AbstractLocalVLM":
        rv = type(self)(
            system_prompt=self._system_prompt,
            use_history=self._use_history,
            model=self._model,
            processor=self._processor,
        )

        if hasattr(self, "history"):
            assert not hasattr(rv, "_history")
            rv.history = deepcopy(self.history)
        else:
            assert not hasattr(rv, "history")
            rv._history = deepcopy(self._history)

        return rv

    def reset(self, system_prompt: str | None, use_history: bool) -> None:
        history = list()
        self._system_prompt = system_prompt
        if self._system_prompt is not None:
            history.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._system_prompt}],
                }
            )
        self._use_history = use_history

        if self._use_history:
            self.history = history
            if hasattr(self, "_history"):
                del self._history
        else:
            self._history = history
            if hasattr(self, "history"):
                del self.history


class AbstractRemoteLLM(AbstractLLM):

    def __init__(
        self,
        system_prompt: str | None = None,
        use_history: bool = True,
        client: Any = None,
    ):
        global _pipeline_locks, _reference_counter

        # set up API client and register lock and reference counter
        if client is not None:
            self._client = client
        else:
            self._client = self._make_client()

        self._pipeline_id = id(self._client)
        if self._pipeline_id not in _pipeline_locks:
            _pipeline_locks[self._pipeline_id] = Lock()
        self._lock = _pipeline_locks[self._pipeline_id]
        if self._pipeline_id not in _reference_counter:
            _reference_counter[self._pipeline_id] = 0
        _reference_counter[self._pipeline_id] += 1

        self.reset(system_prompt, use_history)

    def __call__(self, prompt: str, max_new_tokens: int = 256) -> str:
        if hasattr(self, "history"):
            payload = self.history
        else:
            payload = copy(self._history)
        payload.append({"role": "user", "content": prompt})

        with self._lock:
            response = self._send_payload(payload, max_new_tokens=max_new_tokens)

        payload.append({"role": "assistant", "content": response})
        return response

    def __del__(self) -> None:
        global _pipeline_locks, _reference_counter

        with self._lock:
            _reference_counter[self._pipeline_id] -= 1
            to_delete = _reference_counter[self._pipeline_id] == 0

        if to_delete:
            del _pipeline_locks[self._pipeline_id]
            del _reference_counter[self._pipeline_id]
            del self._client

    @abc.abstractmethod
    def _make_client(self) -> None:
        pass

    @abc.abstractmethod
    def _send_payload(
        self, payload: list[dict[str, Any]], max_new_tokens: int = 256
    ) -> str:
        pass

    def clone(self) -> "AbstractRemoteLLM":
        rv = type(self)(
            system_prompt=self._system_prompt,
            use_history=self._use_history,
            client=self._client,
        )

        if hasattr(self, "history"):
            assert not hasattr(rv, "_history")
            rv.history = deepcopy(self.history)
        else:
            assert not hasattr(rv, "history")
            rv._history = deepcopy(self._history)

        return rv

    def reset(self, system_prompt: str | None, use_history: bool) -> None:
        history = list()
        self._system_prompt = system_prompt
        if self._system_prompt is not None:
            history.append({"role": "system", "content": self._system_prompt})
        self._use_history = use_history

        if self._use_history:
            self.history = history
            if hasattr(self, "_history"):
                del self._history
        else:
            self._history = history
            if hasattr(self, "history"):
                del self.history


class AbstractRemoteVLM(AbstractVLM):

    def __init__(
        self,
        system_prompt: str | None = None,
        use_history: bool = True,
        client: Any = None,
    ):
        global _pipeline_locks, _reference_counter

        if client is not None:
            self._client = client
        else:
            self._client = self._make_client()

        self._pipeline_id = id(self._client)
        if self._pipeline_id not in _pipeline_locks:
            _pipeline_locks[self._pipeline_id] = Lock()
        self._lock = _pipeline_locks[self._pipeline_id]
        if self._pipeline_id not in _reference_counter:
            _reference_counter[self._pipeline_id] = 0
        _reference_counter[self._pipeline_id] += 1

        self.reset(system_prompt, use_history)

    def __call__(
        self,
        prompt: str,
        image: Image.Image | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        # build multimodal payload with optional image and text
        if hasattr(self, "history"):
            payload = self.history
        else:
            payload = copy(self._history)
        payload.append({"role": "user", "content": list()})
        if image is not None:
            payload[-1]["content"].append({"type": "image", "image": image})
        payload[-1]["content"].append({"type": "text", "text": prompt})

        with self._lock:
            response = self._send_payload(payload, max_new_tokens=max_new_tokens)

        payload.append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )
        return response

    def __del__(self) -> None:
        global _pipeline_locks, _reference_counter

        with self._lock:
            _reference_counter[self._pipeline_id] -= 1
            to_delete = _reference_counter[self._pipeline_id] == 0

        if to_delete:
            del _pipeline_locks[self._pipeline_id]
            del _reference_counter[self._pipeline_id]
            del self._client

    @abc.abstractmethod
    def _make_client(self) -> None:
        pass

    @abc.abstractmethod
    def _send_payload(
        self, payload: list[dict[str, Any]], max_new_tokens: int = 256
    ) -> str:
        pass

    def clone(self) -> "AbstractRemoteVLM":
        rv = type(self)(
            system_prompt=self._system_prompt,
            use_history=self._use_history,
            client=self._client,
        )

        if hasattr(self, "history"):
            assert not hasattr(rv, "_history")
            rv.history = deepcopy(self.history)
        else:
            assert not hasattr(rv, "history")
            rv._history = deepcopy(self._history)

        return rv

    def reset(self, system_prompt: str | None, use_history: bool) -> None:
        history = list()
        self._system_prompt = system_prompt
        if self._system_prompt is not None:
            history.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._system_prompt}],
                }
            )
        self._use_history = use_history

        if self._use_history:
            self.history = history
            if hasattr(self, "_history"):
                del self._history
        else:
            self._history = history
            if hasattr(self, "history"):
                del self.history


class AbstractChatGPT(AbstractRemoteVLM):

    def _make_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=OPENAI_API_KEY)

    def _send_payload(
        self, payload: list[dict[str, Any]], max_new_tokens: int = 256
    ) -> str:
        # reformat the payload to match the OpenAI API
        messages = []
        for message in payload:
            if message["role"] == "system":
                messages.append(
                    {"role": "system", "content": message["content"][0]["text"]}
                )
            elif message["role"] == "user":
                content = []
                for item in message["content"]:
                    if item["type"] == "text":
                        content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image":
                        # encode PIL image to base64 PNG for the OpenAI API
                        try:
                            buffered = BytesIO()
                            item["image"].save(buffered, format="PNG")
                            encoded_image = base64.b64encode(
                                buffered.getvalue()
                            ).decode("utf-8")
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{encoded_image}"
                                    },
                                }
                            )
                        except (OSError, IOError) as e:
                            logging.warning(
                                f"Failed to process image: {e}. Skipping image."
                            )
                            content.append(
                                {
                                    "type": "text",
                                    "text": "[Image processing failed - corrupted or truncated image]",
                                }
                            )
                messages.append({"role": "user", "content": content})
            elif message["role"] == "assistant":
                messages.append(
                    {"role": "assistant", "content": message["content"][0]["text"]}
                )

        # send formatted messages to the OpenAI completions endpoint
        response = self._client.chat.completions.create(
            model=self.MODEL_ID.split("/")[1],
            messages=messages,
            max_completion_tokens=max_new_tokens,
        )
        return response.choices[0].message.content


class AbstractLlamaLLM(AbstractLocalLLM):

    def _make_pipeline(self) -> transformers.Pipeline:
        return transformers.pipeline(
            "text-generation",
            device=self._device,
            model=self.MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
        )

    def _send_payload(
        self, payload: list[dict[str, Any]], max_new_tokens: int = 256
    ) -> str:
        outputs = self._pipeline(payload, max_new_tokens=max_new_tokens)
        return outputs[0]["generated_text"][-1]["content"]


class AbstractLlamaVLM(AbstractLocalVLM):

    def _make_model(self) -> transformers.MllamaForConditionalGeneration:
        return transformers.MllamaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
        ).to(device=self._device)

    def _make_processor(self) -> transformers.AutoProcessor:
        return transformers.AutoProcessor.from_pretrained(self.MODEL_ID, token=HF_TOKEN)

    def _send_payload(
        self, payload: list[dict[str, Any]], max_new_tokens: int = 256
    ) -> str:
        # get all the images from the payload
        images = list()
        for message in payload:
            for item in message["content"]:
                if item["type"] == "image":
                    images.append(item["image"])

        # prepare the input
        input_text = self._processor.apply_chat_template(
            payload, add_generation_prompt=True
        )
        inputs = self._processor(
            images if len(images) > 0 else None,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self._device)

        # generate the output and return it
        output = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self._processor.decode(output[0])
        return response.split("<|end_header_id|>")[-1].removesuffix("<|eot_id|>")


class ChatGPT41(AbstractChatGPT):

    @property
    def MODEL_ID(self) -> str:
        return "OpenAI/gpt-4.1"


class ChatGPT41Mini(AbstractChatGPT):

    @property
    def MODEL_ID(self) -> str:
        return "OpenAI/gpt-4.1-mini"


class Llama1B(AbstractLlamaLLM):

    @property
    def MODEL_ID(self) -> str:
        return "meta-llama/Llama-3.2-1B-Instruct"


class Llama3B(AbstractLlamaLLM):

    @property
    def MODEL_ID(self) -> str:
        return "meta-llama/Llama-3.2-3B-Instruct"


class Llama11B(AbstractLlamaVLM):

    @property
    def MODEL_ID(self) -> str:
        return "meta-llama/Llama-3.2-11B-Vision-Instruct"


class Llama11B_FP8(AbstractLlamaVLM):

    def _make_model(self) -> transformers.AutoModelForCausalLM:
        return transformers.AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(device=self._device)

    @property
    def MODEL_ID(self) -> str:
        return "RedHatAI/Llama-3.2-11B-Vision-Instruct-FP8-dynamic"
