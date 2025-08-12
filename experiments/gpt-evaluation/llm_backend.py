# from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import typing as t


class InferenceBackend:
    """One gpu inference backend"""

    def __init__(self, model_path, seed=1337, backend_type="llama_cpp", cache_dir=None):
        self.backend_type = backend_type

        if backend_type == "llama_cpp":
            raise ValueError("Llama_cpp backend is not supported")
            # self.model = Llama(
            #     model_path=model_path,
            #     n_gpu_layers=-1,
            #     seed=seed,
            #     n_ctx=2048,  # Uncomment to increase the context window
            # )
        elif backend_type == "vllm":
            ...
        elif backend_type == "transformers":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, cache_dir=cache_dir, padding_side="left"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=quantization_config,
                cache_dir=cache_dir,
            )
        else:
            raise ValueError(f"Backend type {backend_type} not supported")

    def batch_infer(self, messages: t.List[t.List[t.Dict[str, str]]]) -> t.List[str]:
        if self.backend_type == "llama_cpp":
            outputs = self._infer_llama_cpp(messages)
        elif self.backend_type == "transformers":
            outputs = self._infer_transformers(messages)

        outputs = self._clean_output(outputs)
        return outputs

    def _clean_output(self, output: t.List[str]) -> t.List[str]:
        for i, o in enumerate(output):
            output[i] = o.replace("\"'", '"').replace("'\"", '"')

        return output

    def _infer_llama_cpp(
        self, batch_messages: t.List[t.List[t.Dict[str, str]]]
    ) -> t.List[str]:
        outputs = []
        for messages in batch_messages:
            output = self.model.create_chat_completion(
                messages=messages,
            )  # Generate a completion, can also call create_completion

            output = output["choices"][0]["message"]["content"]
            output = output.replace("<think>", "").replace("</think>", "").strip("\n")

            outputs.append(output)

        return outputs

    def _infer_transformers(
        self, batch_messages: t.List[t.List[t.Dict[str, str]]]
    ) -> t.List[str]:
        texts = []
        for messages in batch_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
            )
            texts.append(text)

        model_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(
            self.model.device
        )
        generated_batch_ids = self.model.generate(**model_inputs, max_new_tokens=32768)

        outputs = []
        for i, generated_ids in enumerate(generated_batch_ids):
            output_ids = generated_ids[len(model_inputs.input_ids[i]) :].tolist()

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            output_ids = generated_ids[len(model_inputs.input_ids[i]) :].tolist()
            content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")

            outputs.append(content)

        return outputs
