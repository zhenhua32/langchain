import asyncio
import inspect
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    cast,
)

from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain.globals import get_llm_cache
from langchain.load.dump import dumpd, dumps
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
    PromptValue,
    RunInfo,
)
from langchain.schema.language_model import BaseLanguageModel, LanguageModelInput
from langchain.schema.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.schema.runnable import RunnableConfig


def _get_verbosity() -> bool:
    from langchain.globals import get_verbose

    return get_verbose()


def _generate_from_stream(stream: Iterator[ChatGenerationChunk]) -> ChatResult:
    generation: Optional[ChatGenerationChunk] = None
    for chunk in stream:
        if generation is None:
            generation = chunk
        else:
            generation += chunk
    assert generation is not None
    return ChatResult(generations=[generation])


async def _agenerate_from_stream(
    stream: AsyncIterator[ChatGenerationChunk],
) -> ChatResult:
    generation: Optional[ChatGenerationChunk] = None
    async for chunk in stream:
        if generation is None:
            generation = chunk
        else:
            generation += chunk
    assert generation is not None
    return ChatResult(generations=[generation])


"""
这是一个Python代码片段，用于定义一个名为`BaseChatModel`的抽象类，它是一个用于聊天模型的基类，继承了`BaseLanguageModel`类，并提供了一些方法来根据输入的消息生成聊天内容¹。

`BaseLanguageModel[BaseMessage]`是一个泛型类，它表示这个类可以接受和返回`BaseMessage`类型的对象。`BaseMessage`是一个用于表示聊天消息的基类，它包含了消息的文本、元数据、标签等属性²。使用泛型类可以让代码更灵活和可重用，因为它可以适应不同类型的输入和输出³。

`ABC`是一个用于创建抽象类的元类，它表示这个类不能被实例化，只能被其他类继承，并且必须实现一些抽象方法。使用抽象类可以让代码更规范和清晰，因为它可以定义一些通用的接口和逻辑，让子类根据具体的需求来实现。

以上就是我对这个代码片段的解释。希望对你有帮助。😊

源: 与必应的对话， 2023/10/28
(1) langchain.schema.language_model .BaseLanguageModel. https://api.python.langchain.com/en/latest/schema/langchain.schema.language_model.BaseLanguageModel.html.
(2) BaseLanguageModel<RunOutput, CallOptions> | ️ Langchain. https://js.langchain.com/docs/api/base_language/classes/BaseLanguageModel.
(3) BaseLanguageModel class - langchain library - Dart API - Pub. https://pub.dev/documentation/langchain/latest/langchain/BaseLanguageModel-class.html.
"""

class BaseChatModel(BaseLanguageModel[BaseMessage], ABC):
    # 看看这个模型类需要实现哪些方法, 我要自己继承一个
    """Base class for Chat models."""

    cache: Optional[bool] = None
    """Whether to cache the response."""
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to add to the run trace."""
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)  # 字段已经废弃了
    """Callback manager to add to the run trace."""
    tags: Optional[List[str]] = Field(default=None, exclude=True)
    """Tags to add to the run trace."""
    metadata: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    """Metadata to add to the run trace."""

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        """Raise deprecation warning if callback_manager is used."""
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    class Config:
        """Configuration for this pydantic object."""

        """
        GitHub Copilot: `arbitrary_types_allowed = True` 是 Pydantic 模型的一个配置选项，用于允许模型中包含任意类型的属性。
        默认情况下，Pydantic 只允许使用预定义的数据类型，例如 int、str、float 等。
        如果需要在模型中使用自定义类型，需要将 `arbitrary_types_allowed` 设置为 True。
        这个选项通常在模型中定义了自定义类型的时候使用。
        """
        arbitrary_types_allowed = True

    # --- Runnable methods ---

    @property
    def OutputType(self) -> Any:
        # 定义输出类型
        """Get the output type for this runnable."""
        return AnyMessage

    def _convert_input(self, input: LanguageModelInput) -> PromptValue:
        """
        转换输入
        """
        if isinstance(input, PromptValue):
            return input
        elif isinstance(input, str):
            return StringPromptValue(text=input)
        elif isinstance(input, list):
            return ChatPromptValue(messages=input)
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """
        调用
        """
        config = config or {}
        return cast(
            ChatGeneration,  # 转换成这个类型, 仅仅是表示理想的输出应该是这个类型, 实际上 cast 函数什么都不做
            # 调用生成方法
            self.generate_prompt(
                # 先转换输入
                [self._convert_input(input)],
                stop=stop,
                # 都是从 config 中获取的
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                **kwargs,
            ).generations[0][0],
        ).message

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """
        异步的, TODO: 还没看
        """
        config = config or {}
        llm_result = await self.agenerate_prompt(
            [self._convert_input(input)],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            **kwargs,
        )
        return cast(ChatGeneration, llm_result.generations[0][0]).message

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        """
        流式的, TODO: 还没看
        """
        if type(self)._stream == BaseChatModel._stream:
            # model doesn't implement streaming, so use default implementation
            yield cast(
                BaseMessageChunk, self.invoke(input, config=config, stop=stop, **kwargs)
            )
        else:
            config = config or {}
            messages = self._convert_input(input).to_messages()
            params = self._get_invocation_params(stop=stop, **kwargs)
            options = {"stop": stop, **kwargs}
            callback_manager = CallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                config.get("metadata"),
                self.metadata,
            )
            (run_manager,) = callback_manager.on_chat_model_start(
                dumpd(self),
                [messages],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
            )
            try:
                generation: Optional[ChatGenerationChunk] = None
                for chunk in self._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                ):
                    yield chunk.message
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                run_manager.on_llm_error(e)
                raise e
            else:
                run_manager.on_llm_end(
                    LLMResult(generations=[[generation]]),
                )

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        if type(self)._astream == BaseChatModel._astream:
            # model doesn't implement streaming, so use default implementation
            yield cast(
                BaseMessageChunk, self.invoke(input, config=config, stop=stop, **kwargs)
            )
        else:
            config = config or {}
            messages = self._convert_input(input).to_messages()
            params = self._get_invocation_params(stop=stop, **kwargs)
            options = {"stop": stop, **kwargs}
            callback_manager = AsyncCallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                config.get("metadata"),
                self.metadata,
            )
            (run_manager,) = await callback_manager.on_chat_model_start(
                dumpd(self),
                [messages],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
            )
            try:
                generation: Optional[ChatGenerationChunk] = None
                async for chunk in self._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                ):
                    yield chunk.message
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                await run_manager.on_llm_error(e)
                raise e
            else:
                await run_manager.on_llm_end(
                    LLMResult(generations=[[generation]]),
                )

    # --- Custom methods ---

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return {}

    def _get_invocation_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        # 加上自身的参数, 合并起来
        params = self.dict()
        params["stop"] = stop
        return {**params, **kwargs}

    def _get_llm_string(self, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if self.is_lc_serializable():
            params = {**kwargs, **{"stop": stop}}
            param_string = str(sorted([(k, v) for k, v in params.items()]))
            llm_string = dumps(self)
            return llm_string + "---" + param_string
        else:
            params = self._get_invocation_params(stop=stop, **kwargs)
            params = {**params, **kwargs}
            return str(sorted([(k, v) for k, v in params.items()]))

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # 看来这个才是核心方法
        """Top Level call"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}

        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        run_managers = callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
        )
        results = []
        # 处理每个消息
        for i, m in enumerate(messages):
            try:
                results.append(
                    # 调用生成方法
                    self._generate_with_cache(
                        m,
                        stop=stop,
                        run_manager=run_managers[i] if run_managers else None,
                        **kwargs,
                    )
                )
            except BaseException as e:
                if run_managers:
                    run_managers[i].on_llm_error(e)
                raise e
        # 对每个输出调用 LLMResult
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        # 组合 llm_output
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        # 所有的 generations
        generations = [res.generations for res in results]
        # 再组合下, 这就是最后的输出
        output = LLMResult(generations=generations, llm_output=llm_output)
        if run_managers:
            run_infos = []
            for manager, flattened_output in zip(run_managers, flattened_outputs):
                manager.on_llm_end(flattened_output)
                run_infos.append(RunInfo(run_id=manager.run_id))
            output.run = run_infos
        return output

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        run_managers = await callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
        )

        results = await asyncio.gather(
            *[
                self._agenerate_with_cache(
                    m,
                    stop=stop,
                    run_manager=run_managers[i] if run_managers else None,
                    **kwargs,
                )
                for i, m in enumerate(messages)
            ],
            return_exceptions=True,
        )
        exceptions = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                if run_managers:
                    await run_managers[i].on_llm_error(res)
                exceptions.append(res)
        if exceptions:
            if run_managers:
                await asyncio.gather(
                    *[
                        run_manager.on_llm_end(
                            LLMResult(
                                generations=[res.generations], llm_output=res.llm_output
                            )
                        )
                        for run_manager, res in zip(run_managers, results)
                        if not isinstance(res, Exception)
                    ]
                )
            raise exceptions[0]
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        await asyncio.gather(
            *[
                run_manager.on_llm_end(flattened_output)
                for run_manager, flattened_output in zip(
                    run_managers, flattened_outputs
                )
            ]
        )
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        # 将模板转换成消息
        prompt_messages = [p.to_messages() for p in prompts]
        # 调用生成方法
        return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return await self.agenerate(
            prompt_messages, stop=stop, callbacks=callbacks, **kwargs
        )

    def _generate_with_cache(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 看下这个, 是个包装 cache 功能的
        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        # 忽略缓存. 仅当 cache 不是 None 且 cache 为 False 时
        disregard_cache = self.cache is not None and not self.cache
        llm_cache = get_llm_cache()
        if llm_cache is None or disregard_cache:
            # llm.cache 是具体实现, self.cache 是布尔值, 用来配置是否缓存
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            if new_arg_supported:
                # 调用生成方法
                return self._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                return self._generate(messages, stop=stop, **kwargs)
        else:
            # 有缓存的时候
            llm_string = self._get_llm_string(stop=stop, **kwargs)
            prompt = dumps(messages)
            # 调用 llm_cache.lookup 的方法, 从缓存中获取
            cache_val = llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                return ChatResult(generations=cache_val)
            else:
                # 必须要求 cache_val 为 list, 否则还是会调用 _generate 方法
                if new_arg_supported:
                    result = self._generate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                else:
                    result = self._generate(messages, stop=stop, **kwargs)
                # 更新缓存
                llm_cache.update(prompt, llm_string, result.generations)
                return result

    async def _agenerate_with_cache(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        new_arg_supported = inspect.signature(self._agenerate).parameters.get(
            "run_manager"
        )
        disregard_cache = self.cache is not None and not self.cache
        llm_cache = get_llm_cache()
        if llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            if new_arg_supported:
                return await self._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                return await self._agenerate(messages, stop=stop, **kwargs)
        else:
            llm_string = self._get_llm_string(stop=stop, **kwargs)
            prompt = dumps(messages)
            cache_val = llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                return ChatResult(generations=cache_val)
            else:
                if new_arg_supported:
                    result = await self._agenerate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                else:
                    result = await self._agenerate(messages, stop=stop, **kwargs)
                llm_cache.update(prompt, llm_string, result.generations)
                return result

    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 这个是要子类实现的
        """Top Level call"""

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, **kwargs), messages, stop, run_manager
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError()

    def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        raise NotImplementedError()

    def __call__(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseMessage:
        generation = self.generate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        ).generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")

    async def _call_async(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseMessage:
        result = await self.agenerate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        )
        generation = result.generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")

    def call_as_llm(
        self, message: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        return self.predict(message, stop=stop, **kwargs)

    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        result = self([HumanMessage(content=text)], stop=_stop, **kwargs)
        return result.content

    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return self(messages, stop=_stop, **kwargs)

    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        result = await self._call_async(
            [HumanMessage(content=text)], stop=_stop, **kwargs
        )
        return result.content

    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return await self._call_async(messages, stop=_stop, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of chat model."""

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the LLM."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict


class SimpleChatModel(BaseChatModel):
    """Simple Chat Model."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 调用 _call 方法, 返回字符串
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        # 封装下模型返回的字符串
        message = AIMessage(content=output_str)
        # 再封装下
        generation = ChatGeneration(message=message)
        # 最后组成 ChatResult
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 需要子类实现
        """Simpler interface."""

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 也就是说参数都已经预先传入了, 所以异步调用的时候不需要传入
        func = partial(
            self._generate, messages, stop=stop, run_manager=run_manager, **kwargs
        )
        # 具体执行, 异步执行
        return await asyncio.get_event_loop().run_in_executor(None, func)
