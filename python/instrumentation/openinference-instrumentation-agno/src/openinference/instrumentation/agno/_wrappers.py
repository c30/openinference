import json
import time
from enum import Enum
from inspect import signature
from secrets import token_hex
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
    cast,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from agno.agent import Agent
from agno.models.base import Model
from agno.team import Team
from agno.tools.function import Function, FunctionCall
from agno.tools.toolkit import Toolkit
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

_AGNO_PARENT_NODE_CONTEXT_KEY = context_api.create_key("agno_parent_node_id")


def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    arguments = _bind_arguments(method, *args, **kwargs)
    arguments = _strip_method_args(arguments)
    return safe_json_dumps(arguments)


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments
    arguments = OrderedDict(
        {key: value for key, value in arguments.items() if value is not None and value != {}}
    )
    return arguments


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


def _generate_node_id() -> str:
    return token_hex(8)  # Generates 16 hex characters (8 bytes)


def _agent_run_attributes(
    agent: Union[Agent, Team], key_suffix: str = ""
) -> Iterator[Tuple[str, AttributeValue]]:
    # Get parent from execution context instead of structural parent
    context_parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

    # Existing agent-specific attributes
    if agent.session_id:
        yield SESSION_ID, agent.session_id

    if agent.user_id:
        yield USER_ID, agent.user_id

    if isinstance(agent, Team):
        # Set graph attributes for team
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name

        # Use context parent instead of structural parent
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)

        # Set legacy team attributes
        yield f"agno{key_suffix}.team", agent.name or ""
        for member in agent.members:
            yield from _agent_run_attributes(member, f".{member.name}")

    elif isinstance(agent, Agent):
        # Set graph attributes for agent
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name

        # Use context parent instead of structural parent
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)

        # Set legacy agent attributes
        if agent.name:
            yield f"agno{key_suffix}.agent", agent.name or ""

        if agent.knowledge:
            yield f"agno{key_suffix}.knowledge", agent.knowledge.__class__.__name__

        if agent.tools:
            tool_names = []
            for tool in agent.tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend([f for f in tool.functions.keys()])
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            yield f"agno{key_suffix}.tools", tool_names


def _setup_team_context(agent: Union[Agent, Team], node_id: str) -> Optional[Any]:
    if isinstance(agent, Team):
        team_ctx = context_api.set_value(_AGNO_PARENT_NODE_CONTEXT_KEY, node_id)
        return context_api.attach(team_ctx)
    return None


class _RunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    """
    We need to keep track of parent/child relationships for agent logging. We do this by:
    1. Each run() method generates a unique node_id and sets it directly as GRAPH_NODE_ID in span
    attributes
    2. Team.run() sets _AGNO_PARENT_NODE_CONTEXT_KEY for child agents
    3. Agent.run() inherits _AGNO_PARENT_NODE_CONTEXT_KEY from team context for parent relationships
    4. _agent_run_attributes() uses _AGNO_PARENT_NODE_CONTEXT_KEY to set GRAPH_NODE_PARENT_ID
    5. This ensures correct parent-child relationships with unique node IDs for each execution
    """

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                run_response = wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                return run_response

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                yield from wrapped(*args, **kwargs)
                run_response = agent.run_response
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            response = await wrapped(*args, **kwargs)
            return response

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                run_response = await wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                return run_response
            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in await wrapped(*args, **kwargs):
                yield response

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                async for response in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                    yield response
                run_response = agent.run_response
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    def process_message(idx: int, role: str, content: str) -> Iterator[Tuple[str, Any]]:
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}", role
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}", content

    messages = arguments.get("messages", [])
    for i, message in enumerate(messages):
        role, content = message.role, message.get_content_string()
        if content:
            yield from process_message(i, role, content)

    tools = arguments.get("tools", [])
    for tool_index, tool in enumerate(tools):
        yield f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}", safe_json_dumps(tool)


def _llm_invocation_parameters(
    model: Model, arguments: Optional[Mapping[str, Any]] = None
) -> Iterator[Tuple[str, Any]]:
    request_kwargs = {}
    # TODO (v2.0.0): with the cleanup of the agno.models.base.Model class we will
    # handle these attributes in a more consistent way.
    if getattr(model, "request_kwargs", None):
        request_kwargs = model.request_kwargs  # type: ignore[attr-defined]
    if getattr(model, "request_params", None):
        request_kwargs = model.request_params  # type: ignore[attr-defined]
    if getattr(model, "get_request_kwargs", None):
        request_kwargs = model.get_request_kwargs()  # type: ignore[attr-defined]
    if getattr(model, "get_request_params", None):
        # Special handling for OpenAIResponses model
        if model.__class__.__name__ == "OpenAIResponses" and arguments:
            messages = arguments.get("messages", [])
            request_kwargs = model.get_request_params(messages=messages)  # type: ignore[attr-defined]
        else:
            request_kwargs = model.get_request_params()  # type: ignore[attr-defined]

    if request_kwargs:
        filtered_kwargs = _filter_sensitive_params(request_kwargs)
        yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(filtered_kwargs)


def _filter_sensitive_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out sensitive parameters from model request parameters."""
    sensitive_keys = frozenset(
        [
            "api_key",
            "api_base",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_access_key",
            "aws_secret_key",
            "azure_endpoint",
            "azure_deployment",
            "azure_ad_token",
            "azure_ad_token_provider",
        ]
    )

    return {
        key: "[REDACTED]"
        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys)
        else value
        for key, value in params.items()
    }


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type(output: str) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_MIME_TYPE, JSON
    yield OUTPUT_VALUE, output


def _parse_model_output(output: Any) -> str:
    if hasattr(output, "model_dump_json"):
        return output.model_dump_json()  # type: ignore[no-any-return]
    elif isinstance(output, dict):
        return json.dumps(output)
    else:
        return str(output)


def _extract_token_usage(response: Any) -> Optional[Dict[str, Any]]:
    """Extract token usage information from model response."""
    try:
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "model_dump"):
                return usage.model_dump()
            elif isinstance(usage, dict):
                return usage
            else:
                # Try to convert usage object to dict
                usage_dict = {}
                for attr in ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"]:
                    if hasattr(usage, attr):
                        usage_dict[attr] = getattr(usage, attr)
                return usage_dict if usage_dict else None
        
        # Try to extract from response dict structure
        if isinstance(response, dict) and "usage" in response:
            return response["usage"]
        
        return None
    except Exception:
        return None


def _calculate_time_per_output_token(duration_ms: float, output_tokens: Optional[int]) -> Optional[float]:
    """Calculate average time per output token in milliseconds."""
    if output_tokens and output_tokens > 0:
        return duration_ms / output_tokens
    return None


class _StreamTimingCollector:
    """Helper class to collect timing metrics for streaming operations."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.token_times: list[float] = []
        self.token_count: int = 0
    
    def start_operation(self):
        """Mark the start of the streaming operation."""
        self.start_time = time.perf_counter()
    
    def record_token(self):
        """Record the receipt of a token."""
        current_time = time.perf_counter()
        
        if self.first_token_time is None:
            self.first_token_time = current_time
        
        self.token_times.append(current_time)
        self.token_count += 1
    
    def get_time_to_first_token_ms(self) -> Optional[float]:
        """Get time to first token in milliseconds."""
        if self.start_time is not None and self.first_token_time is not None:
            return (self.first_token_time - self.start_time) * 1000
        return None
    
    def get_average_time_between_tokens_ms(self) -> Optional[float]:
        """Get average time between tokens in milliseconds."""
        if len(self.token_times) > 1:
            total_time = self.token_times[-1] - self.token_times[0]
            intervals = len(self.token_times) - 1
            return (total_time / intervals) * 1000
        return None


class _ModelWrapper:
    def __init__(self, tracer: trace_api.Tracer, metrics: Optional[Dict[str, Any]] = None) -> None:
        self._tracer = tracer
        self._metrics = metrics or {}

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.invoke"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model, arguments)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)
            span.set_attribute(GEN_AI_CLIENT_OPERATION, "invoke")

            # Record start time
            start_time = time.perf_counter()
            
            try:
                response = wrapped(*args, **kwargs)
                output_message = _parse_model_output(response)

                # Calculate operation duration
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                span.set_attribute(GEN_AI_CLIENT_OPERATION_DURATION, duration_ms)

                # Record duration to histogram
                if duration_histogram := self._metrics.get("duration_histogram"):
                    duration_histogram.record(
                        duration_ms, 
                        attributes={
                            "operation": "invoke",
                            "model": model.id,
                            "provider": model.provider,
                        }
                    )

                # Extract token usage
                token_usage = _extract_token_usage(response)
                if token_usage:
                    span.set_attribute(GEN_AI_CLIENT_TOKEN_USAGE, safe_json_dumps(token_usage))
                    
                    # Calculate time per output token
                    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                    time_per_token = _calculate_time_per_output_token(duration_ms, output_tokens)
                    if time_per_token is not None:
                        span.set_attribute(GEN_AI_CLIENT_TIME_PER_OUTPUT_TOKEN, time_per_token)
                        
                        # Record time per output token to histogram
                        if time_per_token_histogram := self._metrics.get("time_per_output_token_histogram"):
                            time_per_token_histogram.record(
                                time_per_token,
                                attributes={
                                    "operation": "invoke",
                                    "model": model.id,
                                    "provider": model.provider,
                                }
                            )

                span.set_attributes(dict(_output_value_and_mime_type(output_message)))
                return response
            except Exception as e:
                # Still record duration even on error
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                span.set_attribute(GEN_AI_CLIENT_OPERATION_DURATION, duration_ms)
                
                # Record duration to histogram even on error
                if duration_histogram := self._metrics.get("duration_histogram"):
                    duration_histogram.record(
                        duration_ms,
                        attributes={
                            "operation": "invoke",
                            "model": model.id,
                            "provider": model.provider,
                            "error": "true",
                        }
                    )
                raise

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.invoke_stream"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)
            span.set_attribute(GEN_AI_CLIENT_OPERATION, "invoke_stream")

            # Set up timing collection
            timing_collector = _StreamTimingCollector()
            timing_collector.start_operation()
            start_time = time.perf_counter()

            responses = []
            try:
                for chunk in wrapped(*args, **kwargs):
                    timing_collector.record_token()
                    responses.append(chunk)
                    yield chunk
                
                # Calculate final metrics
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                span.set_attribute(GEN_AI_CLIENT_OPERATION_DURATION, duration_ms)

                # Record duration to histogram
                if duration_histogram := self._metrics.get("duration_histogram"):
                    duration_histogram.record(
                        duration_ms,
                        attributes={
                            "operation": "invoke_stream",
                            "model": model.id,
                            "provider": model.provider,
                        }
                    )

                # Set streaming-specific timing metrics
                time_to_first_token = timing_collector.get_time_to_first_token_ms()
                if time_to_first_token is not None:
                    span.set_attribute(GEN_AI_CLIENT_TIME_TO_FIRST_TOKEN, time_to_first_token)
                    
                    # Record time to first token to histogram
                    if time_to_first_token_histogram := self._metrics.get("time_to_first_token_histogram"):
                        time_to_first_token_histogram.record(
                            time_to_first_token,
                            attributes={
                                "operation": "invoke_stream",
                                "model": model.id,
                                "provider": model.provider,
                            }
                        )

                time_between_tokens = timing_collector.get_average_time_between_tokens_ms()
                if time_between_tokens is not None:
                    span.set_attribute(GEN_AI_CLIENT_TIME_BETWEEN_TOKEN, time_between_tokens)
                    
                    # Record time between tokens to histogram
                    if time_between_tokens_histogram := self._metrics.get("time_between_tokens_histogram"):
                        time_between_tokens_histogram.record(
                            time_between_tokens,
                            attributes={
                                "operation": "invoke_stream",
                                "model": model.id,
                                "provider": model.provider,
                            }
                        )

                # Try to extract token usage from the final response or model
                if responses:
                    last_response = responses[-1]
                    token_usage = _extract_token_usage(last_response)
                    if token_usage:
                        span.set_attribute(GEN_AI_CLIENT_TOKEN_USAGE, safe_json_dumps(token_usage))
                        
                        # Calculate time per output token
                        output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or timing_collector.token_count
                        time_per_token = _calculate_time_per_output_token(duration_ms, output_tokens)
                        if time_per_token is not None:
                            span.set_attribute(GEN_AI_CLIENT_TIME_PER_OUTPUT_TOKEN, time_per_token)

                output_message = json.dumps([_parse_model_output(response) for response in responses])
                span.set_attributes(dict(_output_value_and_mime_type(output_message)))
            except Exception as e:
                # Still record duration even on error
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                span.set_attribute(GEN_AI_CLIENT_OPERATION_DURATION, duration_ms)

                # Record duration to histogram even on error
                if duration_histogram := self._metrics.get("duration_histogram"):
                    duration_histogram.record(
                        duration_ms,
                        attributes={
                            "operation": "invoke_stream",
                            "model": model.id,
                            "provider": model.provider,
                            "error": "true",
                        }
                    )
                raise

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.ainvoke"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)
            span.set_attribute(GEN_AI_CLIENT_OPERATION, "ainvoke")

            # Record start time
            start_time = time.perf_counter()
            
            try:
                response = await wrapped(*args, **kwargs)
                output_message = _parse_model_output(response)

                # Calculate operation duration
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                span.set_attribute(GEN_AI_CLIENT_OPERATION_DURATION, duration_ms)

                # Record duration to histogram
                if duration_histogram := self._metrics.get("duration_histogram"):
                    duration_histogram.record(
                        duration_ms, 
                        attributes={
                            "operation": "ainvoke",
                            "model": model.id,
                            "provider": model.provider,
                        }
                    )

                # Extract token usage
                token_usage = _extract_token_usage(response)
                if token_usage:
                    span.set_attribute(GEN_AI_CLIENT_TOKEN_USAGE, safe_json_dumps(token_usage))
                    
                    # Calculate time per output token
                    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                    time_per_token = _calculate_time_per_output_token(duration_ms, output_tokens)
                    if time_per_token is not None:
                        span.set_attribute(GEN_AI_CLIENT_TIME_PER_OUTPUT_TOKEN, time_per_token)
                        
                        # Record time per output token to histogram
                        if time_per_token_histogram := self._metrics.get("time_per_output_token_histogram"):
                            time_per_token_histogram.record(
                                time_per_token,
                                attributes={
                                    "operation": "ainvoke",
                                    "model": model.id,
                                    "provider": model.provider,
                                }
                            )

                span.set_attributes(dict(_output_value_and_mime_type(output_message)))
                return response
            except Exception as e:
                # Still record duration even on error
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                span.set_attribute(GEN_AI_CLIENT_OPERATION_DURATION, duration_ms)
                
                # Record duration to histogram even on error
                if duration_histogram := self._metrics.get("duration_histogram"):
                    duration_histogram.record(
                        duration_ms,
                        attributes={
                            "operation": "ainvoke",
                            "model": model.id,
                            "provider": model.provider,
                            "error": "true",
                        }
                    )
                raise

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                yield response
            return

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.ainvoke_stream"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)
            span.set_attribute(GEN_AI_CLIENT_OPERATION, "ainvoke_stream")

            # Set up timing collection
            timing_collector = _StreamTimingCollector()
            timing_collector.start_operation()
            start_time = time.perf_counter()

            responses = []
            try:
                async for chunk in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                    timing_collector.record_token()
                    responses.append(chunk)
                    yield chunk
                
                # Calculate final metrics
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                span.set_attribute(GEN_AI_CLIENT_OPERATION_DURATION, duration_ms)

                # Record duration to histogram
                if duration_histogram := self._metrics.get("duration_histogram"):
                    duration_histogram.record(
                        duration_ms,
                        attributes={
                            "operation": "ainvoke_stream",
                            "model": model.id,
                            "provider": model.provider,
                        }
                    )

                # Set streaming-specific timing metrics
                time_to_first_token = timing_collector.get_time_to_first_token_ms()
                if time_to_first_token is not None:
                    span.set_attribute(GEN_AI_CLIENT_TIME_TO_FIRST_TOKEN, time_to_first_token)
                    
                    # Record time to first token to histogram
                    if time_to_first_token_histogram := self._metrics.get("time_to_first_token_histogram"):
                        time_to_first_token_histogram.record(
                            time_to_first_token,
                            attributes={
                                "operation": "ainvoke_stream",
                                "model": model.id,
                                "provider": model.provider,
                            }
                        )

                time_between_tokens = timing_collector.get_average_time_between_tokens_ms()
                if time_between_tokens is not None:
                    span.set_attribute(GEN_AI_CLIENT_TIME_BETWEEN_TOKEN, time_between_tokens)
                    
                    # Record time between tokens to histogram
                    if time_between_tokens_histogram := self._metrics.get("time_between_tokens_histogram"):
                        time_between_tokens_histogram.record(
                            time_between_tokens,
                            attributes={
                                "operation": "ainvoke_stream",
                                "model": model.id,
                                "provider": model.provider,
                            }
                        )

                # Try to extract token usage from the final response or model
                if responses:
                    last_response = responses[-1]
                    token_usage = _extract_token_usage(last_response)
                    if token_usage:
                        span.set_attribute(GEN_AI_CLIENT_TOKEN_USAGE, safe_json_dumps(token_usage))
                        
                        # Calculate time per output token
                        output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or timing_collector.token_count
                        time_per_token = _calculate_time_per_output_token(duration_ms, output_tokens)
                        if time_per_token is not None:
                            span.set_attribute(GEN_AI_CLIENT_TIME_PER_OUTPUT_TOKEN, time_per_token)

                output_message = json.dumps([_parse_model_output(response) for response in responses])
                span.set_attributes(dict(_output_value_and_mime_type(output_message)))
            except Exception as e:
                # Still record duration even on error
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                span.set_attribute(GEN_AI_CLIENT_OPERATION_DURATION, duration_ms)

                # Record duration to histogram even on error
                if duration_histogram := self._metrics.get("duration_histogram"):
                    duration_histogram.record(
                        duration_ms,
                        attributes={
                            "operation": "ainvoke_stream",
                            "model": model.id,
                            "provider": model.provider,
                            "error": "true",
                        }
                    )
                raise


def _function_call_attributes(function_call: FunctionCall) -> Iterator[Tuple[str, Any]]:
    function = function_call.function
    function_name = function.name
    function_arguments = function_call.arguments

    yield TOOL_NAME, function_name

    if function_description := getattr(function, "description", None):
        yield TOOL_DESCRIPTION, function_description
    yield TOOL_PARAMETERS, safe_json_dumps(function_arguments)


def _input_value_and_mime_type_for_tool_span(
    arguments: Mapping[str, Any],
) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type_for_tool_span(result: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, str(result)
    yield OUTPUT_MIME_TYPE, TEXT


class _FunctionCallWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        span_name = f"{function_name}"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                **dict(_input_value_and_mime_type_for_tool_span(function_arguments)),
                **dict(_function_call_attributes(function_call)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            response = wrapped(*args, **kwargs)

            if response.status == "success":
                function_result = function_call.result
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(
                    dict(
                        _output_value_and_mime_type_for_tool_span(
                            result=function_result,
                        )
                    )
                )
            elif response.status == "failure":
                function_error_message = function_call.error
                span.set_status(trace_api.StatusCode.ERROR, function_error_message)
                span.set_attribute(OUTPUT_VALUE, function_error_message)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
            else:
                span.set_status(trace_api.StatusCode.ERROR, "Unknown function call status")

        return response

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        span_name = f"{function_name}"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                **dict(_input_value_and_mime_type_for_tool_span(function_arguments)),
                **dict(_function_call_attributes(function_call)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            response = await wrapped(*args, **kwargs)

            if response.status == "success":
                function_result = function_call.result
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(
                    dict(
                        _output_value_and_mime_type_for_tool_span(
                            result=function_result,
                        )
                    )
                )
            elif response.status == "failure":
                function_error_message = function_call.error
                span.set_status(trace_api.StatusCode.ERROR, function_error_message)
                span.set_attribute(OUTPUT_VALUE, function_error_message)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
            else:
                span.set_status(trace_api.StatusCode.ERROR, "Unknown function call status")

        return response


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
LLM_TOOLS = SpanAttributes.LLM_TOOLS
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_FUNCTION_CALL = SpanAttributes.LLM_FUNCTION_CALL
GEN_AI_CLIENT_OPERATION = SpanAttributes.GEN_AI_CLIENT_OPERATION
GEN_AI_CLIENT_OPERATION_DURATION = SpanAttributes.GEN_AI_CLIENT_OPERATION_DURATION
GEN_AI_CLIENT_TIME_TO_FIRST_TOKEN = SpanAttributes.GEN_AI_CLIENT_TIME_TO_FIRST_TOKEN
GEN_AI_CLIENT_TIME_BETWEEN_TOKEN = SpanAttributes.GEN_AI_CLIENT_TIME_BETWEEN_TOKEN
GEN_AI_CLIENT_TIME_PER_OUTPUT_TOKEN = SpanAttributes.GEN_AI_CLIENT_TIME_PER_OUTPUT_TOKEN
GEN_AI_CLIENT_TOKEN_USAGE = SpanAttributes.GEN_AI_CLIENT_TOKEN_USAGE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS
USER_ID = SpanAttributes.USER_ID
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_NAME = SpanAttributes.GRAPH_NODE_NAME
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID

# message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
