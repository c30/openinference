# ruff: noqa: E501


class MetricAttributes:
    """
    OpenInference Metrics Semantic Conventions.
    
    These metrics provide aggregatable data about LLM operations,
    costs, and performance characteristics.
    """
    
    # Token usage metrics
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    """
    Histogram of prompt token counts for LLM operations.
    Unit: tokens
    """
    
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    """
    Histogram of completion token counts for LLM operations.
    Unit: tokens
    """
    
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
    """
    Histogram of total token counts for LLM operations.
    Unit: tokens
    """
    
    # Cost metrics
    LLM_COST_PROMPT = "llm.cost.prompt"
    """
    Histogram of prompt costs for LLM operations.
    Unit: USD
    """
    
    LLM_COST_COMPLETION = "llm.cost.completion"
    """
    Histogram of completion costs for LLM operations.
    Unit: USD
    """
    
    LLM_COST_TOTAL = "llm.cost.total"
    """
    Histogram of total costs for LLM operations.
    Unit: USD
    """
    
    # Performance metrics
    LLM_OPERATION_DURATION = "llm.operation.duration"
    """
    Histogram of LLM operation durations.
    Unit: milliseconds
    """
    
    LLM_TIME_TO_FIRST_TOKEN = "llm.time_to_first_token"
    """
    Histogram of time to first token for streaming LLM operations.
    Unit: milliseconds
    """
    
    LLM_TIME_BETWEEN_TOKENS = "llm.time_between_tokens"
    """
    Histogram of average time between consecutive tokens during streaming.
    Unit: milliseconds
    """
    
    LLM_TIME_PER_OUTPUT_TOKEN = "llm.time_per_output_token"
    """
    Histogram of average time per output token generated.
    Unit: milliseconds
    """
    
    # Request counters
    LLM_REQUESTS_TOTAL = "llm.requests.total"
    """
    Counter of total LLM requests.
    Unit: requests
    """
    
    LLM_REQUESTS_FAILED = "llm.requests.failed"
    """
    Counter of failed LLM requests.
    Unit: requests
    """


class MetricUnits:
    """
    Standard units for OpenInference metrics.
    """
    TOKENS = "tokens"
    USD = "USD"
    MILLISECONDS = "ms"
    REQUESTS = "requests"


class MetricTypes:
    """
    OpenTelemetry metric types used by OpenInference.
    """
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"