"""
Utilities for creating and managing OpenTelemetry metrics in OpenInference instrumentation.
"""

from typing import Optional, Dict, Any, Union
from opentelemetry.metrics import get_meter, Meter, Histogram, Counter
from openinference.semconv.metric import MetricAttributes, MetricUnits


class MetricsHelper:
    """
    Helper class for creating and recording OpenInference metrics.
    
    This class provides a centralized way to create and record metrics
    that follow OpenInference semantic conventions.
    """
    
    def __init__(self, meter_name: str, version: Optional[str] = None):
        """
        Initialize the metrics helper.
        
        Args:
            meter_name: Name of the meter (usually the instrumentation package name)
            version: Version of the instrumentation package
        """
        self._meter: Meter = get_meter(meter_name, version)
        self._metrics: Dict[str, Union[Histogram, Counter]] = {}
    
    def _get_or_create_histogram(self, name: str, description: str, unit: str) -> Histogram:
        """Get or create a histogram metric."""
        if name not in self._metrics:
            self._metrics[name] = self._meter.create_histogram(
                name=name,
                description=description,
                unit=unit
            )
        return self._metrics[name]
    
    def _get_or_create_counter(self, name: str, description: str, unit: str) -> Counter:
        """Get or create a counter metric."""
        if name not in self._metrics:
            self._metrics[name] = self._meter.create_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._metrics[name]
    
    def record_token_count_prompt(self, count: int, attributes: Optional[Dict[str, Any]] = None):
        """Record prompt token count metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_TOKEN_COUNT_PROMPT,
            "Histogram of prompt token counts for LLM operations",
            MetricUnits.TOKENS
        )
        histogram.record(count, attributes or {})
    
    def record_token_count_completion(self, count: int, attributes: Optional[Dict[str, Any]] = None):
        """Record completion token count metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_TOKEN_COUNT_COMPLETION,
            "Histogram of completion token counts for LLM operations",
            MetricUnits.TOKENS
        )
        histogram.record(count, attributes or {})
    
    def record_token_count_total(self, count: int, attributes: Optional[Dict[str, Any]] = None):
        """Record total token count metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_TOKEN_COUNT_TOTAL,
            "Histogram of total token counts for LLM operations",
            MetricUnits.TOKENS
        )
        histogram.record(count, attributes or {})
    
    def record_cost_prompt(self, cost: float, attributes: Optional[Dict[str, Any]] = None):
        """Record prompt cost metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_COST_PROMPT,
            "Histogram of prompt costs for LLM operations",
            MetricUnits.USD
        )
        histogram.record(cost, attributes or {})
    
    def record_cost_completion(self, cost: float, attributes: Optional[Dict[str, Any]] = None):
        """Record completion cost metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_COST_COMPLETION,
            "Histogram of completion costs for LLM operations",
            MetricUnits.USD
        )
        histogram.record(cost, attributes or {})
    
    def record_cost_total(self, cost: float, attributes: Optional[Dict[str, Any]] = None):
        """Record total cost metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_COST_TOTAL,
            "Histogram of total costs for LLM operations",
            MetricUnits.USD
        )
        histogram.record(cost, attributes or {})
    
    def record_operation_duration(self, duration_ms: float, attributes: Optional[Dict[str, Any]] = None):
        """Record operation duration metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_OPERATION_DURATION,
            "Histogram of LLM operation durations",
            MetricUnits.MILLISECONDS
        )
        histogram.record(duration_ms, attributes or {})
    
    def record_time_to_first_token(self, time_ms: float, attributes: Optional[Dict[str, Any]] = None):
        """Record time to first token metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_TIME_TO_FIRST_TOKEN,
            "Histogram of time to first token for streaming LLM operations",
            MetricUnits.MILLISECONDS
        )
        histogram.record(time_ms, attributes or {})
    
    def record_time_between_tokens(self, time_ms: float, attributes: Optional[Dict[str, Any]] = None):
        """Record time between tokens metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_TIME_BETWEEN_TOKENS,
            "Histogram of average time between consecutive tokens during streaming",
            MetricUnits.MILLISECONDS
        )
        histogram.record(time_ms, attributes or {})
    
    def record_time_per_output_token(self, time_ms: float, attributes: Optional[Dict[str, Any]] = None):
        """Record time per output token metric."""
        histogram = self._get_or_create_histogram(
            MetricAttributes.LLM_TIME_PER_OUTPUT_TOKEN,
            "Histogram of average time per output token generated",
            MetricUnits.MILLISECONDS
        )
        histogram.record(time_ms, attributes or {})
    
    def increment_requests_total(self, attributes: Optional[Dict[str, Any]] = None):
        """Increment total requests counter."""
        counter = self._get_or_create_counter(
            MetricAttributes.LLM_REQUESTS_TOTAL,
            "Counter of total LLM requests",
            MetricUnits.REQUESTS
        )
        counter.add(1, attributes or {})
    
    def increment_requests_failed(self, attributes: Optional[Dict[str, Any]] = None):
        """Increment failed requests counter."""
        counter = self._get_or_create_counter(
            MetricAttributes.LLM_REQUESTS_FAILED,
            "Counter of failed LLM requests",
            MetricUnits.REQUESTS
        )
        counter.add(1, attributes or {})