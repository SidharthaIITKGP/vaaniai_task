"""Loopback-only adapter reusing the real Groq protocol implementation without a key."""
from __future__ import annotations

import asyncio
import contextvars
import time
from urllib.parse import urlparse

from ..config import ProviderConfig
from .groq_provider import GroqProvider
from .base import ProviderError, RateLimitError


SIM_REQUEST = contextvars.ContextVar('simulation_request',default='unassigned')


class StandInProvider(GroqProvider):
    def __init__(self, endpoint, client, *, time_scale=1, retries=1):
        parsed = urlparse(endpoint)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
            raise ValueError('Stand-in endpoint must be loopback HTTP; remote inference is forbidden')
        self.endpoint, self._client = endpoint, client
        self.api_key = 'local-simulation-no-credential'
        self.config = ProviderConfig()
        self.scheduler = None  # The stand-in enforces actual rolling per-call quotas.
        self.time_scale, self.max_retries = time_scale,retries
        self.attempts, self.retry_waits = [],[]

    async def _post(self,payload,reservation):
        schema = (payload.get('response_format') or {}).get('json_schema') or {}
        stage = 'routing' if schema.get('name')=='intent_classification' else 'generation'
        request_id = SIM_REQUEST.get()
        payload = dict(payload,simulation_request_id=request_id,simulation_stage=stage)
        for attempt in range(self.max_retries+1):
            started = time.perf_counter()
            try:
                result = await super()._post(payload,reservation)
                self.attempts.append({'request_id':request_id,'stage':stage,'retry':attempt,
                    'status':200,'wall_ms':(time.perf_counter()-started)*1000})
                return result
            except ProviderError as error:
                self.attempts.append({'request_id':request_id,'stage':stage,'retry':attempt,
                    'status':error.status_code,'wall_ms':(time.perf_counter()-started)*1000})
                daily = error.rate_limit_type in {'provider_rpd','provider_tpd'}
                wait = float(error.retry_after) if error.retry_after is not None else .25*(2**attempt)
                retryable = error.status_code in {429,500,503} and not daily and wait<=2
                if attempt>=self.max_retries or not retryable:
                    raise
                self.retry_waits.append({'request_id':request_id,'virtual_ms':wait*1000})
                await asyncio.sleep(wait*self.time_scale)
