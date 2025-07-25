"""
DeepSeek API Client

OpenAI-compatible client for DeepSeek LLM API.
Handles authentication, retries, and error handling.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Iterator
from dataclasses import dataclass
import backoff
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="/Users/carrickcheah/Project/ppo/app_2/.env")

logger = logging.getLogger(__name__)


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek client."""
    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    max_retries: int = 3
    timeout: int = 300  # 5 minutes for large scheduling tasks
    temperature: float = 0.1  # Low temperature for consistent results
    max_tokens: int = 4096


class DeepSeekClient:
    """
    DeepSeek API client with OpenAI compatibility.
    """
    
    def __init__(self, config: Optional[DeepSeekConfig] = None):
        """
        Initialize DeepSeek client.
        
        Args:
            config: Optional configuration. If not provided, loads from environment.
        """
        if config is None:
            # Load from environment
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment")
            
            config = DeepSeekConfig(
                api_key=api_key,
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            )
        
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        # Token tracking for cost estimation
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send completion request to DeepSeek.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Whether to stream the response
            
        Returns:
            Response dictionary with completion and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=stream
            )
            
            if stream:
                return self._handle_stream(response)
            else:
                return self._handle_response(response)
                
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise
            
    def _handle_response(self, response) -> Dict[str, Any]:
        """Handle non-streaming response."""
        # Track tokens
        if hasattr(response, 'usage'):
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
        
        return {
            'content': response.choices[0].message.content,
            'finish_reason': response.choices[0].finish_reason,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                'completion_tokens': response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                'total_tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0
            },
            'model': response.model,
            'id': response.id
        }
        
    def _handle_stream(self, response_stream) -> Iterator[str]:
        """Handle streaming response."""
        for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    def schedule_jobs(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Specialized method for job scheduling tasks.
        
        Args:
            system_prompt: System message with instructions
            user_prompt: User message with job/machine data
            temperature: Temperature for generation
            
        Returns:
            Scheduling response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        start_time = time.time()
        response = self.complete(messages, temperature=temperature)
        elapsed_time = time.time() - start_time
        
        response['elapsed_time'] = elapsed_time
        logger.info(f"Scheduling completed in {elapsed_time:.2f}s")
        
        return response
        
    def get_cost_estimate(self) -> Dict[str, float]:
        """
        Estimate API costs based on token usage.
        
        DeepSeek pricing (approximate):
        - Input: $0.14 per 1M tokens
        - Output: $0.28 per 1M tokens
        """
        input_cost = (self.total_input_tokens / 1_000_000) * 0.14
        output_cost = (self.total_output_tokens / 1_000_000) * 0.28
        
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }
        
    def reset_token_counts(self):
        """Reset token tracking."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0


def test_connection():
    """Test DeepSeek connection with a simple prompt."""
    try:
        client = DeepSeekClient()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, scheduling world!' to confirm connection."}
        ]
        
        response = client.complete(messages)
        print(f"Connection successful!")
        print(f"Response: {response['content']}")
        print(f"Tokens used: {response['usage']['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the connection
    test_connection()