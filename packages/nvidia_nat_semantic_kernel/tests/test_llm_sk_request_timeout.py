# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Semantic Kernel OpenAI and Azure OpenAI clients honor ``request_timeout`` from the LLM config."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.semantic_kernel.llm import azure_openai_semantic_kernel
from nat.plugins.semantic_kernel.llm import openai_semantic_kernel


@pytest.mark.usefixtures("set_test_api_keys")
class TestRequestTimeout:

    @pytest.mark.parametrize("request_timeout, expected", [(12.5, 12.5), (None, None)])
    @patch("openai.AsyncOpenAI")
    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    async def test_openai(self,
                          _mock_sk,
                          mock_client,
                          mock_builder,
                          mock_httpx_async_client,
                          request_timeout,
                          expected):
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        config = OpenAIModelConfig(model_name="gpt-4o", request_timeout=request_timeout)

        async with openai_semantic_kernel(config, mock_builder):
            pass

        assert mock_client.call_args.kwargs.get("timeout") == expected
        assert ("timeout" in mock_client.call_args.kwargs) is (expected is not None)

    @pytest.mark.parametrize("request_timeout, expected", [(12.5, 12.5), (None, None)])
    @patch("openai.AsyncAzureOpenAI")
    @patch("semantic_kernel.connectors.ai.open_ai.AzureChatCompletion")
    async def test_azure_openai(self,
                                _mock_sk,
                                mock_client,
                                mock_builder,
                                mock_httpx_async_client,
                                request_timeout,
                                expected):
        mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        config = AzureOpenAIModelConfig(azure_deployment="gpt-4o",
                                        azure_endpoint="https://example.invalid",
                                        request_timeout=request_timeout)

        async with azure_openai_semantic_kernel(config, mock_builder):
            pass

        assert mock_client.call_args.kwargs.get("timeout") == expected
        assert ("timeout" in mock_client.call_args.kwargs) is (expected is not None)
