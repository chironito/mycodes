#!/usr/bin/env python
# coding: utf-8

# In[2]:


import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio
import tqdm
import random
from typing import List, Dict, Tuple, Union


# In[4]:


async def _single_api_call(
    session: aiohttp.ClientSession,
    endpoint: str,
    method: str='post',
    headers: Dict[str, str]=None,
    payload: Dict[Union[int, str], Union[str, dict, list, int, float]]=None,
    params: Dict[str, Union[int, str]]=None,
    attempt: int=0,
    max_retries: int=3,
    retry_delay_sec: int=1,
    server_breather: bool=True
):
    # print(endpoint, method)
    try:
        kwargs = {keyword: value for keyword, value in zip(("params", "json", "headers"), (params, payload, headers)) if value}
        async with getattr(session, method)(endpoint, **kwargs) as response:
            # response.raise_for_status()
            if response.status == 405:
                attempt = max_retries
            if response.status not in {200, 201, 202, 203, 204, 404}:
                raise Exception(f"Failed with status: {response.status}")
            try:
                result = await response.json()
            except aiohttp.ContentTypeError:
                result = await response.text()
            if server_breather:
                await asyncio.sleep(random.uniform(0.2, 0.6))
            return result
    except Exception as exp:
        if attempt >= max_retries:
            raise RuntimeError(f"Max retries ({max_retries}) reached: {exp}")
        await asyncio.sleep(retry_delay_sec)
        return await _single_api_call(session, endpoint, method, headers, payload, params, attempt+1, max_retries, retry_delay_sec, server_breather)

async def batched_requests(
    endpoint: Union[str, List[str]],
    method: Union[str, List[str]]='post',
    payload: Union[List[Dict[Union[int, str], Union[str, dict, list, int, float]]], Dict[Union[int, str], Union[str, dict, list, int, float]]]=None,
    headers: Union[Dict[str, str], List[Dict[str, str]]]=None,
    params: Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]]=None,
    attempt: int=0,
    max_retries: int=3,
    retry_delay_sec: int=1,
    server_breather: bool=True,
    n_calls: int=1,
    timeout: int=60
):
    timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = []
        if headers is None:
            headers = [{}]
        if payload is None:
            payload = [{}]
        if params is None:
            params = [{}]
        if not isinstance(headers, list):
            headers = [headers]
        if not isinstance(payload, list):
            payload = [payload]
        if not isinstance(params, list):
            params = [params]
        if not isinstance(endpoint, list):
            endpoint = [endpoint]
        if not isinstance(method, list):
            method = [method]
        if n_calls > 1000:
            print(f"Max. n_calls is 1000.")
            n_calls = 1000
        if len(headers) > 1000:
            headers[1000:] = []
        if len(payload) > 1000:
            payload[1000:] = []
        if len(params) > 1000:
            params[1000:] = []
        if len(endpoint) > 1000:
            endpoint[1000:] = []
        if len(method) > 1000:
            method[1000:] = []
        if n_calls != 1:
            headers = [headers[0]]*n_calls
            payload = [payload[0]]*n_calls
            params = [params[0]]*n_calls
            endpoint = [endpoint[0]]*n_calls
            method = [method[0]]*n_calls
        if max_retries > 5:
            print(f"Max. retries is 5")
            max_retries = 5
        progress_bar = tqdm.tqdm(total=len(endpoint), desc="Generating responses")
        async def wrapped_call(endpoint, method, headers=None, payload=None, params=None, attempt=None, max_retries=None, retry_delay_sec=None, server_breather=True):
            # args = [endpoint, method, *filter(bool, (headers, payload, params, attempt, max_retries, retry_delay_sec))]
            kwargs = {"endpoint": endpoint, "method": method}
            for argname, arg in zip(("headers", "payload", "params", "attempt", "max_retries", "retry_delay_sec", "server_breather"), (headers, payload, params, attempt, max_retries, retry_delay_sec, server_breather)):
                kwargs[argname] = arg
            result = await _single_api_call(session, **kwargs)
            progress_bar.update(1)
            return result
        zip_args = [endpoint, method, *(arg for arg in (headers, payload, params) if any(arg))]
        # print(*zip_args)
        tasks = [wrapped_call(*args, attempt=attempt, max_retries=max_retries, retry_delay_sec=retry_delay_sec, server_breather=server_breather) for args in zip(*zip_args)]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            progress_bar.close()
        return results


# In[ ]:
