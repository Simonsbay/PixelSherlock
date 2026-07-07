import copy
import json
from typing import Dict, Iterator, List, Literal, Optional, Union

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, ContentItem, Message, FUNCTION
from qwen_agent.log import logger
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool

def extra_image_res(response):
    files = []
    if isinstance(response.content, list):
        for item in response.content:
            if item.image and item.image not in files:
                files.append(item)
    new_msg = Message(role=FUNCTION,
            name=response.name,
            content=files,
            extra=response.extra)
    return new_msg
    
def reformat_search_result():
    pass

class FormatOutputAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 rag_cfg: Optional[Dict] = None):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         rag_cfg=rag_cfg)

    def _run(self,
             messages: List[Message],
             lang: Literal['en', 'zh'] = 'en',
             **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        num_llm_calls_available = 9
        response = []
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
            output_stream = self._call_llm(messages=messages,
                                           functions=[func.function for func in self.function_map.values()],
                                           extra_generate_cfg=extra_generate_cfg)
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output
            if output:
                response.extend(output)
                messages.extend(output)
                used_any_tool = False
                for out in output:
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool :
                        if tool_name =="image_recognize":
                            tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                            tool_result = json.loads(tool_result)
                            summ_res = []
                            urls = []
                            print("@@@",tool_result)
                            summ_res = tool_result["summ_res"]
                            urls = tool_result["urls"]
                            fn_msg = Message(role=FUNCTION,
                                            name=tool_name,
                                            content=json.dumps(summ_res, ensure_ascii=False, indent=4),
                                            extra={'function_id': out.extra.get('function_id', '1')})
                            messages.append(fn_msg)
                            show_msg= Message(role=FUNCTION,
                                            name=tool_name,
                                            content=json.dumps(urls, ensure_ascii=False, indent=4),
                                            extra={'function_id': out.extra.get('function_id', '1')})
                            response.append(show_msg)
                        else:
                            tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                            fn_msg = Message(role=FUNCTION,
                                            name=tool_name,
                                            content=tool_result,
                                            extra={'function_id': out.extra.get('function_id', '1')})
                            messages.append(fn_msg)
                            if tool_name =="image_multi_crop":
                                response.append(extra_image_res(fn_msg))
                            else:
                                response.append(fn_msg)
                                
                        yield response
                        used_any_tool = True
                if not used_any_tool:
                    break
        yield response