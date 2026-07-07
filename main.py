from parse_util import *

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print, multimodal_typewriter_print
from qwen_agent import Agent
from qwen_agent.llm.schema import ContentItem, Message
import copy
import os
from dashscope import MultiModalConversation
import dashscope 
import pandas as pd
from tqdm import tqdm
import csv
import time  # 新增time模块
import onnxruntime
import onnx
import cv2
import numpy as np
from qwen_agent.gui import WebUI
from prompts import *

from parse_util import global_crawler
import text_search
import image_multi_crop
import image_recognize2
from reformat_out_agent import FormatOutputAgent

os.environ['GRADIO_TEMP_DIR'] = os.getenv('GRADIO_TEMP_DIR', './gradio_workspace')
llm_cfg = {
    'model_type': 'qwenvl_dashscope',
    #"model": "qwen3-vl-235b-a22b-instruct",
    'model': 'qwen3-vl-30b-a3b-instruct',
    'api_key': os.getenv('DASHSCOPE_API_KEY', ''),
    'generate_cfg': {
        "top_p": 0.00001,
        "top_k": 20,
        "temperature": 0.00001,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5
    }
}

def image_intent_recognition(local_path):
    """
    图像意图识别工具
    返回结果：
    1 - 动物/植物/食物
    2 - 建筑
    3 - 通用（其他类别）
    """
    image_path = f"{local_path}"
    # 系统提示词，明确指定返回格式和分类规则
    system_prompt = INTEND_PROMPT
    messages = [
        {
            "role": "system",
            "content": [{"text": system_prompt}]
        },
        {
            'role': 'user',
            'content': [
                {'image': image_path},
                {'text': '请分析本次用户提问的意图类别'}
            ]
        }
    ]

    try:
        response = MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY', ''),  # 请填入您的API Key
            model='qwen3-vl-30b-a3b-instruct',
            messages=messages)
        
        # 提取预测结果
        prediction = ""
        try:
            prediction = response["output"]["choices"][0]["message"].content[0]["text"].strip()
            
            # 确保只返回1,2,3中的某一个
            if prediction in ['1', '2', '3']:
                return int(prediction)
            else:
                return 3
                
        except (KeyError, IndexError, TypeError) as e:
            print(f"解析响应失败, 错误: {e}")
            return 3  # 解析失败时返回默认类别
            
    except Exception as e:
        print(f"API调用失败: {e}")
        return 3 # API调用失败时返回默认类别
    

class MainAgent(Agent):
    def __init__(self,
                 llm = None):
        super().__init__(llm=llm)
        # Nest one vl assistant for image understanding
        self.intent_agent = Assistant(llm=llm_cfg)
        # 改写建筑物工具调用
        self.person_agent = FormatOutputAgent(
                llm=llm_cfg,
                function_list=[
                'image_multi_crop',
                'image_recognize',
                'web_text_search',],
                system_message=PERSON_PROMPT,)
        self.location_agnet = FormatOutputAgent(
                llm=llm_cfg,
                function_list=[
                'image_multi_crop',
                'image_recognize',
                'web_text_search',],
                system_message=LOCATION_PROMPT,)
        self.general_agent = FormatOutputAgent(
                llm=llm_cfg,
                function_list= [
                'image_multi_crop',
                'image_recognize',
                'web_text_search',],
                system_message=GENERAL_PROMPT,)
        
        
    def _run(self, messages, lang: str = 'zh', **kwargs):

        """Define the workflow"""
        assert isinstance(messages[-1]['content'], list)
        assert any([item.image for item in messages[-1]['content']]), 'This agent requires input of images'
        #多轮对话处理逻辑：在一轮对话得到结果时，如果用户第二轮提问包含图片
        if len(messages)!=1:
            messages = [messages[-1]]
        new_messages = copy.deepcopy(messages)
        img_path = new_messages[0]["content"][1]["image"]
        img_path = img_path.replace("file://","")
        print(img_path)
        intent = image_intent_recognition(img_path)
        #1.首先进行意图识别
        #意图1：动物/植物/食物
        #意图2：建筑
        #意图3：通用能力调用
        print("!!!!!!!!!!!",intent)
        if intent == 1:
            new_messages = copy.deepcopy(messages)
            response = []
            for rsp in self.person_agent.run(new_messages):
                yield response + rsp
            response.extend(rsp)
            new_messages.extend(rsp)
        elif intent == 2:
            new_messages = copy.deepcopy(messages)
            response = []
            for rsp in self.location_agnet.run(new_messages):
                yield response + rsp
            response.extend(rsp)
            new_messages.extend(rsp)
        else:
            new_messages = copy.deepcopy(messages)
            response = []
            for rsp in self.general_agent.run(new_messages):
                yield response + rsp
            response.extend(rsp)
            new_messages.extend(rsp)
            
bot = MainAgent()
messages = []
messages.append(Message('user', [ContentItem(text="这是哪?")]))
messages[-1]['content'].append(ContentItem(image="/gemini-3/space/data/wangcy/Vision-DeepResearch-main/agent_v3/test_data/location/test5.jpg"))
response_plain_text = ''
for ret_messages in bot.run(messages):
    response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
    
# def app_gui():
#     bot = MainAgent()
#     WebUI(bot).run()

# if __name__ == '__main__':
#     app_gui()