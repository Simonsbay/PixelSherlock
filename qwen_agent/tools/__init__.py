# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .amap_weather import AmapWeather
from .base import TOOL_REGISTRY, BaseTool
from .code_interpreter import CodeInterpreter
from .doc_parser import DocParser
from .extract_doc_vocabulary import ExtractDocVocabulary
from .image_analyse import ImageAnalyse
from .image_gen import ImageGen
from .image_text_web_search import ImageTextWebSearch
from .python_executor import PythonExecutor
from .retrieval import Retrieval
from .image_zoom_in_qwen3vl import ImageZoomInToolQwen3VL
from .image_search import ImageSearch
from .search_tools import FrontPageSearch, HybridSearch, KeywordSearch, VectorSearch
from .simple_doc_parser import SimpleDocParser
from .storage import Storage
from .web_extractor import WebExtractor
from .mcp_manager import MCPManager


# Delayed imports to avoid circular import
# These modules import qwen_agent, so we can't import them at module level
_image_multi_crop = None
_image_recognize = None
_web_search = None


def __getattr__(name):
    """Lazy import to avoid circular import issues."""
    global _image_multi_crop, _image_recognize, _web_search

    if name == 'ImageMultiCrop':
        if _image_multi_crop is None:
            from image_multi_crop import ImageMultiCrop
            _image_multi_crop = ImageMultiCrop
        return _image_multi_crop

    if name == 'ImageRecognize':
        if _image_recognize is None:
            from image_recognize2 import ImageRecognize
            _image_recognize = ImageRecognize
        return _image_recognize

    if name == 'WebSearch':
        if _web_search is None:
            from text_search import WebSearch
            _web_search = WebSearch
        return _web_search

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'BaseTool',
    'CodeInterpreter',
    'ImageGen',
    'AmapWeather',
    'TOOL_REGISTRY',
    'DocParser',
    'KeywordSearch',
    'Storage',
    'Retrieval',
    'ImageSearch',
    'WebExtractor',
    'SimpleDocParser',
    'VectorSearch',
    'HybridSearch',
    'FrontPageSearch',
    'ExtractDocVocabulary',
    'PythonExecutor',
    'MCPManager',
    'WebSearch',
    'ImageMultiCrop',
    'ImageRecognize',
]


