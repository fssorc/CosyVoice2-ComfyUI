from .nodes import  CosyVoice2Node,TextNode,CosyVoice2ModelDownload

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
   # "LoadSRT":LoadSRT,
    "TextNode": TextNode,
    "CosyVoice2Node": CosyVoice2Node,
    "CosyVoice2ModelDownload": CosyVoice2ModelDownload,
}
