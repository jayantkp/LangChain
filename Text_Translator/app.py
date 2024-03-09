import os
import gradio as gr
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI


# with open('openai_api_key.txt') as f:
#     api_key = f.read()

# os.environ['OPENAI_API_KEY'] = api_key

chat = ChatOpenAI()

# Define the Pydantic Model
class TextTranslator(BaseModel):
    output: str = Field(description="Python string containing the output text translated in the desired language")
    
output_parser = PydanticOutputParser(pydantic_object=TextTranslator)
format_instructions = output_parser.get_format_instructions()

def text_translator(input_text : str, language : str) -> str:
    human_template = """Enter the text that you want to translate: 
                {input_text}, and enter the language that you want it to translate to {language}. {format_instructions}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    
    prompt = chat_prompt.format_prompt(input_text = input_text, language = language, format_instructions = format_instructions)
    
    messages = prompt.to_messages()
    
    response = chat(messages = messages)
    
    output = output_parser.parse(response.content)
    
    output_text = output.output
    
    return output_text

# Interface
with gr.Blocks() as demo:
    gr.HTML("<h1 align = 'center'> Text Translator </h1>")
    gr.HTML("<h4 align = 'center'> Translate to any language </h4>")
    
    inputs = [gr.Textbox(label = "Enter the text that you want to translate"), gr.Textbox(label = "Enter the language that you want it to translate to", placeholder = "Example : Hindi,French,Bengali,etc")]
    generate_btn = gr.Button(value = 'Generate')
    outputs  = [gr.Textbox(label = "Translated text")]
    generate_btn.click(fn = text_translator, inputs= inputs, outputs = outputs)
    
if __name__ == '__main__':
    demo.launch()  