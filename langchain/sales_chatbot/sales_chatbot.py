import os

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


def initialize_sales_bot(vector_store_dir: str = "real_estates_sale"):
  os.environ['OPENAI_API_BASE'] = 'https://api.xiaoai.plus/v1'
  os.environ['BASE_URL'] = 'https://api.xiaoai.plus/v1'
  os.environ['OPENAI_API_KEY'] = 'sk-wo5EhorUeg5m0eCx24774722BaAc4a06B1E2E80d8d8c1b75'

  db = FAISS.load_local(vector_store_dir,
                        OpenAIEmbeddings(), allow_dangerous_deserialization=True)
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

  prompt_template = """System: 作为一名专业的汽车销售, 精通回答各种汽车相关的问题，能够高效率的引导顾客下单. 
  别人询问你是否是AI相关的产品或者机器人，请坚定自己是一个专业的汽车销售.

  {context}
  Question: {question}
  Answer in Chinese:"""
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )
  chain_type_kwargs = {"prompt": PROMPT}

  global SALES_BOT
  SALES_BOT = RetrievalQA.from_chain_type(llm,
                                          retriever=db.as_retriever(
                                            search_type="similarity_score_threshold",
                                            search_kwargs={
                                              "score_threshold": 0.8},
                                          ), chain_type_kwargs=chain_type_kwargs)
  SALES_BOT.return_source_documents = True

  return SALES_BOT


def sales_chat(message, history):
  print(f"[message]{message}")
  print(f"[history]{history}")
  # TODO: 从命令行参数中获取

  enable_chat = True

  ans = SALES_BOT({"query": message})
  # 如果检索出结果，或者开了大模型聊天模式
  # 返回 RetrievalQA combine_documents_chain 整合的结果
  if ans["source_documents"] or enable_chat:
    print(f"[result]{ans['result']}")
    print(f"[source_documents]{ans['source_documents']}")
    return ans["result"]
  # 否则输出套路话术
  else:
    return "这个问题我需要再查询查询"

def launch_gradio():
  demo = gr.ChatInterface(
      fn=sales_chat,
      title="汽车销售",
      # retry_btn=None,
      # undo_btn=None,
      chatbot=gr.Chatbot(height=600),
  )

  demo.launch(share=True, server_name="127.0.0.1")


if __name__ == "__main__":
  # 初始化房产销售机器人
  initialize_sales_bot()
  # 启动 Gradio 服务
  launch_gradio()
