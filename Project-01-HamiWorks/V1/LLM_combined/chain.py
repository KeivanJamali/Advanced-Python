from langchain.chat_models import init_chat_model
from LLM_combined.prompts import *
from LLM_combined.schema import FinalOutput


llm = init_chat_model(model="openai:gpt-5-nano")
chain = prompt_to_grade | llm.with_structured_output(FinalOutput)

# res = chain.invoke("سلام. خوبی؟")
# print(res.tool_calls[0]["args"]["query"])
# print(res.tool_calls[0]["args"]["language"])
# print()

# res = chain.invoke({"query": "Hi. How are you?", "language": "Persian"})
# print(res.content)