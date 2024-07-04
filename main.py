from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate# type: ignore
from langchain.chat_models import ChatOpenAI# type: ignore
from langchain.chains import LLMChain# type: ignore
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

chat = ChatOpenAI()

memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory('messages.json'),
    memory_key='messages', 
    return_messages=True,
    llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=['content', 'messages'],
    messages=[
        MessagesPlaceholder(variable_name='messages'),
        HumanMessagePromptTemplate.from_template('{content}')
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

while True:
    content = input('>> ')
    result = chain({'content': content})

    print(result['text'])
