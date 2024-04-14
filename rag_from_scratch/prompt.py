from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


# Question Rephrasing
multiquery_template = """You are an AI research assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

ragfusion_template = """You are an expert research assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question}

Make sure each output is a question
Output (4 queries):"""

decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

recursive_decomposition_template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

# Summarization
basic_template = """Answer the following question based on this context:

{context}

Question: {question}
"""

q_and_a_template = """"

Answer the following question using provided question and answer pairs.  

{context}

Format the response into a high level summary followed by relevant details and terms.
When possible cite your sources
Provide an examples if appropriate, but always build the response on the same example.

Question:  {question}
"""


research_template = """
Answer the question using the following context.  
Format the response into a high level summary, 
Provide an examples if appropriate, but always try to build on the same example.
Always include relevant details and terms and when possible cite your sources:

{context}

Question: {question}
"""

research_to_personality_template = """
You love teaching me!  Provide a very detailed response to the following question using the results of your research provided below.

{research}

Question: {question}
"""


##### FEW SHOT PROMPTING
# Few Shot Examples
stepback_question_fewshot_examples_dict = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]

stepback_question_fewshot_examples_template = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    ),
    examples=stepback_question_fewshot_examples_dict,
)


generate_stepback_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        stepback_question_fewshot_examples_template,
        ("user", "{question}"),
    ]
)

stepback_template = """Answer the question using the following context.  
Format the response into a high level summary, always include relevant details and terms,
But ignore information irrelevant to the question.
When possible cite your sources.
Provide an examples if appropriate, but always try to build on the same example.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
