from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def main():
    print("Hello from langchain-course!")

    information = """Earth is the third planet from the Sun and the only astronomical object known to harbor life.
This is enabled by Earth being an ocean world..."""  # shortened for clarity

    summary_template = """
Given the information {information} about a planet, you are a very professional professor. You need to:
1. Analyze the given data carefully.
2. Write a short summary.
3. Provide 2 very interesting points about it.
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],  # <-- list of variable names (not the content)
        template=summary_template
    )

    # NOTE: model name must match ollama list entry exactly (you have deepseek-r1:8b)
    llm = ChatOllama(model="deepseek-r1:8b", temperature=0)

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)  # adjust if you want specific attributes

if __name__ == "__main__":
    main()
