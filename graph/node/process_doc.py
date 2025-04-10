from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """
    print("---CHECK RELEVANCE---")

    class grade(BaseModel):
        """Binary score for relevance check"""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    llm_with_tool = model.with_structured_output(grade)

    # prompt = PromptTemplate()
