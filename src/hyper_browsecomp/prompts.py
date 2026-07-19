QUERY_TEMPLATE = """
{question}

Your response must use this exact format:

Explanation: {{brief evidence-based reasoning}}
Exact Answer: {{the shortest correct answer}}
Confidence: {{0-100%}}
""".strip()


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct based only on the precise [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Return your judgement in exactly this format:

extracted_final_answer: The final exact answer extracted from [response], or None if no exact final answer exists.
reasoning: Explain whether the extracted answer matches [correct_answer]. Focus only on meaningful differences.
correct: yes or no
confidence: The confidence score between 0 and 100 extracted from [response]. Use 100 if no confidence score is available.
""".strip()


WEB_ONLY_AGENT_PROMPT = """
You are a web research agent. You must answer using tools, not memory.

If you need information from the internet, use web_search first to find candidate sources, then use web_fetch to retrieve the relevant page content. Prefer official and primary sources whenever possible.

Do not use bash or python for internet retrieval. Use bash or python only when the web tools are insufficient and you genuinely need computation, parsing, or transformation that cannot be done from the tool outputs alone.

When the answer is known, respond in exactly this format:

Explanation: {brief evidence-based reasoning}
Exact Answer: {the shortest correct answer}
Confidence: {0-100%}
""".strip()
