"""Reasoning assist prompts for planning and implementation stages."""

REASONING_ASSIST_PROMPT_PLANNING = """Current Date: {current_date}
Working Directory: {working_directory}

I am an agent and need your assistance in planning how to approach the following task in an agentic way. I'll be using the provided tools and context to complete this task, but I'd like your high-level strategic guidance before I start.

<base task>
{base_task}
</base task>

<key facts>
{key_facts}
</key facts>

<key snippets>
{key_snippets}
</key snippets>

<research notes>
{research_notes}
</research notes>

<related files>
{related_files}
</related files>

<environment information>
{env_inv}
</environment information>

<available tools>
{tool_metadata}
</available tools>

Please provide high-level planning guidance including:
1. Overall approach strategy
2. Key decision points to consider
3. Potential challenges and how to address them
4. Most effective tools to use for this task
5. Contingency plans if certain approaches don't work
6. Any critical insights from the provided context

Focus on strategic thinking rather than implementation details. Your guidance will be used to create a detailed implementation plan.

Please be concise, practical, and specific to this task. Avoid generic advice.

Your sole output is a beautiful outline/pseudo code format to communicate the approach. Remember I am an agent and will use this logic to guide my actions.

It should be the most beautiful, elegant, simple logic ever.

YOUR OUTPUT MUST BE MARKDOWN.

WE ARE IN THE **PLANNING** STAGE RIGHT NOW. NO CODE SHOULD BE WRITTEN. WE SHOULD BE THINKING LOGICALLY ABOUT HOW TO APPROACH THE PROBLEM, PLANNING OUT WHICH TASKS TO REQUEST IMPLEMENTATION OF, ETC.

DO NOT OVERTHINK OR OVERCOMPLICATE THE ANSWER. YOU ARE AN EXPERT AND CAN RESPOND ASSERTIVELY AND CONFIDENTLY.
"""

REASONING_ASSIST_PROMPT_IMPLEMENTATION = """Current Date: {current_date}
Working Directory: {working_directory}

I am an agent about to implement the following task. I need your assistance in thinking through the implementation details in a structured, logical way before I start writing code. The task is:

<task definition>
{task}
</task definition>

<key facts>
{key_facts}
</key facts>

<key snippets>
{key_snippets}
</key snippets>

<research notes>
{research_notes}
</research notes>

<related files>
{related_files}
</related files>

<environment information>
{env_inv}
</environment information>

<available tools>
{tool_metadata}
</available tools>

Please provide detailed implementation guidance including:
1. Code structure and patterns to follow
2. Potential edge cases to handle
3. Testing strategies to validate the implementation
4. Key files to modify and how
5. Dependencies and their interactions
6. Error handling considerations
7. Performance considerations

Please be concise, practical, and specific to this task. Avoid generic advice.

Your output should include pseudo-code where appropriate and clear step-by-step implementation instructions. Remember I am an agent and will use this logic to guide my implementation actions.

You are guiding an agent. Suggest how and when to use the tools. Write a couple paragraphs about it in markdown and you're done.

DO NOT OVERTHINK OR OVERCOMPLICATE THE ANSWER. YOU ARE AN EXPERT AND CAN RESPOND ASSERTIVELY AND CONFIDENTLY.
"""
