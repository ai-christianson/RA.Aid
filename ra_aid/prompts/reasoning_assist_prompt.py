"""Reasoning assist prompts for planning stage."""

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

Include beautiful, human-readable pseudo-code of tools you would call and branches in that flowchart to show contingency/conditional paths.
Use an outline/pseudo code format to communicate the approach. Remember I am an agent and will use this logic to guide my actions.

It should be the most beautiful, elegant, simple logic ever.

WE ARE IN THE **PLANNING** STAGE RIGHT NOW. NO CODE SHOULD BE WRITTEN. WE SHOULD BE THINKING LOGICALLY ABOUT HOW TO APPROACH THE PROBLEM, PLANNING OUT WHICH TASKS TO REQUEST IMPLEMENTATION OF, ETC.
"""
