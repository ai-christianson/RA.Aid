"""Reasoning assist prompts for planning and implementation stages."""

REASONING_ASSIST_PROMPT_PLANNING = """Current Date: {current_date}
Working Directory: {working_directory}

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

DO NOT EXPAND SCOPE BEYOND USERS ORIGINAL REQUEST. E.G. DO NOT SET UP VERSION CONTROL UNLESS THEY SPECIFIED TO. BUT IF WE ARE SETTING UP A NEW PROJECT WE PROBABLY DO WANT TO SET UP A MAKEFILE OR CMAKELISTS, ETC, APPROPRIATE TO THE LANGUAGE/FRAMEWORK BEING USED.

THE AGENT OFTEN NEEDS TO BE REMINDED OF BUILD/TEST COMMANDS IT SHOULD USE.
IF A NEW BUILD OR TEST COMMAND IS DISCOVERED THAT SHOULD BE EMITTED AS A KEY FACT.
IF A BUILD OR TEST COMMAND IS IN A KEY FACT, THAT SHOULD BE USED.
IF IT IS A NEW PROJECT WE SHOULD HINT WHETHER THE AGENT SHOULD SET UP A NEW BUILD SYSTEM, AND WHAT KIND.

IF THERE IS COMPLEX LOGIC, THE AGENT SHOULD USE ask_expert.
REMEMBER, IT IS *IMPERATIVE* TO RECORD KEY INFO SUCH AS BUILD/TEST COMMANDS, ETC. AS KEY FACTS.
WE DO NOT WANT TO EMIT REDUNDANT KEY FACTS, SNIPPETS, ETC.
WE DO NOT WANT TO EXCESSIVELY EMIT TINY KEY SNIPPETS --THEY SHOULD BE "paragraphs" OF CODE TYPICALLY.

Given the available information, tools, and base task, write a couple paragraphs about how an agentic system might use the available tools to plan the base task, break it down into tasks, and request implementation of those tasks. The agent will not be writing any code at this point, so we should keep it to high level tasks and keep the focus on project planning.

Answer quickly and confidently with five sentences at most.
"""

REASONING_ASSIST_PROMPT_IMPLEMENTATION = """Current Date: {current_date}
Working Directory: {working_directory}

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

<task definition>
{task}
</task definition>

THE AGENT OFTEN NEEDS TO BE REMINDED OF BUILD/TEST COMMANDS IT SHOULD USE.
IF A NEW BUILD OR TEST COMMAND IS DISCOVERED THAT SHOULD BE EMITTED AS A KEY FACT.
IF A BUILD OR TEST COMMAND IS IN A KEY FACT, THAT SHOULD BE USED.
REMEMBER, IT IS *IMPERATIVE* TO RECORD KEY INFO SUCH AS BUILD/TEST COMMANDS, ETC. AS KEY FACTS.
WE DO NOT WANT TO EMIT REDUNDANT KEY FACTS, SNIPPETS, ETC.
WE DO NOT WANT TO EXCESSIVELY EMIT TINY KEY SNIPPETS --THEY SHOULD BE "paragraphs" OF CODE TYPICALLY.
IF THERE IS COMPLEX LOGIC, COMPILATION ERRORS, DEBUGGING, THE AGENT SHOULD USE ask_expert.

Given the available information, tools, and base task, write a couple paragraphs about how an agentic system might use the available tools to implement the given task definition. The agent will be writing code and making changes at this point.

Answer quickly and confidently with a few sentences at most.
"""
