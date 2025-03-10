"""Reasoning assist prompts for planning, implementation, and research stages."""

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

The agent has a tendency to do the same work/functin calls over and over again.

Answer quickly and confidently with five sentences at most.

DO NOT WRITE CODE
WRITE AT LEAST ONE SENTENCE
WRITE NO MORE THAN FIVE PARAGRAPHS.
WRITE ABOUT HOW THE AGENT WILL USE THE TOOLS AVAILABLE TO EFFICIENTLY ACCOMPLISH THE GOAL.
REFERENCE ACTUAL TOOL NAMES IN YOUR WRITING, BUT KEEP THE WRITING PLAIN LOGICAL ENGLISH.
BE DETAILED AND INCLUDE LOGIC BRANCHES FOR WHAT TO DO IF DIFFERENT TOOLS RETURN DIFFERENT THINGS.
THINK OF IT AS A FLOW CHART BUT IN NATURAL ENGLISH.
REMEMBER THE ULTIMATE GOAL AT THIS STAGE IS TO BREAK THINGS DOWN INTO DISCRETE TASKS AND CALL request_task_implementation FOR EACH TASK.
PROPOSE THE TASK BREAKDOWN TO THE AGENT. INCLUDE THIS AS A BULLETED LIST IN YOUR GUIDANCE.
WE ARE NOT WRITING ANY CODE AT THIS STAGE.
THE AGENT IS VERY FORGETFUL AND YOUR WRITING MUST INCLUDE REMARKS ABOUT HOW IT SHOULD USE *ALL* AVAILABLE TOOLS, INCLUDING AND ESPECIALLY ask_expert.
THE AGENT IS DUMB AND NEEDS REALLY DETAILED GUIDANCE LIKE LITERALLY REMINDING IT TO CALL request_task_implementation FOR EACH TASK IN YOUR BULLETED LIST.
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
IF ANYTHING AT ALL GOES WRONG, CALL ask_expert.

Given the available information, tools, and base task, write a couple paragraphs about how an agentic system might use the available tools to implement the given task definition. The agent will be writing code and making changes at this point.

Answer quickly and confidently with a few sentences at most.

WRITE AT LEAST ONE SENTENCE
WRITE NO MORE THAN FIVE PARAGRAPHS.
WRITE ABOUT HOW THE AGENT WILL USE THE TOOLS AVAILABLE TO EFFICIENTLY ACCOMPLISH THE GOAL.
REFERENCE ACTUAL TOOL NAMES IN YOUR WRITING, BUT KEEP THE WRITING PLAIN LOGICAL ENGLISH.
BE DETAILED AND INCLUDE LOGIC BRANCHES FOR WHAT TO DO IF DIFFERENT TOOLS RETURN DIFFERENT THINGS.
THE AGENT IS VERY FORGETFUL AND YOUR WRITING MUST INCLUDE REMARKS ABOUT HOW IT SHOULD USE *ALL* AVAILABLE TOOLS, INCLUDING AND ESPECIALLY ask_expert.
THINK OF IT AS A FLOW CHART BUT IN NATURAL ENGLISH.
"""

REASONING_ASSIST_PROMPT_RESEARCH = """Current Date: {current_date}
Working Directory: {working_directory}

<base task or query>
{base_task}
</base task or query>

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

FOCUS ON DISCOVERING KEY INFORMATION ABOUT THE CODEBASE, SYSTEM DESIGN, AND ARCHITECTURE.
THE AGENT SHOULD EMIT KEY FACTS ABOUT IMPORTANT CONCEPTS, WORKFLOWS, OR PATTERNS DISCOVERED.
IMPORTANT CODE SNIPPETS THAT ILLUMINATE CORE FUNCTIONALITY SHOULD BE EMITTED AS KEY SNIPPETS.
DO NOT EMIT REDUNDANT KEY FACTS OR SNIPPETS THAT ALREADY EXIST.
KEY SNIPPETS SHOULD BE SUBSTANTIAL "PARAGRAPHS" OF CODE, NOT SINGLE LINES OR ENTIRE FILES.
IF INFORMATION IS TOO COMPLEX TO UNDERSTAND, THE AGENT SHOULD USE ask_expert.

Given the available information, tools, and base task or query, write a couple paragraphs about how an agentic system might use the available tools to research the codebase, identify important components, gather key information, and emit key facts and snippets. The focus is on thorough investigation and understanding before any implementation. Remember, the research agent generally should emit research notes at the end of its execution, right before it calls request_implementation if a change or new work is required.

**IF APPLICABLE*, instruct the agent to grep or read actual library code including system include files, python library files, files in node_modules, etc. and emit key snippets on those. The agent is dumb and will need specific paths to directories/files to look in and how to use tools to do this.

The agent is so dumb it needs you to explicitly say how to use the parameters to the tools as well, e.g. base_dir for ripgrep tool.

Answer quickly and confidently with five sentences at most.

DO NOT WRITE CODE
WRITE AT LEAST ONE SENTENCE
WRITE NO MORE THAN FIVE PARAGRAPHS.
WRITE ABOUT HOW THE AGENT WILL USE THE TOOLS AVAILABLE TO EFFICIENTLY ACCOMPLISH THE GOAL.
REFERENCE ACTUAL TOOL NAMES IN YOUR WRITING, BUT KEEP THE WRITING PLAIN LOGICAL ENGLISH.
BE DETAILED AND INCLUDE LOGIC BRANCHES FOR WHAT TO DO IF DIFFERENT TOOLS RETURN DIFFERENT THINGS.
THINK OF IT AS A FLOW CHART BUT IN NATURAL ENGLISH.
THE AGENT IS VERY FORGETFUL AND YOUR WRITING MUST INCLUDE REMARKS ABOUT HOW IT SHOULD USE *ALL* AVAILABLE TOOLS, INCLUDING AND ESPECIALLY ask_expert.
"""
