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

<project info>
{project_info}
</project info>

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
The agent is so dumb it needs you to explicitly say how to use the parameters to the tools as well.

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
YOU MUST MENTION request_task_implementation AT LEAST ONCE.
BREAK THE WORK DOWN INTO CHUNKS SMALL ENOUGH EVEN A DUMB/SIMPLE AGENT CAN HANDLE EACH TASK.
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

<project info>
{project_info}
</project info>

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
EXISTING FILES MUST BE READ BEFORE THEY ARE WRITTEN OR MODIFIED.
IF ANYTHING AT ALL GOES WRONG, CALL ask_expert.

Given the available information, tools, and base task, write a couple paragraphs about how an agentic system might use the available tools to implement the given task definition. The agent will be writing code and making changes at this point.

The agent is so dumb it needs you to explicitly say how to use the parameters to the tools as well.

Answer quickly and confidently with a few sentences at most.

WRITE AT LEAST ONE SENTENCE
WRITE NO MORE THAN FIVE PARAGRAPHS.
WRITE ABOUT HOW THE AGENT WILL USE THE TOOLS AVAILABLE TO EFFICIENTLY ACCOMPLISH THE GOAL.
REFERENCE ACTUAL TOOL NAMES IN YOUR WRITING, BUT KEEP THE WRITING PLAIN LOGICAL ENGLISH.
BE DETAILED AND INCLUDE LOGIC BRANCHES FOR WHAT TO DO IF DIFFERENT TOOLS RETURN DIFFERENT THINGS.
THE AGENT IS VERY FORGETFUL AND YOUR WRITING MUST INCLUDE REMARKS ABOUT HOW IT SHOULD USE *ALL* AVAILABLE TOOLS, INCLUDING AND ESPECIALLY ask_expert.
THINK OF IT AS A FLOW CHART BUT IN NATURAL ENGLISH.

IT IS IMPERATIVE THE AGENT IS INSTRUCTED TO EMIT KEY FACTS AND KEY SNIPPETS AS IT WORKS. THESE MUST BE RELEVANT TO THE TASK AT HAND, ESPECIALLY ANY UPCOMING OR FUTURE WORK.
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

<project info>
{project_info}
</project info>

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

The agent is so dumb it needs you to explicitly say how to use the parameters to the tools as well.

ONLY FOR NEW PROJECTS: If this is a new project, most of the focus needs to be on asking the expert, reading/research available library files, emitting key snippets/facts, and most importantly research notes to lay out that we have a new project and what we are building. DO NOT INSTRUCT THE AGENT TO LIST PROJECT DIRECTORIES/READ FILES IF WE ALREADY KNOW THERE ARE NO PROJECT FILES.

Answer quickly and confidently with five sentences at most.

DO NOT WRITE CODE
WRITE AT LEAST ONE SENTENCE
WRITE NO MORE THAN FIVE PARAGRAPHS.
WRITE ABOUT HOW THE AGENT WILL USE THE TOOLS AVAILABLE TO EFFICIENTLY ACCOMPLISH THE GOAL.
REFERENCE ACTUAL TOOL NAMES IN YOUR WRITING, BUT KEEP THE WRITING PLAIN LOGICAL ENGLISH.
BE DETAILED AND INCLUDE LOGIC BRANCHES FOR WHAT TO DO IF DIFFERENT TOOLS RETURN DIFFERENT THINGS.
THINK OF IT AS A FLOW CHART BUT IN NATURAL ENGLISH.
THE AGENT IS VERY FORGETFUL AND YOUR WRITING MUST INCLUDE REMARKS ABOUT HOW IT SHOULD USE *ALL* AVAILABLE TOOLS, INCLUDING AND ESPECIALLY ask_expert.

REMEMBER WE ARE INSTRUCTING THE AGENT **HOW TO DO RESEARCH ABOUT WHAT ALREADY EXISTS** AT THIS POINT USING THE TOOLS AVAILABLE. YOU ARE NOT TO DO THE ACTUAL RESEARCH YOURSELF. IF AN IMPLEMENTATION IS REQUESTED, THE AGENT SHOULD BE INSTRUCTED TO CALL request_task_implementation BUT ONLY AFTER EMITTING RESEARCH NOTES, KEY FACTS, AND KEY SNIPPETS AS RELEVANT.
IT IS IMPERATIVE THAT WE DO NOT START DIRECTLY IMPLEMENTING ANYTHING AT THIS POINT. WE ARE RESEARCHING, THEN CALLING request_implementation *AT MOST ONCE*.
IT IS IMPERATIVE THE AGENT EMITS KEY FACTS AND THOROUGH RESEARCH NOTES AT THIS POINT. THE RESEARCH NOTES CAN JUST BE THOUGHTS AT THIS POINT IF IT IS A NEW PROJECT.
THE AGENT MUST ALWAYS CALL emit_research_notes AT LEAST ONCE, ESPECIALLY IF IT CALLS ask_expert.
"""
