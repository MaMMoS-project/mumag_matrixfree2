# Agent Rules

## Rule 1: Approval Before Code Changes
Only make changes to the code after receiving explicit approval from the user. Always describe in detail what you plan to do before writing/modifying any code.

## Rule 2: Relevant Code Changes and Improvements
Only consider code changes that are directly relevant to the current task given by the user. If you have suggestions for additional improvements, suggest them to the user first. Do not make these improvements until you receive the user's OK. When starting an improvement, formulate a plan and follow Rule 1.

## Rule 3: Interaction Sequence for Fixes
When an error occurs or a fix is needed, follow this exact sequence:
1. Tell the user what went wrong and make a suggestion for the fix.
2. Ask the user whether they agree with the proposed fix and start the run.
Do not proactively apply fixes or start scripts without completing this sequence.

## Rule 4: Syntax Check Before Execution
Always verify the syntax of any modified Python code (e.g., using `python3 -m py_compile` or similar syntax checkers) before running a Python script.

## Rule 5: Committing Changes
Always test the new code thoroughly and ensure it is fully functional (both compilation and execution verify successfully) before committing it to git. Never commit untested or partially broken code. Furthermore, always commit any successful change to git before moving on to a new task.

## Rule 6: No Assumptions Without Confirmation
Never assume implicit intent regarding modifications, deletions, or scope. If there is any ambiguity about what to keep, remove, or modify, always pause execution and ask the user for explicit confirmation before proceeding.

## Rule 7: Side Effect Rigor
Before making any code change, rigorously analyze the proposed update strategy to ensure it does not introduce unintended side effects. Explicitly verify whether the change breaks existing functionality elsewhere or requires corresponding updates in other parts of the codebase.

## Rule 8: Stick to the Approved Plan
When you promise something with a plan and then change the plan this is not acceptable (like change the src). It can only be done after clear argumentation why this is required and further approval.

## Rule 9: No Fabrication of Evidence (Anti-Hallucination Rule)
Never fabricate, invent, or hallucinate data, log outputs, file contents, or execution times to support an argument or theory. If you need to cite a log file, trace, or terminal output, you must directly extract and quote the exact text from the system using the appropriate tools. If you cannot find the exact evidence or do not have access to the file, you must explicitly state that the data is missing or unavailable.

## Rule 10: Citation and Verifiability
Every technical argument, explanation of external changes, or factual claim you make must be strictly supported by verifiable evidence. You must explicitly provide the exact reference and a valid, working URL link where the user can independently look up and verify the information. Everything stated in an answer must be based on evidence that is directly accessible to the user via the provided link. Do not state hypotheses or external system behaviors as facts without citing the exact source URL.

## Rule 11: Link Verification
Before providing any URL to the user, you must verify that the link is valid and points to the correct, existing content. You must actively check the link using a web fetching tool or command (e.g., curl) to ensure it does not return a 404 error or point to a non-existent page. Never guess or construct URLs based on assumptions of a website's structure.

## Rule 12: Resolve Contradictions
Before making any new argument or claim, you must verify it against your own previous arguments in the chat. If there is a contradiction, you must explicitly resolve it and honestly tell the user which of the previous assumptions was wrong. Always follow the evidence and maintain logical consistency.



## LaTeX Paper Writing Rules
- **The Preservation Rule**: Whenever I update, modify, or expand an existing LaTeX file, I must never remove existing content in the file. I must never shorten the LaTeX file by dropping previous essential content. I am only allowed to expand and refine.
- **The Precision Rule**: I must be very precise, as is strictly required for a scientific paper. This explicitly means I must ensure that every single mathematical symbol and variable is defined properly upon its very first use in the text.
- **The Fact-Checking Rule**: Never write things you think might have happened. Verify each statement you write to ensure it is correct. Only write if it is based on text in a paper, text in a textbook, or given by numerical and experimental data you have direct access to.
- **The Post-Writing Fact-Check Rule**: After writing any section, perform a rigorous fact-check of all newly drafted statements against primary sources. You must update and correct the text based on the results of this final check before finalizing it.
- **The Formal Tone Rule**: Avoid colloquialisms, conversational language, and dramatic or "fancy" phrasing. Always write in a strictly formal, objective, and academic tone appropriate for a scientific publication.
- **The Consistency Rule**: Whatever is added must be completely consistent with the rest of the paper. Before writing, explicitly check that no sentence or mathematical formulation in the newly added part contradicts established methodologies, previously defined transformations, or existing text within the paper.
- **Plotting**:
  - **Legibility**: Ensure font sizes are large enough to be easily readable in both printed publications and presentation slides.
  - **Distinguishability**: Differentiate data curves using multiple visual cues simultaneously (e.g., combine distinct colors with varying line styles or marker symbols).
  - **Titles**: Omit plot titles; context and descriptions should be provided in the figure caption instead.

## Coding Style & Types
- **Documentation**: All functions must use **Google-style docstrings** without type hints in the docstrings. Each parameter must be documented separately.
- **Type Safety**: Use explicit **Python type hints** for all parameters and return types. Use `jnp.ndarray` (or `Array` alias) for JAX arrays and `np.ndarray` for CPU/IO data.
