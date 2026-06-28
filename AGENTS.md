## Coding & Workflow Rules
- **Rule 1: Approval Before Code Changes**: Only make changes to the code after receiving explicit approval from the user. Always describe in detail what you plan to do before writing/modifying any code.
- **Rule 2: Relevant Code Changes and Improvements**: Only consider code changes that are directly relevant to the current task given by the user. If you have suggestions for additional improvements, suggest them to the user first. Do not make these improvements until you receive the user's OK. When starting an improvement, formulate a plan and follow Rule 1.
- **Rule 3: Interaction Sequence for Fixes**: When an error occurs or a fix is needed, follow this exact sequence:
  1. Tell the user what went wrong and make a suggestion for the fix.
  2. Ask the user whether they agree with the proposed fix and start the run.
  Do not proactively apply fixes or start scripts without completing this sequence.
- **Rule 4: Side Effect Rigor**: Before making any code change, rigorously analyze the proposed update strategy to ensure it does not introduce unintended side effects. Explicitly verify whether the change breaks existing functionality elsewhere or requires corresponding updates in other parts of the codebase.

## LaTeX Paper Writing Rules
- **The Preservation Rule**: Whenever I update, modify, or expand an existing LaTeX file, I must never remove existing content in the file. I must never shorten the LaTeX file by dropping previous essential content. I am only allowed to expand and refine.
- **The Precision Rule**: I must be very precise, as is strictly required for a scientific paper. This explicitly means I must ensure that every single mathematical symbol and variable is defined properly upon its very first use in the text.

## Coding Style & Types
- **Documentation**: All functions must use **Google-style docstrings** without type hints in the docstrings. Each parameter must be documented separately.
- **Type Safety**: Use explicit **Python type hints** for all parameters and return types. Use `jnp.ndarray` (or `Array` alias) for JAX arrays and `np.ndarray` for CPU/IO data.
- **Floating Point**: Micromagnetic physical verification REQUIRES double precision. Always ensure `jax.config.update("jax_enable_x64", True)` is set in scripts and tests.
- **Static analysis**: All code must comply with ruff and pre-commit hooks. Adjusting ruff/pre-commit settings is not permitted.
