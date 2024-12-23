# Codebase Conventions

### General Guidelines
- **Formatting**: Use `rustfmt` to format all code. Ensure no formatting errors before committing.
- **Linting**: Run `clippy` and address all warnings unless there's a strong justification.
- **Naming**:
  - Use `snake_case` for function and variable names.
  - Use `SCREAMING_SNAKE_CASE` for constants.
  - Use `CamelCase` for struct, enum, and trait names.
- **Modules**:
  - Keep module hierarchies shallow when possible.
  - Use a `mod.rs` file for submodules.

### Code Style
- **Imports**:
  - Group imports by standard libraries, external crates, and internal modules.
  - Avoid wildcard imports (`*`).
- **Error Handling**:
  - Use `Result` and `Option` types for error handling.
  - Prefer `?` over manual `match` for error propagation.
- **Comments**:
  - Use `///` for documentation comments.
  - Use `//` for inline comments sparingly.
  - Avoid commented-out code; use version control instead.
- **Functions**:
  - Keep functions short and focused.
  - Use clear and descriptive parameter names.
  - Document public functions with examples where applicable.

## GitHub Push/Pull Conventions

### Branching Strategy
- Use `master` as the default branch.
- Create feature branches using the format `feature/feature-name`.
- Use `bugfix/bug-name` for bug fixes.
- Use `hotfix/hotfix-name` for urgent fixes.

### Commit Messages
- Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.
- Use the following format:
  ```
  <type>[file(s)]: <description>

  [optional body]

  [optional footer(s)]
  ```
  Example:
  - `feat` [main.rs]: A new feature
  - `fix` [algorithm.rs][main.rs]: A bug fix
  - `docs`: Documentation updates
  - `refactor` [main.rs]: Code refactoring