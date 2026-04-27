# Coding Guide

This document describes the coding standards and development workflow used in this repository. The goal is to maintain consistent code style, clean collaboration, and predictable releases.

---

# Naming Conventions

Follow these naming conventions across the repository:

| Type                        | Convention              | Example            |
| --------------------------- | ----------------------- | ------------------ |
| Variables                   | snake_case              | data_frame         |
| Functions                   | snake_case              | load_dataset()     |
| Constants                   | PascalCase              | ModelConfig        |
| Public Classes / Interfaces | PascalCase              | DataProcessor      |
| Private/Internal Functions  | _underscored_snake_case | _internal_helper() |

Keeping naming consistent makes the code easier to read and maintain.

> Unused variables can be replaced with just underscore like this ` _, b = (0, 2.36)`

---

# Development Workflow

This repository follows a Git Flow–inspired workflow to maintain stability while allowing active development.

**Branch Flow**

dev → feature → PR → dev → release PR → main

---

# Repository Branch Structure

## main

Stable, production-ready code.
Always deployable or demo ready.
Protected branch.
No direct pushes allowed.

## dev

Integration branch for development.
All feature work is merged here first.
Protected branch.

## Feature Branches

Feature branches are created from dev.

Naming format:

```feature/#<ticketNo>-<short-description>```

Example:

```feature/#42-risk-classification```

---

# Development Workflow

## 1. Start a Feature

Pull the latest dev branch and create a feature branch.

```
git checkout dev
git pull origin dev
git checkout -b feature/#ticketNo-short-description
```

---

## 2. Work on the Feature

Before implementing any changes, make sure the notebook is synchronized.

```
jupytext --sync main_notebook.py
```

Implement the feature.

After making changes, synchronize the notebook again:

```
jupytext --sync main_notebook.ipynb
```

Guidelines:

Commit regularly.
Use clear commit messages.
Ensure code runs locally before pushing.

---

## 3. Sync With Latest dev

Before creating a Pull Request, update your branch with the latest changes from dev.

```git
git checkout dev
git pull origin dev
git checkout feature/#<ticketNo>-<feature-name>
git merge dev
```

This helps prevent merge conflicts when merging the feature.

---

## 4. Create Pull Request

Open a Pull Request:

`feature/#<ticketNo>-<feature-name> → dev`

The PR description should include:

What was implemented.
Any notes for reviewers.

---

## 5. Code Review

At least one team member approval is required.
Address comments before merging.

---

## 6. Merge Feature

Use Merge Commit when merging into dev.

After merging:

Delete the feature branch (optional but recommended to keep the repository clean).

---

# Iteration / Sprint Release

At the end of a development iteration:

1. Create a Pull Request

```
dev → main
```

2. Review and approve.

3. Merge into main.

4. Synchronize branches

```
main → dev
```

This ensures dev always contains the latest released code.

---

# Notebook Synchronization Rule

Always synchronize the notebook before committing.

```
jupytext --sync main_notebook.py
```

Failure to do so may cause merge conflicts or lost notebook changes.

---

# Module Placement

Keep the repository modular so conflicts are minimized and the notebook stays concise.

Example project structure:

```
├── main_notebook.ipynb
├── main_notebook.py
├── manual_coding_style.md
├── manual_pre-requisite.md
├── modules
│   ├── misc
│   │   └── example.py
│   └── pre_processing
├── README.md
```

---

## Example Module

modules/misc/example.py

```python
def print_hello_world():
    print("Hello World")
```

Usage inside the notebook:

```python
from modules.misc.example import print_hello_world

print_hello_world()
```

This approach keeps the notebook focused on experimentation while placing reusable logic in modules.
