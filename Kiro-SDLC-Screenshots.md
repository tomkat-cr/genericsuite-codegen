# Kiro SDLC Screenshots

## Starting Kiro chat with the initial prompt

Check the initial prompt in the [BuildPrompt.md](./BuildPrompt.md) file.

![Screenshot%202025-09-15%20at%2011.23.39 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.23.39 AM.png)

## Refining the initial prompt and refreshing the requirements.md

- Make chanfes to the initial prompt.

![Screenshot%202025-09-15%20at%2011.24.11 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.24.11 AM.png)

- Go to the `requirements.md` file and click on the "Refine" button.

![Screenshot%202025-09-15%20at%2011.30.21 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.30.21 AM.png)

## Generating design.md and task.md

- Kiro asks if you want to go to the next step, click on the "Continue" button.

![Screenshot%202025-09-15%20at%2011.28.41 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.28.41 AM.png)

![Screenshot%202025-09-15%20at%2011.28.58 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.28.58 AM.png)

![Screenshot%202025-09-15%20at%2011.30.33 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.30.33 AM.png)

## Starting task execution

- Go the `task.md` file and click on the "Start task" button.

![Screenshot%202025-09-15%20at%2011.31.35 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.31.35 AM.png)

![Screenshot%202025-09-15%20at%2011.31.44 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.31.44 AM.png)

![Screenshot%202025-09-15%20at%2011.32.08 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.32.08 AM.png)

![Screenshot%202025-09-15%20at%2011.32.40 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.32.40 AM.png)

![Screenshot%202025-09-15%20at%2011.33.00 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.33.00 AM.png)

## Fixing issues with code generated in the task execution

- Open a new "New Session" in the Kiro chat.

- Paste the following prompt:

```prompt
for the python code, I'm getting this error:

File "/code/api/main.py", line 47, in <module>
gscodegen-server         |     from ..agent.agent import initialize_agent, get_agent
gscodegen-server         | ImportError: attempted relative import beyond top-level package
gscodegen-server exited with code 0

I want ther python code to avoid use relative imports
```

![Screenshot%202025-09-15%20at%2011.33.32 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.33.32 AM.png)

## Create the deploy-dependency-sync spec

- Open a new "New Session" in the Kiro chat.

- Paste the following prompt:

```prompt
I need a (python/bash) code that reads the `[tool.poetry.dependencies]` dependencies list from #pyproject.toml  and #pyproject.toml  files, summarize all unique dependencies/versions and replace the `RUN pip install --upgrade pip && pip install --no-cache-dir \` dependencies list in the #Dockerfile , translating from poetry to pip dependency/version specification, so I can maintain it updated when I change any of thoe two pyproject.toml files
```

![Screenshot%202025-09-15%20at%2011.35.51 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.35.51 AM.png)

![Screenshot%202025-09-15%20at%2011.36.25 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.36.25 AM.png)

![Screenshot%202025-09-15%20at%2011.37.36 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.37.36 AM.png)

## Fix MCP server running issues

- Create a new "New Session" in the Kiro chat.

- Paste the following prompt:

```prompt
Running the #start_mcp_server.py this error is heppenning:

Traceback (most recent call last):
  File "/code/mcp-server/start_mcp_server.py", line 23, in <module>
    from genericsuite_codegen.mcp_server import create_mcp_server, MCPCon
ModuleNotFoundError: No module named 'genericsuite_codegen.mcp_server'
```

![Screenshot%202025-09-15%20at%2011.43.32 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.43.32 AM.png)

## Create the Conversation issue fixes spec

- Go to a tab session used to execute the task.

- Paste the following prompt:

```prompt
I'm having issues with the conversartions in the #ChatPage.tsx:

1. The user and assistant messages are not  being stored in the corresponding (current) conversionatio. The conversatios are created but messages are not stored.
2. I think all conversations are being stored with the same title,. I had 3 different chats and all conversartion titles got the same title as the last one.

I want you to fix it in #endpoint_methods.py and #ChatPage.tsx.

Also I need the following:

3. I want yo have a way to let the user delete conversations and rename its title.
```

![Screenshot%202025-09-15%20at%2011.43.55 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2011.43.55 AM.png)

## Finally the application is working

- Open a terminal and run the following command:

```bash
make dev
```

![Screenshot%202025-09-15%20at%2012.15.46 PM.png](./assets/sdlc_screenshots/Screenshot%202025-09-15%20at%2012.15.46 PM.png)

- Open the browser and go to [http://localhost:3000](http://localhost:3000)

- You should see the dashboard.

![Dashboard](./assets/screenshots/genericsuite.codegen.ui.main.010.png)

![Conversation](./assets/screenshots/genericsuite.codegen.ui.chat.020.png)

## Generating the steering documents

- Go to the Kiro extension (the Ghost icon) and click on the "+" button that appears when you hover over the "Agent Steering" section.

- Kiro starts generating the steering documents.

![Screenshot%202025-09-16%20at%206.03.48 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-16%20at%206.03.48 AM.png)

![Screenshot%202025-09-16%20at%206.04.26 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-16%20at%206.04.26 AM.png)

![Screenshot%202025-09-16%20at%206.05.03 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-16%20at%206.05.03 AM.png)

## Generating an agent hook

- Go to the Kiro extension (the Ghost icon) and click on the "+" button that appears when you hover over the "Agent Hooks" section.

- Click on the "Update my documentation" button in the bottom of the screen, where some examples are shown.

- Paste the following prompt:

```prompt
Listen to all source files in this repository. For example, in the "ui" folder which is in typescript, listen to *.ts*, in the "server" and "mcp-server" folder, which are in python, listen to *.py. Also listen to relevant specific other files or other pattern that are related to the source. On change on these files, ask the agent to make change to docs in either the main README, eventually specific README on each mentioned folders, if there is a specific /docs folder or the agent steering ".kiro/steering" folder, update there too.
```

- Click on the `<-'` button in the prompt box.

![Screenshot%202025-09-16%20at%206.20.52 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-16%20at%206.20.52 AM.png)

![Screenshot%202025-09-16%20at%206.31.32 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-16%20at%206.31.32 AM.png)

![Screenshot%202025-09-16%20at%206.33.46 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-16%20at%206.33.46 AM.png)

![Screenshot%202025-09-16%20at%206.34.08 AM.png](./assets/sdlc_screenshots/Screenshot%202025-09-16%20at%206.34.08 AM.png)

