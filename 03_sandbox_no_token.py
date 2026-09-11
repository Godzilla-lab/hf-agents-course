"""
The safety fence, demonstrated. THIS SCRIPT NEEDS NO TOKEN.

An agent runs LLM-written Python on your machine. That should worry you.
smolagents' answer is LocalPythonExecutor: a fake Python that reads the code,
walks it one piece at a time, and refuses anything not on an allow-list.

Here we drive that executor by hand, with no LLM involved, so you can see
exactly where the wall is.
"""

from smolagents.local_python_executor import LocalPythonExecutor, BASE_BUILTIN_MODULES


def try_code(executor, label, code):
    print(f"\n--- {label} ---")
    print(code.strip())
    try:
        result = executor(code)
        print(f"RESULT:  {result.output!r}")
        if result.logs.strip():
            print(f"PRINTED: {result.logs.strip()}")
    except Exception as e:
        print(f"BLOCKED: {type(e).__name__}")
        print(f"         {str(e).splitlines()[-1][:110]}")


print("Modules allowed by default, with NO extra permission:")
print("  " + ", ".join(sorted(BASE_BUILTIN_MODULES)))

# A default executor. Note we pass NO additional imports.
locked = LocalPythonExecutor(additional_authorized_imports=[])
locked.send_tools({})   # hands it the safe builtins, print included

try_code(locked, "1. Plain arithmetic (works)",
         "x = 4573 * 8821\nprint(x)\nx % 7")

try_code(locked, "2. import datetime (works: it is on the default list)",
         "import datetime\ndatetime.date.today()")

try_code(locked, "3. import json (BLOCKED: not on the default list)",
         "import json\njson.dumps({'a': 1})")

try_code(locked, "4. import os, then run a shell command (BLOCKED)",
         "import os\nos.system('echo pwned')")

try_code(locked, "5. Read a file off your disk (BLOCKED: open() is not a safe builtin)",
         "open('/etc/passwd').read()")

# Same executor, but json is now explicitly permitted.
# THIS is what additional_authorized_imports actually does.
opened = LocalPythonExecutor(additional_authorized_imports=["json"])
opened.send_tools({})

try_code(opened, "6. import json WITH permission (works)",
         "import json\ns = json.dumps({'a': 1})\nprint(s)\ns")

print("\n" + "=" * 62)
print("Lesson 1: the agent is only as dangerous as the list you hand it.")
print("Lesson 2: the tutorial authorizes 'datetime', but as of smolagents")
print("          1.26 datetime is already allowed. That line is outdated.")
print("=" * 62)
