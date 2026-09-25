from google import genai
import json

client = genai.Client()


calc_tool = {
    "type": "function",
    "name": "calc_tool",
    "description": "Gets the value from add/sub/mul/div of two operands.",
    "parameters": {
        "type": "object",
        "properties": {
            "op1": {
                "type": "integer",
                "description": "Operand 1",
            },
            "op2": {
                "type": "integer",
                "description": "Operand 2",
            },
            "operation": {
                "type": "string",
                "description": "Operation (add/sub/mul/div)",
            },
        },
        "required": ["op1","op2"],
    },
}
available_functions = {
    "calc_tool": lambda op1, op2, operation: {
        "answer": {
            "add": op1 + op2,
            "sub": op1 - op2,
            "mul": op1 * op2,
            "div": op1 / op2
        }[operation]
    },
}
history = [
    {
        "type": "user_input",
        "content": [{"type": "text", "text": "What is 7/7?"}]
    }
]

while True:
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=history,
        store=False,
        tools=[calc_tool],
    )
    function_results=[]
    for step in interaction.steps:
        history.append(step.model_dump())
        if step.type == "function_call":
            print("this is a function call")
            result = available_functions[step.name](**step.arguments)
            print(f"Called {step.name}({step.arguments}) → {result}")
            fn_result = {
                "type": "function_result",
                "name": step.name,
                "call_id": step.id,
                "result": [{"type": "text", "text": json.dumps(result)}],
            }
            function_results.append(fn_result)
            history.append(fn_result)
    if not function_results:
        break

print(interaction.output_text)