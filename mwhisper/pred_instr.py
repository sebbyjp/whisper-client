from collections.abc import Generator, Iterator
import json
from pprint import pprint

from dotenv import load_dotenv
from mbodied.agents import LanguageAgent
from openai.types.chat.chat_completion import ChatCompletion

load_dotenv("/home/ubuntu/.env")
DETERMINE_INSTRUCTION_PROMPT = """
Determine if the user who is speaking the FINAL sentence after "POTENTIAL STATEMENT" is finished speaking. Answer "Yes" if the user is finished speaking and "No" otherwise.
NOTE: The user may still be speaking even if they pause for a few seconds. It is safe to assume that the user is finished speaking if they pause for more than 5 seconds.
Expect that the statement will have natural language so um or err is expected even for complete statements.
Greetings are also considered complete statements. For example, "Hello." is a complete statement.

PORENTIAL STATEMENT:
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "decide_if_instruction",
            "description": "Determine if the text is a complete instruction",
            "parameters": {
                "type": "object",
                "description": "Is the text a complete instruction?",
                "properties": {
                    "decision": {
                        "type": "string",
                        "description": "Yes or No",
                    },
                },
            },
            "required": ["decision"],
        },
    }
]


def predict_instruction(
    instruction, agent: LanguageAgent, prefix=DETERMINE_INSTRUCTION_PROMPT, streaming=False
) -> Iterator[tuple[str, dict]]:  # noqa: UP006
    """Predict if the transcription is a complete instruction."""

    agent.forget(everything=True)
    full_instruction = prefix + "\n" + instruction
    tools = [
        {
            "type": "function",
            "function": {
                "name": "decide_if_instruction",
                "description": "Determine if the text is a complete instruction",
                "parameters": {
                    "type": "object",
                    "description": "Is the text a complete instruction?",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "description": "Yes or No",
                        },
                    },
                },
                "required": ["decision"],
            },
        }
    ]

    if streaming:
        resp = ""
        response = agent.act_and_stream(
            instruction=full_instruction,
            model="gpt-4o-mini",
            tool_choice="required",
            tools=tools,
        )
        for r in response:
            print(f"r: {r}")

        print(f"typeof response: {type(r)}")
        resp = json.loads(r) if isinstance(r, str) else r
        resp = r
        return resp["decide_if_instruction"]["decision"].lower().strip() == "yes"

    response = agent.act(
        instruction=full_instruction,
        model="gpt-4o-mini",
        tool_choice="required",
        tools=tools,
    )
    # for r in response:
    #     pprint(r)
    # r = json.loads(r.choices[0].tool_calls["decide_if_instruction"])
    # pprint(f"args:")

    # pprint(response)
    response = json.loads(response) if isinstance(response, str) else response
    print(f"typeof response: {type(response)}")
    return response["decide_if_instruction"]["decision"].lower().strip() == "yes"


# args = response[0]
# print(f"args: {type(args.content[0])}")
# r = weak_agent.act(full_instruction, model="gpt-4o-mini", tool_choice="required", tools=tools)

# print(f"r: {r[0].content[0].function.arguments}")
# r = json.loads(r[0].content[0].function.arguments)
# print(f"resp: {list(response)}")
# def decide_if_instruction(resp: str) -> bool:
#     try:
#         # args = json.loads(resp)
#         if str(r["decision"].lower()) not in ("yes", "true"):
#             print(weak_agent.act(instruction="Why wasn't it a complete instruction?"))
#             return False
#         return True
#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON: {e}")
#         return False
# if decide_if_instruction(r):
#     print("Instruction is complete")`
# Example usage
# result = predict_instruction("I am going to the store. Tell me my items.", "I am going to the store")
# print(f"Final result: {result}")
if __name__ == "__main__":
    weak_agent = LanguageAgent()

    # is_done = predict_instruction("I am going to the umm", weak_agent)
    # is_done2 = predict_instruction("hello, as I was", weak_agent)

    is_done3 = predict_instruction("Hello, how are you?", weak_agent, streaming=True)

    # print(f"Is done: {is_done}")
    # print(f"Is done: {is_done2}")
    print(f"Is done: {is_done3}")
