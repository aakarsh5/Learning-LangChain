#this is for representational pourpose it doesnot ensure
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceTB/SmolLM3-3B",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

class Review(TypedDict):
    summary:Annotated[str,"A brief summary of review"]
    sentiment:Annotated[Literal["pos","neg"],"Give sentiment of summary positive or negative"]
    pros : Annotated[Optional[list[str]],"Write all the pros"]
    cons : Annotated[Optional[list[str]],"write all the cons"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""So I’ve been using the new iPad Air (M2) for about two weeks now, and overall, I’m really happy with it.

Performance:
It’s super fast. I mostly use it for reading, watching YouTube/Netflix, note-taking, and some light photo editing in Lightroom. It hasn’t lagged once. Everything opens quickly and runs smoothly. I haven’t tested it with heavy stuff like video editing, but for my needs, it’s more than enough.

Design & Display:
It looks great, really thin and light. I got the blue color and it’s subtle but nice. The screen is sharp and bright, but I wish it had the 120Hz refresh rate like the Pro. Once you’ve seen that smooth scrolling, it’s hard to go back, but I got used to it.

Battery Life:
Battery is solid. I charge it every 2-3 days with casual use. If you’re watching videos or drawing for hours, it might be less, but still decent.

Accessories:
I also got the Apple Pencil Pro and a third-party case. Pencil works great for sketching and taking notes. I didn’t get the Magic Keyboard — too expensive — but I use a Bluetooth one I already had and it works fine.

Minor Gripes:

No Face ID (not a big deal, but would’ve been nice)

128GB base storage is okay for now, but I can see it filling up fast

Price with accessories adds up quick

Final Thoughts:
If you don’t need all the fancy stuff in the iPad Pro, this is a great middle ground. Super fast, looks good, and does almost everything I need. Just wish the screen was a bit smoother.""")

print("before result")
print(result)
print("after result")
print(result['sentiment'])