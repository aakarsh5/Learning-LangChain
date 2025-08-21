from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings


textSplitter = SemanticChunker(
    HuggingFaceEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text = """The toaster had a rebellious streak. Every morning, it launched slices of bread across the kitchen like it was auditioning for a circus act. One day, the bread landed squarely on the cactus in the corner, which had been quietly plotting its escape from the living room. The cactus took this as a sign—it was time to act.

Outside, the moon winked at the ocean, and the waves blushed in reply. Penguins waddled by in their tuxedos, clearly overdressed for the occasion, but they didn’t mind. Tuesdays always smelled like potential, especially when paired with lemon-scented ambition. Somewhere in the attic, a violin played itself, echoing the rhythm of forgotten memories and half-baked dreams.

Meanwhile, socks continued to vanish mysteriously in the dryer. Some say they attend secret sock conventions, others believe they’ve joined the cactus in its quest for freedom. Either way, the toaster kept launching, the cactus kept scheming, and the moon kept winking—because in this house, even the ordinary had a flair for drama."""

docs = textSplitter.create_documents([text])
print(len(docs))
print(docs)
