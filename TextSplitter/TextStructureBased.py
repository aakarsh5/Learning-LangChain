from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """LangChain is a rapidly emerging framework that offers a ver-
satile and modular approach to developing applications powered by large
language models (LLMs). By leveraging LangChain, developers can sim-
plify complex stages of the application lifecycle—such as development,
productionization, and deployment—making it easier to build scalable,
stateful, and contextually aware applications. It provides tools for han-
dling chat models, integrating retrieval-augmented generation (RAG),
and offering secure API interactions. With LangChain, rapid deployment
of sophisticated LLM solutions across diverse domains becomes feasible.
However, despite its strengths, LangChain’s emphasis on modularity and
integration introduces complexities and potential security concerns that
warrant critical examination. This paper provides an in-depth analysis
of LangChain’s architecture and core components, including LangGraph,
LangServe, and LangSmith. We explore how the framework facilitates the
development of LLM applications, discuss its applications across multi-
ple domains, and critically evaluate its limitations in terms of usability,
security, and scalability. By offering valuable insights into both the capa-
bilities and challenges of LangChain, this paper serves as a key resource
for developers and researchers interested in leveraging LangChain for
innovative and secure LLM-powered applications"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)

docs = splitter.split_text(text)

print(docs)