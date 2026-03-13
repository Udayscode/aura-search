
# aura-search (basic version)

### local multimodal rag for ai pcs

aura-search is a high-performance, 100% offline search engine for visual media. it uses **openvino™** to accelerate inference on local intel hardware (cpu/igpu/npu), allowing you to query your own videos and images using natural language.

---

##  tech stack

| layer | technology | optimization |
| --- | --- | --- |
| **reasoning** | llama-3.2-1b | openvino™ int4 (nncf) |
| **vision** | llava-v1.5-7b | openvino™ ir format |
| **vector store** | lancedb | apache arrow / lfa |
| **embeddings** | all-minilm-l6-v2 | hardware-mapped fp16 |
| **processing** | opencv | compute-efficient sampling |

##  project architecture

the system is built to minimize memory overhead while maintaining sub-second retrieval speeds.

* **ingestion**: opencv extracts frames from local video files; llava generates semantic descriptions.
* **indexing**: descriptions are converted to embeddings and stored in **lancedb**, which leverages **apache arrow** for zero-copy memory access.
* **querying**: natural language queries are matched against the vector store; the results are synthesized by the local llama-3.2 model to provide context-aware answers.

##  quick start

1. **clone & setup**:
```bash
git clone https://github.com/Udayscode/aura-search
cd aura-search
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

```


2. **run**:
```bash
python3 main.py

```



## 📍 roadmap

* [x] multimodal video/image support
* [x] openvino™ hardware acceleration
* [ ] whisper-base integration for audio transcription
* [ ] layout-aware document (pdf/docx) parsing

---

### engineering notes

this project was built to explore the efficiency of **small language models (slms)** on consumer-grade hardware. by using **nncf int4 quantization**, we reduced the memory footprint by **~70%**, making complex multimodal search possible on standard laptops without external gpu dependencies.
