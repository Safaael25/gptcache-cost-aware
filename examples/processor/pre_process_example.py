import time
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.processor.pre import get_system_and_last_content, get_role_and_last_content
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

cache.set_openai_key()

onnx = Onnx()
data_manager = manager_factory("sqlite,faiss", vector_params={"dimension": onnx.dimension})

cache.init(
    pre_process_func=get_system_and_last_content,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)

question = "Who won the world series in 2020?"

for _ in range(3):
    start = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": question}
        ],
    )
    print(round(time.time() - start, 3))
    print(response["choices"][0]["message"]["content"])