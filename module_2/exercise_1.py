from sentence_transformers import SentenceTransformer


def main() -> None:
    model_name = "BAAI/bge-base-en-v1.5"

    model = SentenceTransformer(model_name)
    res = model.encode("RAG is awesome")
    print(res.shape)

    model.encode(["apple", "car"])
    print(res[:100])


if __name__ == "__main__":
    main()
