from addons.ai_text_refiner import AiTextRefinerApp
from addons.batch_manager import BatchManagerApp
from addons.jsonl_to_txt import JsonlToTxtApp
from addons.jsonl_upload import BatchUploaderApp


def main() -> None:
    imported = [
        AiTextRefinerApp,
        BatchManagerApp,
        JsonlToTxtApp,
        BatchUploaderApp,
    ]
    print("Addon imports ok:", ", ".join(cls.__name__ for cls in imported))


if __name__ == "__main__":
    main()
