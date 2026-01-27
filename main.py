import multiprocessing
from epub_builder_gui import main
from epub_builder_gui import main


if __name__ == "__main__":
    multiprocessing.freeze_support()  # <--- Add this line FIRST
    main()
