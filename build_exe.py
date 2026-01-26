import subprocess

subprocess.run(
    [
        "pyinstaller",
        "--onefile",
        "--noconsole",
        "epub_builder_gui.py",
    ],
    check=True,
)
