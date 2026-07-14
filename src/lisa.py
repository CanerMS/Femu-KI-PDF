import json
from os.path import join
from typing import Any, TextIO, TypedDict

class InputItem(TypedDict):
	id: str
	content: str

class OutputItem(TypedDict):
	relevant: bool
	score: float

def read_input(stream: TextIO) -> list[InputItem]:
	data = json.load(stream)

	if not _is_valid_input(data):
		raise RuntimeError("Invalid input")

	return data # type: ignore[no-any-return]

def write_items_to_directory(items: list[InputItem], directory: str) -> None:
	for item in items:
		file = join(directory, item["id"] + ".txt")

		# Always overwrite the file as the content might have changed.
		with open(file, "w", encoding="utf-8") as stream:
			stream.write(item["content"])

def write_output(items: list[OutputItem], stream: TextIO) -> None:
	json.dump(items, stream)

def _is_valid_input(data: Any) -> bool:
	if not isinstance(data, list):
		return False

	for item in data:
		if not isinstance(item, dict):
			return False

		item_id = item.get("id")

		# Identifiers must be alphanumeric so they can be used in filenames without hassle.
		if not isinstance(item_id, str) or not item_id.isalnum():
			return False

		content = item.get("content")

		if not isinstance(content, str):
			return False

	return True
