import sys
from argparse import ArgumentParser, ArgumentTypeError

from feedback import correct

def main() -> None:
	parser = ArgumentParser()
	parser.add_argument("id", type=_check_id)
	parser.add_argument("relevant", choices=["false", "true"])

	args = parser.parse_args()

	try:
		correct(args.id + ".txt", "useful" if args.relevant != "false" else "not_useful")
	except Exception as exception:
		print(exception, file=sys.stderr)
		sys.exit(1)

def _check_id(value: str) -> str:
	if not value.isalnum():
		raise ArgumentTypeError("must be alphanumeric")

	return value

if __name__ == "__main__":
	main()
