# Checks CSV lines for correct formatting and field validity.
from pathlib import Path

# Validates each annotation row: correct column count, non-empty fields, numeric comment ID.
def check_annotations(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.rstrip("\n")

            # Skip empty lines
            if not raw.strip():
                continue

            parts = raw.split("|")

            # Must contain exactly 3 columns
            if len(parts) != 3:
                print(f"[ERROR] Line {i}: found {len(parts)} columns, expected 3.")
                print("       Line content:", repr(raw))
                continue

            image_name, comment_number, comment = [p.strip() for p in parts]

            # Checks for missing image name or comment
            if not image_name or not comment:
                print(f"[ERROR] Line {i}: empty image_name or comment.")
                print("       Line content:", repr(raw))

            # Checks whether comment_number is an integer
            try:
                int(comment_number)
            except ValueError:
                print(f"[ERROR] Line {i}: comment_number is not an integer: {repr(comment_number)}")
                print("       Line content:", repr(raw))


# Entry point that runs annotation validation on the fixed CSV file.
def main():
    csv_path = Path("data/flickr30k/annotations/annotations.csv")
    print("Checking file:", csv_path.resolve())
    check_annotations(csv_path)


if __name__ == "__main__":
    main()
