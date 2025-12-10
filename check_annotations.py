from pathlib import Path

def check_annotations(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.rstrip("\n")

            # Üres sorok átugrása
            if not raw.strip():
                continue

            parts = raw.split("|")

            # 1) Ellenőrizzük, hogy van-e pontosan 3 mező
            if len(parts) != 3:
                print(f"[HIBA] Sor {i}: {len(parts)} oszlopot találtam, 3 helyett.")
                print("       Sor tartalma:", repr(raw))
                continue

            image_name, comment_number, comment = [p.strip() for p in parts]

            # 2) Ellenőrizzük, hogy van-e image_name és comment
            if not image_name or not comment:
                print(f"[HIBA] Sor {i}: üres image_name vagy comment.")
                print("       Sor tartalma:", repr(raw))

            # 3) Ellenőrizzük, hogy a comment_number tényleg egész szám-e
            try:
                int(comment_number)
            except ValueError:
                print(f"[HIBA] Sor {i}: comment_number nem integer: {repr(comment_number)}")
                print("       Sor tartalma:", repr(raw))


def main():
    csv_path = Path("data/flickr30k/annotations/annotations.csv")
    print("Ellenőrzöm a fájlt:", csv_path.resolve())
    check_annotations(csv_path)


if __name__ == "__main__":
    main()
