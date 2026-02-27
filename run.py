import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="pretrain")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    print("Mode:", args.mode)

    if args.dry_run:
        print("Dry run successful")
        return

    print("Training would start here")


if __name__ == "__main__":
    main()
