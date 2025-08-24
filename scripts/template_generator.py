import os
import argparse

def create_paper_template(title, year, tag):
    folder_name = f"{year}/[{tag}]_{title.replace(' ', '-').lower()}"
    base_path = os.path.join("papers", folder_name)

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "notebooks"), exist_ok=True)

    # Main script
    with open(os.path.join(base_path, "main.py"), "w") as f:
        f.write("# Entry point for implementation\n")

    # Config file
    with open(os.path.join(base_path, "config.yaml"), "w") as f:
        f.write("# Configuration parameters\n")

    # README
    with open(os.path.join(base_path, "README.md"), "w") as f:
        f.write(f"# [{tag}] {title}\n\nImplementation of the paper '{title}' ({year}).\n")

    # Colab link placeholder
    with open(os.path.join(base_path, "colab_link.md"), "w") as f:
        f.write("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kushvinth/paper2code)\n")

    print(f"Template created at: {base_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create paper implementation template.")
    parser.add_argument("--title", required=True, help="Title of the paper")
    parser.add_argument("--year", required=True, help="Year of the paper")
    parser.add_argument("--tag", required=True, help="Field tag (e.g. RL, CV, NLP, Generative)")

    args = parser.parse_args()
    create_paper_template(args.title, args.year, args.tag)
    
    ## Example uv run template_ge.py --title "Attention Is All You Need" --year 2017 --tag NLP
