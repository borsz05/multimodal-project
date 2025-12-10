from src.config import paths

def main():
    print("Project root:", paths.project_root)
    print("Data dir:", paths.data_dir)
    print("Results dir:", paths.results_dir)
    print("Models dir:", paths.models_dir)

if __name__ == "__main__":
    main()
