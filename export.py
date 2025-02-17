from main import NeuralNetwork, MemoryCell, Game, create_tileset
import glob
import torch
import math
import gzip
import sys
import os

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint_paths = glob.glob("out/takeiteasy*.pth.gz")
    highest_timestamp = sorted([n.split("_")[1] for n in checkpoint_paths])[-1]
    checkpoint_paths = [path for path in checkpoint_paths if highest_timestamp in path]

    checkpoint_path = sorted(checkpoint_paths, key=lambda x: int(x.split("_")[2][:-len('.pth.gz')]))[-1]
    print(f"Playing with '{checkpoint_path}'")

    model = NeuralNetwork().to(device)

    print("Loading...")
    with gzip.GzipFile(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Done!")

    model.eval()

    tileset = create_tileset()
    game = Game(tileset, device)

    output_folder = "out/onnx"
    output_name = f"{os.path.basename(checkpoint_path)[:-len('.pth.gz')]}.onnx"
    output_path = os.path.join(output_folder, output_name)

    torch.onnx.export(
        model,                  # model to export
        (game.get_state().unsqueeze(0),),        # inputs of the model,
        output_path,
        input_names=["input"],  # Rename inputs for the ONNX model
        dynamo=True,             # True or False to select the exporter to use
        external_data=False  # File is less than 2GB. Export as one file.
    )