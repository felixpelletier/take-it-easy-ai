from main import PolicyNeuralNetwork, Game, create_tileset
from checkpoint_helpers import get_latest_checkpoint_path
import torch
import gzip
import os

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint_path = get_latest_checkpoint_path("takeiteasy", "out")
    print(f"Exporting with '{checkpoint_path}'")

    model = PolicyNeuralNetwork().to(device)

    print("Loading...")
    with gzip.GzipFile(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, weights_only=False)
    model.load_state_dict(checkpoint['policy']['model_state_dict'])
    print("Done!")

    model.eval()

    tileset = create_tileset()
    game = Game(tileset, device)
    
    output_folder = "out/onnx"
    base_name = os.path.basename(checkpoint_path)[:-len('.pth.gz')]
    output_name = f"{base_name}.onnx"
    output_path = os.path.join(output_folder, output_name)

    torch.onnx.export(
        model,                  # model to export
        (game.get_state().unsqueeze(0),),        # inputs of the model,
        output_path,
        input_names=["input"],  # Rename inputs for the ONNX model
        dynamo=True,             # True or False to select the exporter to use
        external_data=False  # File is less than 2GB. Export as one file.
    )

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantized_model = quantize_dynamic(output_path, os.path.join(output_folder, f"{base_name}.q8.onnx"), weight_type=QuantType.QInt8)
    quantized_model = quantize_dynamic(output_path, os.path.join(output_folder, f"{base_name}.q4.onnx"), weight_type=QuantType.QInt4)