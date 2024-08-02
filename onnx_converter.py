import clip
import torch
import numpy as np
import onnx
import onnxruntime as ort

from torch import nn

# A wrapper of the CLIP model which just calls CLIP.encode_text() in its forward() function.
# This lets us export the TextModel as an ONNX graph easily.
class TextModel(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, texts):
        return self.clip_model.encode_text(texts)

def main():
    import argparse

    # Largely derived from an answer here: https://github.com/openai/CLIP/issues/122
    # But, we need to export the image and text encoders separately to be able to implement the CLIP.encode_text() and CLIP.encode_image() functions in Rust.
    # I also added actual verification of the resulting graphs.

    parser = argparse.ArgumentParser(description='Convert the CLIP model to ONNX format.')

    # TODO - Note I've only tested with ViT-L/14@336px, but I think the others should work too.
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    parser.add_argument('model', type=str, help='Name of the CLIP model')
    parser.add_argument('output', type=str, help='Path to the output ONNX file')
    args = parser.parse_args()

    m, pre = clip.load(args.model, device="cpu", jit=False)
    npx = m.visual.input_resolution
    dummy_image = torch.randn(10, 3, npx, npx)
    dummy_texts = clip.tokenize(["quick brown fox", "lorem ipsum"])
    print("Model loaded, running forward() to get pytorch results...")
    torch_result = m.forward(dummy_image, dummy_texts)

    print("Exporting the model forward() function to ONNX format...")
    torch.onnx.export(m, (dummy_image, dummy_texts), args.output, export_params=True,
      input_names=["IMAGE", "TEXT"],
      output_names=["LOGITS_PER_IMAGE", "LOGITS_PER_TEXT"],
      opset_version=14,
      dynamic_axes={
          "IMAGE": {
              0: "image_batch_size",
          },
          "TEXT": {
              0: "text_batch_size",
          },
          "LOGITS_PER_IMAGE": {
              0: "image_batch_size",
              1: "text_batch_size",
          },
          "LOGITS_PER_TEXT": {
              0: "text_batch_size",
              1: "image_batch_size",
          },
      }
    )

    print("Loading and verifying the exported model with ONNX API...")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    # Export the Visual model
    print("Exporting the Visual model to ONNX format...")
    torch.onnx.export(m.visual, dummy_image, args.output.replace(".onnx", "_visual.onnx"), export_params=True,
      input_names=["IMAGE"],
      output_names=["FEATURES_EMBEDDED"],
      opset_version=14,
      dynamic_axes={
          "IMAGE": {
              0: "image_batch_size",
          },
          "FEATURES_EMBEDDED": {
              0: "image_batch_size",
          },
      }
    )

    print("Loading and verifying the exported model with ONNX API...")
    onnx_model_visual = onnx.load(args.output.replace(".onnx", "_visual.onnx"))
    onnx.checker.check_model(onnx_model_visual)

    # TODO Rather than exporting the params, embeddings, LayerNorm etc individually, just wrap the encode_text() in our own model and export that.
    #   Then we have it all in one ONNX graph.

    print("Exporting the Text transformer to ONNX format...")
    text_model = TextModel(m)

    # Export the Text transformer
    torch.onnx.export(text_model, dummy_texts, args.output.replace(".onnx", "_transformer.onnx"), export_params=True,
      input_names=["TEXT"],
      output_names=["FEATURES_EMBEDDED"],
      opset_version=14,
      dynamic_axes={
          "TEXT": {
              0: "text_batch_size",
          },
          "FEATURES_EMBEDDED": {
              0: "text_batch_size",
          },
      }
    )

    print("Loading and verifying the exported model with ONNX API...")
    onnx_model_transformer = onnx.load(args.output.replace(".onnx", "_transformer.onnx"))
    onnx.checker.check_model(onnx_model_transformer)

    print("Running the exported forward() model with ONNXRuntime to verify the results...")

    ort_sess = ort.InferenceSession(args.output)
    ort_result=ort_sess.run(["LOGITS_PER_IMAGE", "LOGITS_PER_TEXT"],
      {"IMAGE": dummy_image.numpy(), "TEXT": dummy_texts.numpy()})

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    np.testing.assert_allclose(to_numpy(torch_result[0]), ort_result[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_result[1]), ort_result[1], rtol=1e-03, atol=1e-05)

    # Numpy prints the array if the asset fails, so these aren't needed but it made me feel better
    #   than having it silently pass the first time I ran it.
    # print("Torch out: ", to_numpy(torch_result[0]))
    # print("ORT out: ", ort_result[0])

    # Print the output dimensions for the combined forward() model
    print("Torch out[0] shape: ", to_numpy(torch_result[0]).shape)
    print("Torch out[1] shape: ", to_numpy(torch_result[1]).shape)
    print("ORT out[0] shape: ", ort_result[0].shape)
    print("ORT out[1] shape: ", ort_result[1].shape)

    print("Exported CLIP model (combined forward()) has been tested with ONNXRuntime, and the result looks good!")

    # Similarly test the Visual and Text transformer models using ORT

    # Get the torch results for the visual model.
    torch_result_visual = m.visual(dummy_image)

    print("Running the exported Visual model with ONNXRuntime to verify the results...")
    ort_sess_visual = ort.InferenceSession(args.output.replace(".onnx", "_visual.onnx"))
    ort_result_visual=ort_sess_visual.run(["FEATURES_EMBEDDED"], {"IMAGE": dummy_image.numpy()})
    np.testing.assert_allclose(to_numpy(torch_result_visual), ort_result_visual[0], rtol=1e-03, atol=1e-05)

    # print("Torch out: ", to_numpy(torch_result_visual[0]))
    # print("ORT out: ", ort_result_visual[0])

    # Print the output dimensions for the visual model
    print("Torch out shape: ", to_numpy(torch_result_visual).shape)
    print("ORT out shape: ", ort_result_visual[0].shape)

    print("Exported Visual model has been tested with ONNXRuntime, and the result looks good!")

    # Get the torch results for the Text transformer model.
    torch_result_transformer = text_model(dummy_texts)

    print("Running the exported Text transformer model with ONNXRuntime to verify the results...")
    ort_sess_transformer = ort.InferenceSession(args.output.replace(".onnx", "_transformer.onnx"))
    ort_result_transformer=ort_sess_transformer.run(["FEATURES_EMBEDDED"], {"TEXT": dummy_texts.detach().numpy()})
    np.testing.assert_allclose(to_numpy(torch_result_transformer), ort_result_transformer[0], rtol=1e-03, atol=1e-05)

    # Print the output dimensions for the Text transformer model
    print("Torch out shape: ", to_numpy(torch_result_transformer).shape)
    print("ORT out shape: ", ort_result_transformer[0].shape)

    # print("Torch out: ", to_numpy(torch_result_transformer))
    # print("ORT out: ", ort_result_transformer[0])

    print("Exported Text transformer model has been tested with ONNXRuntime, and the result looks good!")

    print("All tests passed successfully! Check the README for information on using these models; the text transformer requires some addtl. logic in target application")

if __name__ == "__main__":
    main()