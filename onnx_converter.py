import clip
import torch

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert a model to ONNX format')
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    parser.add_argument('model', type=str, help='Name of the CLIP model')
    parser.add_argument('output', type=str, help='Path to the output ONNX file')
    args = parser.parse_args()

    m, pre = clip.load(args.model, device="cpu", jit=False)
    npx = m.visual.input_resolution
    dummy_image = torch.randn(10, 3, npx, npx)
    dummy_texts = clip.tokenize(["quick brown fox", "lorem ipsum"])
    m.forward(dummy_image,dummy_texts) # Original CLIP result (1)

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

    # Now run onnxruntime to verify
    import onnxruntime as ort

    ort_sess = ort.InferenceSession(args.output)
    result=ort_sess.run(["LOGITS_PER_IMAGE", "LOGITS_PER_TEXT"],
      {"IMAGE": dummy_image.numpy(), "TEXT": dummy_texts.numpy()})
    result # verify that result is comparable to (1)

if __name__ == "__main__":
    main()