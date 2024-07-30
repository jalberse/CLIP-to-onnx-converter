import clip
import torch

def main():
    import argparse

    # Largely derived from: https://github.com/openai/CLIP/issues/122
    # But, we need to export the image and text encoders separately to be able to implement the CLIP.encode_text() and CLIP.encode_image() functions in Rust.

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
    m.forward(dummy_image,dummy_texts) # Original CLIP result (1)

    # TODO - VIZLIB-37

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

    # Export the Visual model
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

    # Export the Text transformer
    torch.onnx.export(m.transformer, dummy_texts, args.output.replace(".onnx", "_transformer.onnx"), export_params=True,
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

    # Now we must export the information necessary to re-implement the CLIP.encode_text()
    # function from the CLIP python code, in Rust.
    # The token_embedding, however, is handled by the instant-clip-tokenizer crate to my knowledge.
    # That leaves us with:
    # ln_final (LayerNorm).
    # positional_embedding (nn.Parameter)
    # text_projection (nn.Parameter)
    # logit_scale (nn.Parameter)
    # 
    # nn.Parameter is just a special Tensor; we can export those plainly and load
    #  them as an ndarray in Rust with tch.
    # LayerNorm is effectively a single-layer network; I will just save() it and load it
    #  with tch, as I think ONNX is overkill for it (and I'm not sure if it's technically able
    #  to be exported to ONNX).

    torch.save(m.text_encoder.ln_final, args.output.replace(".onnx", "_ln_final.pt"))
    torch.save(m.text_encoder.positional_embedding, args.output.replace(".onnx", "_positional_embedding.pt"))
    torch.save(m.text_encoder.text_projection, args.output.replace(".onnx", "_text_projection.pt"))
    torch.save(m.text_encoder.logit_scale, args.output.replace(".onnx", "_logit_scale.pt"))

    # Now run onnxruntime to verify
    import onnxruntime as ort

    ort_sess = ort.InferenceSession(args.output)
    result=ort_sess.run(["LOGITS_PER_IMAGE", "LOGITS_PER_TEXT"],
      {"IMAGE": dummy_image.numpy(), "TEXT": dummy_texts.numpy()})
    result # verify that result is comparable to (1)

if __name__ == "__main__":
    main()