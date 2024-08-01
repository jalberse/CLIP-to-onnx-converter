# CLIP to ONNX Converter

Exports CLIP for usage with ONNX.

Note that CLIP has multiple constituent models: namely, the visual net and the text transformer, for CLIP.encode_image() and CLIP.encode_text() respectively. The forward() model function simply calls them both and does some simple comparison operations.

We would like to be able to use all 3 model functions, since they all have specific use cases.
But you can't just store them in one ONNX, since it will just construct the graph for the CLIP type's forward() function.
But I had a use case where I wanted to use ONNX for CLIP (for example, to simplify distributing and running these models on client machines or to call them from another language like Rust, rather than on a server using something like CLIP-as-a-service in python).
So, we'd like to export everything necessary to accomplish these 3 model functions as distinct ONNX graphs + tensor data.
This project provides that export.

Note that the forward() model function doesn't truly need its own .ONNX file - you can simply use the other two models, and re-implement the CLIP.forward() function accordingly. This saves ~1GB disk space, which is the size of the combined model (for the largest ViT model).

To replicate CLIP.encode_image(), we can simply export the CLIP.visual VisionTransformer(nn.Module) and its forward() function will be saved to ONNX properly.
To replicate CLIP.encode_text(), things are a bit more complex. The function does not just pass to a forward() call on the Transformer, but makes some additional modifications. These would need to be re-implemented in the target language; the various tensors involved are also saved out by this script.

## Note on RAM

This requires a lot of RAM - set up some swap space. A total of ~42GB is sufficient. If you get a "Killed" message, this is the likely culprit.

## Warning in output about aten::index

ONNX does not have the full opset of PyTorch, so it uses multiple ONNX operators.
In this case, it would break if negative indices are used.
But our verification shows that typical inputs give equivalent results (i.e. we're not using negative indices), so this should be OK.
In the future, perhaps another opset release will obviate this warning.