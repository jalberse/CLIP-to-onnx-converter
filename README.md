# CLIP to ONNX Converter

Exports CLIP for usage with ONNX.

Note that CLIP has multiple constituent models: namely, the visual net and the text transformer, for CLIP.encode_image() and CLIP.encode_text() respectively. The forward() model function simply calls them both and does some simple operations before a final cosine similarity test.

We would like to be able to use all 3 model functions, since they all have specific use cases.
But you can't just store them in one ONNX, since it will just construct the graph for the CLIP type's forward() function.
But, using the ONNX format for CLIP can be useful to (for example):

* take advantage of ONNX performance
* to simplify distributing and running these models on client machines with different architectures (abstracted under the ONNX runtime).
* and to call the model from another language, such as in Rust with ORT.

So, we'd like to export everything necessary to accomplish these 3 model functions as distinct ONNX graphs + tensor data.
This project provides that export.

(If you only want to run CLIP on a server and you're fine with Python, consider using [clip-as-a-service](https://clip-as-service.jina.ai/index.html) instead. It is the simpler solution)

To replicate CLIP.encode_image(), we can simply export the CLIP.visual VisionTransformer(nn.Module) and its forward() function will be saved to ONNX properly.
To replicate CLIP.encode_text(), things are a bit more complex. The function does not just pass to a forward() call on the Transformer, but makes some additional modifications. These would need to be re-implemented in the target language around calls to the ONNX graph; the various tensors involved are also saved out by this script to be loaded and used in these operations

Note that the forward() model function doesn't truly need its own .ONNX file - you can simply use the other two models, and re-implement the CLIP.forward() function accordingly wherever these models are being used. This saves ~1.6 GB disk space, which is the size of the combined model (for the largest ViT model).

Reference the [CLIP repository](https://github.com/openai/CLIP) to view the encode_image(), encode_text(), and forward() definitions for CLIP.

This project runs the pytorch and ONNX models and compares the result after export to verify the correctness of the resulting graphs.

## Note on RAM

This requires a lot of RAM - set up some swap space. A total of ~42GB is sufficient. If you get a "Killed" message, this is the likely culprit.

## Warning in output about aten::index

ONNX does not have the full opset of PyTorch, so it uses multiple ONNX operators to recreate the op.
In this case, it would break if negative indices are used.
But our verification shows that typical inputs give equivalent results (i.e. we're not using negative indices), so this should be OK.
In the future, perhaps another opset release will obviate this warning.
