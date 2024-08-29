# CLIP to ONNX Converter

Exports CLIP for usage with ONNX.

Using the ONNX format for CLIP can be useful to (for example):

* take advantage of ONNX potential performance gains across a wide range of hardware configurations,
* to simplify distributing and running these models on client machines with different architectures (abstracted under the ONNX runtime),
* and to call the model from another language, such as in Rust with ORT (though it can be used anywhere the ONNX runtime is supported).

Note that CLIP has multiple constituent models: namely, the visual net and the text transformer, for CLIP.encode_image() and CLIP.encode_text() respectively. The forward() model function simply calls them both and does some simple operations before a final cosine similarity test.
We would like to be able to use all 3 model functions, since they all have specific use cases.
But you can't just store them in one ONNX, since it will just construct the graph for the CLIP type's forward() function.

So, we'd like to export everything necessary to accomplish these 3 model functions as distinct ONNX graphs + tensor data.
This project provides that export.

Three ONNX files are output. The base ONNX (forward()), *_transformer.onnx (encode_text()), and *_visual.onnx (encode_image()).

(If you only want to run CLIP on a server and you're fine with Python, consider using [clip-as-a-service](https://clip-as-service.jina.ai/index.html) instead. It is the simpler solution)

Note that the forward() model function doesn't truly need its own .ONNX file - you can simply use the other two models, and re-implement the CLIP.forward() function accordingly wherever these models are being used, since it's just a simple combination of both of those with some additional tensor operations. This saves ~1.6 GB disk space, which is the size of the combined model (for the largest ViT model). But, you need to re-implement it. Decide what's best for your project. Know that holding these models in VRAM can be expensive.

Reference the [CLIP repository](https://github.com/openai/CLIP) to view the encode_image(), encode_text(), and forward() definitions for CLIP.

This project runs the pytorch and ONNX models and compares the result after export to verify the correctness of the resulting graphs.

## Note on RAM

This requires a lot of RAM - set up some swap space if necessary. A total of ~42GB is sufficient. If you get a "Killed" message, this is the likely culprit.

## Warning in output about aten::index

ONNX does not have the full opset of PyTorch, so it uses multiple ONNX operators to recreate the op.
In this case, it would break if negative indices are used.
But our verification shows that typical inputs give equivalent results (i.e. we're not using negative indices), so this should be OK.
In the future, perhaps another opset release will obviate this warning.

## Note on stability

I'm still using this script and might change it to suit my needs (e.g. by adding options for using f16 vs f32 precision). Provided without warranty etc etc.
