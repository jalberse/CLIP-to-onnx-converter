def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert a model to ONNX format')
    parser.add_argument('model', type=str, help='Path to the model file')
    parser.add_argument('output', type=str, help='Path to the output ONNX file')
    args = parser.parse_args()

    # TODO https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
    
