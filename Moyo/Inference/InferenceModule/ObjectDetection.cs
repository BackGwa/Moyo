using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace Moyo
{
    public partial class YOLO
    {
        private List<InferenceResult> ObjectDetection(ref Mat srcImage)
        {
            Tensor<float> tensor = Preprocessing(ref srcImage);

            NamedOnnxValue[] input = new[] { NamedOnnxValue.CreateFromTensor(inputColumn, tensor) };
            List<DenseTensor<float>> output;

            using (var result = inferenceSession.Run(input))
            {
                Dictionary<string, DenseTensor<float>> resultDict = result.ToDictionary(x => x.Name, x => x.Value as DenseTensor<float>);
                output = modelOutputs.AsParallel().Select(item => resultDict[item]).ToList();
            }

            Postprocessing postprocess = new Postprocessing(this);
            return postprocess.ObjectDetection(output[0], ref srcImage);
        }
    }
}