using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Collections.Generic;
using System.Linq;

namespace Moyo
{
    public partial class YOLO
    {
        private List<InferenceResult> Classification(ref Mat srcImage)
        {
            Tensor<float> tensor = Preprocessing(ref srcImage);

            NamedOnnxValue[] input = new[] { NamedOnnxValue.CreateFromTensor(inputColumn, tensor) };

            using (var result = inferenceSession.Run(input))
            {
                var outputTensor = result.First(x => x.Name == outputColumn).AsTensor<float>();
                var outputArray = outputTensor.ToArray();

                float maxConfidence = float.MinValue;
                int maxIndex = -1;
                for (int i = 0; i < modelLabels.Length; i++)
                {
                    float confidence = outputArray[i];
                    if (confidence > maxConfidence)
                    {
                        if (confidence > inferenceOptions.confidence)
                        {
                            maxConfidence = confidence;
                            maxIndex = i;
                        }
                    }
                }

                Top1 top1Result = null;
                if (maxIndex != -1)
                {
                    top1Result = new Top1
                    {
                        classes = maxIndex,
                        confidence = maxConfidence
                    };
                }

                var indexedScores = outputArray
                    .Select((confidence, index) => new { Index = index, Confidence = confidence })
                    .OrderByDescending(s => s.Confidence)
                    .Take(5)
                    .ToList();

                var top5Result = new Top5
                {
                    classes = indexedScores.Select(s => s.Index).ToArray(),
                    confidences = indexedScores.Select(s => s.Confidence).ToArray()
                };

                var inferenceResult = new InferenceResult
                {
                    top1 = top1Result,
                    top5 = top5Result
                };

                return new List<InferenceResult> { inferenceResult };
            }
        }
    }
}