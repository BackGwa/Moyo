using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace Moyo
{
    public partial class YOLO
    {
        public partial class Postprocessing
        {
            public unsafe List<InferenceResult> ObjectDetection(DenseTensor<float> predictions, ref Mat srcImage)
            {
                int imageWidth = srcImage.Width;
                int imageHeight = srcImage.Height;

                float xGain = model.modelResolution.Width / (float)imageWidth;
                float yGain = model.modelResolution.Height / (float)imageHeight;
                float xPad = (model.modelResolution.Width - imageWidth * xGain) / 2f;
                float yPad = (model.modelResolution.Height - imageHeight * yGain) / 2f;

                int d0 = predictions.Dimensions[0];
                int d1 = predictions.Dimensions[1];
                int d2 = predictions.Dimensions[2];

                int totalDetections = d2;

                List<InferenceResult> finalPredictions = new List<InferenceResult>(totalDetections * model.modelLabels.Length);
                Memory<float> memory = predictions.Buffer;

                using (var handle = memory.Pin())
                {
                    float* pOut = (float*)handle.Pointer;

                    Parallel.For(
                        fromInclusive: 0,
                        toExclusive: totalDetections,
                        localInit: () => new List<InferenceResult>(),
                        body: (j, loopState, localList) =>
                        {
                            for (int i = 0; i < d0; i++)
                            {
                                int baseIndex = i * (d1 * d2) + j;
                                float cx = pOut[baseIndex + 0 * d2];
                                float cy = pOut[baseIndex + 1 * d2];
                                float w_ = pOut[baseIndex + 2 * d2];
                                float h_ = pOut[baseIndex + 3 * d2];

                                float xMin = ((cx - w_ / 2f) - xPad) / xGain;
                                float yMin = ((cy - h_ / 2f) - yPad) / yGain;
                                float xMax = ((cx + w_ / 2f) - xPad) / xGain;
                                float yMax = ((cy + h_ / 2f) - yPad) / yGain;

                                for (int l = 0; l < model.modelLabels.Length; l++)
                                {
                                    float confidence = pOut[baseIndex + (4 + l) * d2];
                                    if (confidence < model.inferenceOptions.confidence)
                                        continue;

                                    localList.Add(new InferenceResult
                                    {
                                        objectRect = new ObjectRect
                                        {
                                            xMin = Utils.BoundingLimit(xMin, imageWidth),
                                            yMin = Utils.BoundingLimit(yMin, imageHeight),
                                            xMax = Utils.BoundingLimit(xMax, imageWidth - 1),
                                            yMax = Utils.BoundingLimit(yMax, imageHeight - 1)
                                        },
                                        classes = l,
                                        confidence = confidence,
                                        id = 0
                                    });
                                }
                            }
                            return localList;
                        },
                        localFinally: localList =>
                        {
                            lock (finalPredictions)
                            {
                                finalPredictions.AddRange(localList);
                            }
                        }
                    );
                }

                var filtered = NMS(finalPredictions, model.inferenceOptions.iou);
                return filtered.Take(model.inferenceOptions.detectLimit).ToList();
            }

            private List<InferenceResult> NMS(List<InferenceResult> predictions, float threshold)
            {
                var filteredPredictions = new List<InferenceResult>();
                predictions = predictions.AsParallel().OrderByDescending(p => p.confidence).ToList();

                var suppressed = new bool[predictions.Count];
                for (int i = 0; i < predictions.Count; i++)
                {
                    if (suppressed[i]) continue;

                    filteredPredictions.Add(predictions[i]);

                    Parallel.For(i + 1, predictions.Count, j =>
                    {
                        if (Utils.ComputeIoU(predictions[i].objectRect, predictions[j].objectRect) > threshold)
                        {
                            suppressed[j] = true;
                        }
                    });
                }
                return filteredPredictions;
            }
        }
    }
}