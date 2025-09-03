using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace Moyo
{
    public partial class YOLO
    {
        public InferenceOptions inferenceOptions = InferenceOptions.DEFAULT;

        public List<InferenceResult> Inference(Mat srcImage, InferenceOptions? options = null)
        {
            return _Inference(ref srcImage, options);
        }

        public List<InferenceResult> Inference(string srcImagePath, InferenceOptions? options = null)
        {
            Mat srcImage = Cv2.ImRead(srcImagePath);
            return _Inference(ref srcImage, options);
        }

        private List<InferenceResult> _Inference(ref Mat srcImage, InferenceOptions? options)
        {
            if (options == null)
                inferenceOptions = InferenceOptions.DEFAULT;
            else
                inferenceOptions = options;

            List<DenseTensor<float>> predictions;

            switch (modelType)
            {
                case ModelType.ObjectDetection:
                    return ObjectDetection(ref srcImage);
                case ModelType.Classification:
                    return Classification(ref srcImage);
                case ModelType.Segmentation:
                    return null;
                case ModelType.PoseEstimation:
                    return null;
                case ModelType.OrientedBoundingBox:
                    return null;
                default:
                    return new List<InferenceResult>();
            }
        }
    }
}