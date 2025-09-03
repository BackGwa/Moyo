using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;

namespace Moyo
{
    public partial class YOLO : IDisposable
    {
        public readonly string[] modelLabels;

        private readonly InferenceSession inferenceSession;
        private readonly ModelType modelType;
        private readonly Size modelResolution;
        private readonly int modelChannels;
        private readonly string inputColumn;
        private readonly string outputColumn;
        private string[] modelOutputs;

        public YOLO(string modelPath, bool useCUDA = false, int cudaDevice = 0)
        {
            // 파일 유효성 점검
            if (!File.Exists(modelPath))
                throw new FileNotFoundException("유효하지 않은 모델 파일입니다. 모델의 경로나 접근 권한을 검토해보세요.", modelPath);

            // 확장자 점검
            if (Path.GetExtension(modelPath).ToLower() != ".onnx")
                throw new FileLoadException("'.onnx' 유형의 모델만 사용할 수 있습니다. 사용하려는 모델을 '.onnx'로 변환하거나, 파일을 점검해보세요.");

            // 추론 세션 설정 
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            options.EnableCpuMemArena = true;
            options.EnableMemoryPattern = true;

            if (useCUDA)
            {
                try
                {
                    options.AppendExecutionProvider_CUDA(cudaDevice);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"CUDA를 할당하는데, 실패하였습니다. CPU를 대체 디바이스로 사용하여 계속 진행하겠습니다.\n상세한 내용은 아래를 참고하세요.\n\n{ex}");
                }
            }

            // 추론 세션 생성
            inferenceSession = new InferenceSession(modelPath, options);
            Metadata metadata = new Metadata(inferenceSession.ModelMetadata);

            // 모델 유형 및 라벨 가져오기
            modelType = metadata.ParseType();
            modelLabels = metadata.ParseLabels();

            // 모델 유형 검증
            if (modelType == ModelType.UnknownModel)
                throw new ArgumentException("알 수 없는 유형의 모델입니다. 모델 유형에 대한 키를 읽지못했거나, 지원하지 않는 모델입니다.");

            // 모델 입출력 열 가져오기
            inputColumn = inferenceSession.InputMetadata.Keys.First();
            outputColumn = inferenceSession.OutputMetadata.Keys.First();

            // 모델 해상도 가져오기
            int modelWidth = inferenceSession.InputMetadata[inputColumn].Dimensions[2];
            int modelHeight = inferenceSession.InputMetadata[inputColumn].Dimensions[3];
            modelResolution = new Size(modelWidth, modelHeight);
            modelChannels = inferenceSession.InputMetadata[inputColumn].Dimensions[1];

            // 모델 출력
            modelOutputs = inferenceSession.OutputMetadata.Keys.ToArray();

            // 모델 워밍업
            Mat warmupImage = new Mat(modelHeight, modelWidth, MatType.CV_8UC3);
            warmupImage.SetTo(new Scalar(0, 0, 0));

            Inference(warmupImage);
        }

        public void Dispose()
        {
            inferenceSession?.Dispose();
        }
    }

    public enum ModelType
    {
        ObjectDetection,
        Segmentation,
        Classification,
        PoseEstimation,
        OrientedBoundingBox,
        UnknownModel
    }
}