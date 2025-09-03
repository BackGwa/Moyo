using OpenCvSharp;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Moyo
{
    public partial class YOLO
    {
        private unsafe Tensor<float> Preprocessing(ref Mat srcImage)
        {
            using (Mat resizedImage = new Mat())
            using (Mat normalizeImage = new Mat())
            {
                Cv2.Resize(srcImage, resizedImage, modelResolution);
                Cv2.CvtColor(resizedImage, resizedImage, ColorConversionCodes.BGR2RGB);
                resizedImage.ConvertTo(normalizeImage, MatType.CV_32FC3, 1.0 / 255.0);

                int bufferSize = modelResolution.Width * modelResolution.Height;
                float[] dstArray = new float[1 * modelChannels * bufferSize];
                fixed (float* dstPtr = dstArray)
                {
                    IntPtr dstIntPtr = (IntPtr)dstPtr;
                    float* srcPtr = (float*)normalizeImage.Data;

                    Parallel.For(0, modelResolution.Height, y =>
                    {
                        float* localDstPtr = (float*)dstIntPtr;
                        int rowStart = y * modelResolution.Width;
                        for (int x = 0; x < modelResolution.Width; x++)
                        {
                            int pixelIndex = rowStart + x;
                            int srcIndex = pixelIndex * 3;
                            int dstIndexR = pixelIndex;
                            int dstIndexG = bufferSize + pixelIndex;
                            int dstIndexB = 2 * bufferSize + pixelIndex;

                            localDstPtr[dstIndexR] = srcPtr[srcIndex];
                            localDstPtr[dstIndexG] = srcPtr[srcIndex + 1];
                            localDstPtr[dstIndexB] = srcPtr[srcIndex + 2];
                        }
                    });
                }
                
                return new DenseTensor<float>(dstArray, new[] { 1, modelChannels, modelResolution.Height, modelResolution.Width });
            }
        }
    }
}